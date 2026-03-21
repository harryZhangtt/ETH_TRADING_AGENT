from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from augment import FeatureGroups
from envs import TradingEnv, TradingEnvConfig


PRICE_FEATURE_NAMES: Tuple[str, ...] = (
    "eth_log_open_to_open_ret_24h",
    "eth_log_open_to_open_ret_7d",
    "eth_log_high_low_range_1h",
    "btc_log_open_to_open_ret_24h",
    "btc_log_open_to_open_ret_7d",
    "btc_log_high_low_range_1h",
)

VOLUME_FEATURE_NAMES: Tuple[str, ...] = (
    "eth_log1p_volume_1h",
    "eth_dlog1p_volume_1h",
    "btc_log1p_volume_1h",
    "btc_dlog1p_volume_1h",
)

ONCHAIN_FEATURE_NAMES: Tuple[str, ...] = (
    "log_supply_change",
    "log_daily_txn",
    "log_unique_addr",
    "log_unique_addr_change",
    "log_avg_txn_fee",
    "log_avg_txn_fee_change",
    "log_avg_block_size",
    "log_avg_block_size_change",
    "log_daily_token_transfer",
    "log_daily_token_transfer_change",
)

DEFAULT_FEATURE_COLS: Tuple[str, ...] = (
    *PRICE_FEATURE_NAMES,
    *VOLUME_FEATURE_NAMES,
    *ONCHAIN_FEATURE_NAMES,
)


@dataclass(frozen=True)
class DataPipelineConfig:
    csv_path: str
    timestamp_col: str = "timestamp"

    history_days: int = 60
    sigma_days: int = 20
    multi_day_return_days: int = 7

    rows_per_day: int = 24

    feature_cols: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def _check_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _fit_standardizer(x_train: np.ndarray) -> Standardizer:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return Standardizer(mean=mean, std=std)


def _safe_log_ratio(num: pd.Series, den: pd.Series, eps: float = 1e-12) -> pd.Series:
    num = num.astype(np.float64).clip(lower=eps)
    den = den.astype(np.float64).clip(lower=eps)
    return np.log(num / den)


def _effective_window_rows(cfg: DataPipelineConfig) -> int:
    return cfg.history_days * cfg.rows_per_day


def _effective_sigma_rows(cfg: DataPipelineConfig) -> int:
    return cfg.sigma_days * cfg.rows_per_day


def _effective_multi_day_rows(cfg: DataPipelineConfig) -> int:
    return cfg.multi_day_return_days * cfg.rows_per_day


def _build_feature_groups(feature_cols: Sequence[str]) -> FeatureGroups:
    price_like = tuple(
        i for i, c in enumerate(feature_cols) if c in PRICE_FEATURE_NAMES
    )
    volume_like = tuple(
        i for i, c in enumerate(feature_cols) if c in VOLUME_FEATURE_NAMES
    )
    onchain_like = tuple(
        i for i, c in enumerate(feature_cols) if c in ONCHAIN_FEATURE_NAMES
    )

    return FeatureGroups(
        price_like=price_like,
        volume_like=volume_like,
        onchain_like=onchain_like,
        never_mask=(),
    )


def _load_csv(cfg: DataPipelineConfig) -> pd.DataFrame:
    csv_path = Path(cfg.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if cfg.timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {cfg.timestamp_col}")

    drop_cols = [c for c in ("ticker", "caller") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    ts_numeric = pd.to_numeric(df[cfg.timestamp_col], errors="raise")
    df = df.copy()
    df[cfg.timestamp_col] = pd.to_datetime(
        ts_numeric, unit="s", utc=True, errors="raise"
    )

    df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)

    if df[cfg.timestamp_col].duplicated().any():
        dup_count = int(df[cfg.timestamp_col].duplicated().sum())
        raise ValueError(
            f"Found {dup_count} duplicate timestamps after loading. "
            "Please deduplicate before training."
        )

    if len(df) == 0:
        raise ValueError("CSV is empty after loading.")

    diffs = df[cfg.timestamp_col].diff().dropna()
    if len(diffs) == 0:
        raise ValueError("CSV must contain at least 2 rows.")

    bad_steps = diffs != pd.Timedelta(hours=1)
    if bad_steps.any():
        bad_count = int(bad_steps.sum())
        example_steps = diffs[bad_steps].astype(str).value_counts().head(10).to_dict()
        raise ValueError(
            f"Timestamp grid is not strictly hourly. "
            f"Found {bad_count} non-1h gaps. Examples: {example_steps}"
        )

    return df


def _add_hourly_features(df: pd.DataFrame, cfg: DataPipelineConfig) -> pd.DataFrame:
    required_raw = [
        "Open",
        "high",
        "low",
        "volume",
        "btc_open",
        "btc_high",
        "btc_low",
        "btc_volume",
        *ONCHAIN_FEATURE_NAMES,
    ]
    _check_columns(df, required_raw)

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in required_raw:
        df[col] = pd.to_numeric(df[col], errors="raise")

    one_day_rows = cfg.rows_per_day
    multi_day_rows = _effective_multi_day_rows(cfg)
    sigma_rows = _effective_sigma_rows(cfg)

    df["_eth_log_open_to_open_ret_1h"] = _safe_log_ratio(
        df["Open"], df["Open"].shift(1)
    )

    df["eth_log_open_to_open_ret_24h"] = _safe_log_ratio(
        df["Open"], df["Open"].shift(one_day_rows)
    )
    df["eth_log_open_to_open_ret_7d"] = _safe_log_ratio(
        df["Open"], df["Open"].shift(multi_day_rows)
    )
    df["eth_log_high_low_range_1h"] = _safe_log_ratio(df["high"], df["low"])

    df["btc_log_open_to_open_ret_24h"] = _safe_log_ratio(
        df["btc_open"], df["btc_open"].shift(one_day_rows)
    )
    df["btc_log_open_to_open_ret_7d"] = _safe_log_ratio(
        df["btc_open"], df["btc_open"].shift(multi_day_rows)
    )
    df["btc_log_high_low_range_1h"] = _safe_log_ratio(df["btc_high"], df["btc_low"])

    df["eth_log1p_volume_1h"] = np.log1p(
        df["volume"].astype(np.float64).clip(lower=0.0)
    )
    df["eth_dlog1p_volume_1h"] = df["eth_log1p_volume_1h"].diff()

    df["btc_log1p_volume_1h"] = np.log1p(
        df["btc_volume"].astype(np.float64).clip(lower=0.0)
    )
    df["btc_dlog1p_volume_1h"] = df["btc_log1p_volume_1h"].diff()

    df["_target_next_hour_open_to_open_log_return"] = _safe_log_ratio(
        df["Open"].shift(-2),
        df["Open"].shift(-1),
    )
    df["_rolling_sigma_hourly"] = (
        df["_eth_log_open_to_open_ret_1h"]
        .rolling(sigma_rows, min_periods=sigma_rows)
        .std()
    )

    for col in ONCHAIN_FEATURE_NAMES:
        df[col] = df[col].shift(one_day_rows)

    return df


def _prepare_actionable_table(
    df: pd.DataFrame,
    cfg: DataPipelineConfig,
) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    feature_cols = (
        cfg.feature_cols if cfg.feature_cols is not None else DEFAULT_FEATURE_COLS
    )
    feature_cols = tuple(feature_cols)

    required = [
        cfg.timestamp_col,
        "_target_next_hour_open_to_open_log_return",
        "_rolling_sigma_hourly",
        *feature_cols,
    ]
    _check_columns(df, required)

    out = df[
        [
            cfg.timestamp_col,
            "_target_next_hour_open_to_open_log_return",
            "_rolling_sigma_hourly",
            *feature_cols,
        ]
    ].copy()
    out = out.dropna().reset_index(drop=True)

    min_required = _effective_window_rows(cfg) + 1
    if len(out) < min_required:
        raise ValueError(
            f"Not enough actionable rows after preprocessing: len={len(out)}. "
            f"Need at least {min_required}."
        )

    return out, feature_cols


def _make_segment_arrays(
    df: pd.DataFrame,
    core_start: int,
    core_end: int,
    feature_cols: Sequence[str],
    scaler: Standardizer,
    window_rows: int,
    timestamp_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any], Dict[str, int]]:
    context_start = max(0, core_start - (window_rows - 1))
    seg = df.iloc[context_start:core_end].copy()

    if len(seg) < window_rows:
        raise ValueError(
            f"Segment too short: len={len(seg)}, window_rows={window_rows}"
        )

    x_raw = seg[list(feature_cols)].to_numpy(dtype=np.float32)
    x = scaler.transform(x_raw).astype(np.float32)

    r_mkt_log = seg["_target_next_hour_open_to_open_log_return"].to_numpy(
        dtype=np.float32
    )
    sigma = seg["_rolling_sigma_hourly"].to_numpy(dtype=np.float32)
    dates = seg[timestamp_col].tolist()

    meta = {
        "context_start": context_start,
        "core_start": core_start,
        "core_end": core_end,
        "segment_len": len(seg),
        "first_actionable_abs_row": context_start + (window_rows - 1),
        "last_actionable_abs_row": context_start + (len(seg) - 1),
    }

    return x, r_mkt_log, sigma, dates, meta


# ── Public API ────────────────────────────────────────────────────────────────

def load_actionable_df(cfg: DataPipelineConfig) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    """Load CSV, engineer features, and return the cleaned actionable DataFrame.

    Responsibilities end here — no splitting, no scaling, no env construction.
    The caller (e.g. train.py) owns all of that.

    Returns
    -------
    df           : actionable DataFrame with NaN rows dropped
    feature_cols : ordered tuple of feature column names
    """
    raw_df = _load_csv(cfg)
    feat_df = _add_hourly_features(raw_df, cfg)
    return _prepare_actionable_table(feat_df, cfg)


def fit_standardizer(
    df: pd.DataFrame,
    train_start: int,
    train_end: int,
    feature_cols: Sequence[str],
) -> Standardizer:
    """Fit a Standardizer on the training slice only."""
    x_train = df.iloc[train_start:train_end][list(feature_cols)].to_numpy(
        dtype=np.float32
    )
    return _fit_standardizer(x_train)


def make_env_from_slice(
    df: pd.DataFrame,
    core_start: int,
    core_end: int,
    feature_cols: Sequence[str],
    scaler: Standardizer,
    env_cfg: TradingEnvConfig,
    window_rows: int,
    timestamp_col: str = "timestamp",
) -> TradingEnv:
    """Build a TradingEnv from a row slice of the actionable DataFrame.

    The slice [core_start, core_end) defines the *actionable* rows; the
    function prepends window_rows-1 context rows automatically so the first
    observation is valid.
    """
    x, r_mkt_log, sigma, dates, _ = _make_segment_arrays(
        df=df,
        core_start=core_start,
        core_end=core_end,
        feature_cols=feature_cols,
        scaler=scaler,
        window_rows=window_rows,
        timestamp_col=timestamp_col,
    )
    return TradingEnv(X=x, r_mkt_log=r_mkt_log, sigma=sigma, dates=dates, config=env_cfg)


def build_feature_groups(feature_cols: Sequence[str]) -> FeatureGroups:
    """Public wrapper — build FeatureGroups from an ordered feature list."""
    return _build_feature_groups(feature_cols)
