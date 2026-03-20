"""
Repair irregular timestamp gaps in hourly ETH metrics data.

This script produces two repaired versions of the same input CSV:

1. average
   Linear interpolation across numeric columns on a strict hourly grid.
   This is simple, but it can smooth away real volatility.

2. conservative
   A market-aware fill intended for trading features:
   - close / btc_close: forward-fill
   - Open / btc_open: previous repaired close
   - high / btc_high: max(open, close)
   - low / btc_low: min(open, close)
   - volume / btc_volume: 0 for imputed rows
   - all other numeric columns: forward-fill / backfill

The output timestamps are always written back as Unix seconds because the
training pipeline expects that format.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/metrics/eth_metrics_combined.csv")
DEFAULT_OUTPUT_DIR = Path("data/gap_repair")
ONE_HOUR = pd.Timedelta(hours=1)

IDENTITY_COLS: Tuple[str, ...] = ("ticker", "caller")
ETH_PRICE_COLS: Tuple[str, ...] = ("Open", "high", "low", "close", "volume")
BTC_PRICE_COLS: Tuple[str, ...] = (
    "btc_open",
    "btc_high",
    "btc_low",
    "btc_close",
    "btc_volume",
)


@dataclass(frozen=True)
class RepairSummary:
    method: str
    input_rows: int
    trimmed_rows: int
    output_rows: int
    inserted_rows: int
    original_gap_count: int
    trim_before: Optional[str]
    output_path: Path


def _normalize_numeric_timestamps(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    abs_values = values.abs()
    out = values.copy()

    out.loc[abs_values >= 1e18] = out.loc[abs_values >= 1e18] / 1e9
    mask_ns = (abs_values >= 1e14) & (abs_values < 1e18)
    out.loc[mask_ns] = out.loc[mask_ns] / 1e9
    mask_ms = (abs_values >= 1e12) & (abs_values < 1e14)
    out.loc[mask_ms] = out.loc[mask_ms] / 1e3
    mask_short = (abs_values >= 1e6) & (abs_values < 1e9)
    out.loc[mask_short] = out.loc[mask_short] * 1e3

    return out


def _parse_timestamp_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    numeric_mask = numeric.notna()
    if numeric_mask.any():
        normalized = _normalize_numeric_timestamps(numeric.loc[numeric_mask])
        parsed.loc[numeric_mask] = pd.to_datetime(
            normalized, unit="s", utc=True, errors="coerce"
        )

    text_mask = ~numeric_mask
    if text_mask.any():
        parsed.loc[text_mask] = pd.to_datetime(
            series.loc[text_mask], utc=True, errors="coerce"
        )

    return parsed


def _load_frame(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Input CSV is missing a 'timestamp' column: {csv_path}")
    if df.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    df = df.copy()
    df["timestamp"] = _parse_timestamp_column(df["timestamp"])
    if df["timestamp"].isna().any():
        bad_count = int(df["timestamp"].isna().sum())
        raise ValueError(f"Found {bad_count} invalid timestamps in {csv_path}")

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.reset_index(drop=True)
    return df


def _count_non_hourly_gaps(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 0
    diffs = pd.Series(index).diff().dropna()
    return int((diffs != ONE_HOUR).sum())


def _build_hourly_grid(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    indexed = df.set_index("timestamp").sort_index()
    full_index = pd.date_range(
        start=indexed.index.min(),
        end=indexed.index.max(),
        freq="1h",
        tz="UTC",
    )
    reindexed = indexed.reindex(full_index)
    imputed_mask = reindexed.index.to_series().map(lambda ts: ts not in indexed.index)
    reindexed.index.name = "timestamp"
    return reindexed, imputed_mask


def _fill_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in IDENTITY_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    return df


def _numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> List[str]:
    exclude_set = set(exclude)
    cols: List[str] = []
    for col in df.columns:
        if col in exclude_set:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().any() or pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _recompute_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    if {"market_cap", "close", "supply"}.issubset(df.columns):
        close = pd.to_numeric(df["close"], errors="coerce")
        supply = pd.to_numeric(df["supply"], errors="coerce")
        df["market_cap"] = close * supply
    return df


def _repair_average(df: pd.DataFrame, imputed_mask: pd.Series) -> pd.DataFrame:
    repaired = df.copy()
    repaired = _fill_identity_columns(repaired)

    numeric_cols = _numeric_columns(repaired, exclude=IDENTITY_COLS)
    for col in numeric_cols:
        repaired[col] = pd.to_numeric(repaired[col], errors="coerce")
    if numeric_cols:
        repaired[numeric_cols] = repaired[numeric_cols].interpolate(
            method="time",
            limit_direction="both",
        )

    for col in repaired.columns:
        if col in numeric_cols or col in IDENTITY_COLS:
            continue
        repaired[col] = repaired[col].ffill().bfill()

    repaired = _recompute_market_cap(repaired)
    return repaired


def _apply_price_fill(
    repaired: pd.DataFrame,
    imputed_mask: pd.Series,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
) -> pd.DataFrame:
    if close_col in repaired.columns:
        repaired[close_col] = pd.to_numeric(repaired[close_col], errors="coerce")
        repaired[close_col] = repaired[close_col].ffill().bfill()

    if open_col in repaired.columns and close_col in repaired.columns:
        repaired[open_col] = pd.to_numeric(repaired[open_col], errors="coerce")
        prev_close = repaired[close_col].shift(1).fillna(repaired[close_col])
        repaired.loc[imputed_mask, open_col] = prev_close.loc[imputed_mask]
        repaired[open_col] = repaired[open_col].ffill().bfill()

    if high_col in repaired.columns and open_col in repaired.columns and close_col in repaired.columns:
        repaired[high_col] = pd.to_numeric(repaired[high_col], errors="coerce")
        replacement = np.maximum(repaired[open_col], repaired[close_col])
        repaired.loc[imputed_mask, high_col] = replacement.loc[imputed_mask]
        repaired[high_col] = repaired[high_col].ffill().bfill()

    if low_col in repaired.columns and open_col in repaired.columns and close_col in repaired.columns:
        repaired[low_col] = pd.to_numeric(repaired[low_col], errors="coerce")
        replacement = np.minimum(repaired[open_col], repaired[close_col])
        repaired.loc[imputed_mask, low_col] = replacement.loc[imputed_mask]
        repaired[low_col] = repaired[low_col].ffill().bfill()

    if volume_col in repaired.columns:
        repaired[volume_col] = pd.to_numeric(repaired[volume_col], errors="coerce")
        repaired.loc[imputed_mask, volume_col] = 0.0
        repaired[volume_col] = repaired[volume_col].fillna(0.0)

    return repaired


def _repair_conservative(df: pd.DataFrame, imputed_mask: pd.Series) -> pd.DataFrame:
    repaired = df.copy()
    repaired = _fill_identity_columns(repaired)

    hold_cols = set(ETH_PRICE_COLS) | set(BTC_PRICE_COLS) | {"market_cap"}
    numeric_cols = _numeric_columns(repaired, exclude=IDENTITY_COLS)
    for col in numeric_cols:
        repaired[col] = pd.to_numeric(repaired[col], errors="coerce")

    passive_cols = [col for col in numeric_cols if col not in hold_cols]
    for col in passive_cols:
        repaired[col] = repaired[col].ffill().bfill()

    repaired = _apply_price_fill(
        repaired,
        imputed_mask,
        open_col="Open",
        high_col="high",
        low_col="low",
        close_col="close",
        volume_col="volume",
    )
    repaired = _apply_price_fill(
        repaired,
        imputed_mask,
        open_col="btc_open",
        high_col="btc_high",
        low_col="btc_low",
        close_col="btc_close",
        volume_col="btc_volume",
    )
    repaired = _recompute_market_cap(repaired)
    return repaired


def _finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].astype("int64") // 1_000_000_000
    return out


def _repair(
    df: pd.DataFrame,
    method: str,
    trim_before: Optional[pd.Timestamp],
) -> Tuple[pd.DataFrame, int, int, int]:
    working = df.copy()
    input_rows = len(working)

    if trim_before is not None:
        working = working[working["timestamp"] >= trim_before].copy()
        if working.empty:
            raise ValueError(
                f"trim-before={trim_before.isoformat()} removed every row from the dataset."
            )

    trimmed_rows = len(working)
    original_gap_count = _count_non_hourly_gaps(pd.DatetimeIndex(working["timestamp"]))
    reindexed, imputed_mask = _build_hourly_grid(working)

    if method == "average":
        repaired = _repair_average(reindexed, imputed_mask)
    elif method == "conservative":
        repaired = _repair_conservative(reindexed, imputed_mask)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return _finalize_output(repaired), input_rows, trimmed_rows, original_gap_count


def _validate_hourly(df: pd.DataFrame) -> None:
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    diffs = ts.diff().dropna()
    if (diffs != ONE_HOUR).any():
        raise RuntimeError("Repaired output is not strictly hourly.")


def _resolve_trim_before(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _output_path(output_dir: Path, input_csv: Path, method: str) -> Path:
    stem = input_csv.stem
    return output_dir / f"{stem}_{method}_fill.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair non-hourly gaps in a CSV by writing strict-hourly outputs "
            "for both average interpolation and conservative OHLCV fill."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT),
        help="Input CSV with a timestamp column. Default: data/metrics/eth_metrics_combined.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for repaired CSV outputs.",
    )
    parser.add_argument(
        "--trim-before",
        default=None,
        help=(
            "Optional UTC timestamp. Rows earlier than this are dropped before repair. "
            "Example: 2016-05-26T00:00:00Z"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["average", "conservative"],
        choices=["average", "conservative"],
        help="One or both repair methods to run.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trim_before = _resolve_trim_before(args.trim_before)

    df = _load_frame(input_csv)

    summaries: List[RepairSummary] = []
    for method in args.methods:
        repaired, input_rows, trimmed_rows, original_gap_count = _repair(
            df=df,
            method=method,
            trim_before=trim_before,
        )
        _validate_hourly(repaired)
        output_path = _output_path(output_dir, input_csv, method)
        repaired.to_csv(output_path, index=False)

        summaries.append(
            RepairSummary(
                method=method,
                input_rows=input_rows,
                trimmed_rows=trimmed_rows,
                output_rows=len(repaired),
                inserted_rows=len(repaired) - trimmed_rows,
                original_gap_count=original_gap_count,
                trim_before=trim_before.isoformat() if trim_before is not None else None,
                output_path=output_path,
            )
        )

    for summary in summaries:
        print(
            f"[{summary.method}] wrote {summary.output_path} | "
            f"input_rows={summary.input_rows} | "
            f"trimmed_rows={summary.trimmed_rows} | "
            f"inserted_rows={summary.inserted_rows} | "
            f"output_rows={summary.output_rows} | "
            f"original_gap_count={summary.original_gap_count} | "
            f"trim_before={summary.trim_before}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
