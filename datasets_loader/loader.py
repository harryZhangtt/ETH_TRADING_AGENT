"""
datasets_loader/loader.py
=========================
Loads ``data/midway/eth_metrics_midway.csv`` — which already contains every
derived feature — and builds the composite dataset for the PPO training pipeline.

Pipeline (no feature engineering needed)
-----------------------------------------
1. _load_midway   – reads the pre-computed midway CSV, parses timestamps
2. _validate      – null-fraction checks per SEQ_FEATURE_COLS column
3. _build_windows – strided sliding-window view  →  float32 [N, T, F]
4. _build_port    – placeholder portfolio state  →  float32 [N, P]

Storage: .npz (compressed numpy archive)
-----------------------------------------
Chosen because it is framework-agnostic, stores named arrays + metadata in one
file, has no extra dependencies, and converts to torch in one line:
    x = torch.from_numpy(np.load(path)["x_seq"])
For datasets that exceed ~1 GB, swap save/load for an HDF5 backend with the
same public interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .schema import SEQ_FEATURE_COLS, XPortSchema, XSeqSchema
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from datasets_loader.schema import SEQ_FEATURE_COLS, XPortSchema, XSeqSchema

log = logging.getLogger(__name__)


# ── Dataset configuration ─────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetConfig:
    """All parameters that control dataset construction.

    Attributes
    ----------
    midway_csv:
        Path to ``eth_metrics_midway.csv`` — the pre-computed feature source.
    window_size:
        Number of hourly time steps per sample.  Must match
        ``ModelConfig.max_seq_len`` (default 60).
    step_size:
        Sliding-window stride.
        1  = fully overlapping (maximum data, slower training).
        24 = one window per day (less overlap, faster training).
    output_path:
        Destination for the persisted dataset (.npz required).
    null_warn_threshold:
        Log WARNING for any feature column whose NaN fraction exceeds this.
    null_fail_threshold:
        Raise ValueError when a column's NaN fraction exceeds this — signals
        a corrupted or truncated midway CSV.
    """

    midway_csv: str = "data/midway/eth_metrics_midway.csv"
    window_size: int = 60
    step_size: int = 1
    output_path: str = "data/dataset_loader/eth_dataset.npz"
    null_warn_threshold: float = 0.30
    null_fail_threshold: float = 1.0   # hard-fail only on completely empty columns


# ── Result container ──────────────────────────────────────────────────────

@dataclass
class EthDataset:
    """Fully-processed arrays ready for the training loop.

    Shapes
    ------
    x_seq         : float32 [N, window_size, seq_feature_dim]
    x_port        : float32 [N, port_dim]   — zeros until portfolio state is designed
    timestamps    : int64   [N]             — Unix-second of the LAST step in each window
    feature_names : object  [seq_feature_dim] — string labels (for logging / indexing)

    Converting to PyTorch tensors
    -----------------------------
        import torch, numpy as np
        data = np.load("eth_dataset.npz", allow_pickle=True)
        x_seq  = torch.from_numpy(data["x_seq"])   # float32 [N, T, F]
        x_port = torch.from_numpy(data["x_port"])  # float32 [N, P]
    """

    x_seq: np.ndarray           # [N, T, F]
    x_port: np.ndarray          # [N, P]
    timestamps: np.ndarray      # [N]
    feature_names: np.ndarray   # [F]  dtype=object (strings)

    # ── post-init validation ──────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.x_seq.ndim != 3:
            raise ValueError(
                f"x_seq must be 3-D [N, T, F], got shape {self.x_seq.shape}"
            )
        if self.x_port.ndim != 2:
            raise ValueError(
                f"x_port must be 2-D [N, P], got shape {self.x_port.shape}"
            )
        n = len(self.x_seq)
        if len(self.x_port) != n or len(self.timestamps) != n:
            raise ValueError(
                f"Leading dimension mismatch: x_seq={n}, "
                f"x_port={len(self.x_port)}, timestamps={len(self.timestamps)}"
            )
        if self.x_seq.shape[2] != len(self.feature_names):
            raise ValueError(
                f"x_seq feature dim ({self.x_seq.shape[2]}) != "
                f"len(feature_names) ({len(self.feature_names)})"
            )

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> Path:
        """Persist to a compressed numpy archive (.npz).

        Raises
        ------
        ValueError       if *path* is None or doesn't end with ``.npz``.
        """
        if path is None:
            raise ValueError("path must be provided to EthDataset.save()")
        p = Path(path)
        if p.suffix != ".npz":
            raise ValueError(f"output_path must end in .npz, got: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            x_seq=self.x_seq,
            x_port=self.x_port,
            timestamps=self.timestamps,
            feature_names=self.feature_names,
        )
        log.info(
            "Dataset saved → %s  [N=%d, T=%d, F=%d]",
            p, len(self.x_seq), self.x_seq.shape[1], self.x_seq.shape[2],
        )
        return p

    @classmethod
    def load(cls, path: str) -> "EthDataset":
        """Load a previously saved .npz dataset.

        Raises
        ------
        FileNotFoundError   if the file does not exist.
        ValueError          if required keys are missing.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        data = np.load(p, allow_pickle=True)
        required = {"x_seq", "x_port", "timestamps", "feature_names"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Dataset at {p} is corrupt — missing keys: {missing}")
        return cls(
            x_seq=data["x_seq"],
            x_port=data["x_port"],
            timestamps=data["timestamps"],
            feature_names=data["feature_names"],
        )

    def __repr__(self) -> str:
        return (
            f"EthDataset(windows={len(self.x_seq)}, "
            f"window_size={self.x_seq.shape[1]}, "
            f"feature_dim={self.x_seq.shape[2]}, "
            f"port_dim={self.x_port.shape[1]})"
        )


# ── Main loader ───────────────────────────────────────────────────────────

class EthDatasetLoader:
    """Builds an EthDataset from the pre-computed midway CSV.

    Usage
    -----
        cfg = DatasetConfig(window_size=60)
        dataset = EthDatasetLoader(cfg).build()
        dataset.save(cfg.output_path)

    Extending
    ---------
    * New feature column → add to SEQ_FEATURE_COLS in schema.py and ensure
      the midway CSV contains it.  No loader code changes required.
    * Portfolio state   → update XPortSchema.feature_cols in schema.py and
      replace _build_port below.
    * Window size       → change DatasetConfig.window_size.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.cfg = config
        self._x_schema = XSeqSchema()
        self._p_schema = XPortSchema()

    # ── Public API ────────────────────────────────────────────────────────

    def build(self) -> EthDataset:
        """Full pipeline: load → validate → window → return."""
        df = self._load_midway()
        df = self._validate(df)
        x_seq, timestamps = self._build_windows(df)
        x_port = self._build_port(len(x_seq))
        return EthDataset(
            x_seq=x_seq,
            x_port=x_port,
            timestamps=timestamps,
            feature_names=np.array(list(self._x_schema.feature_cols), dtype=object),
        )

    # ── Step 1: load ──────────────────────────────────────────────────────

    def _load_midway(self) -> pd.DataFrame:
        """Read the midway CSV and normalise the timestamp column to UTC datetime."""
        path = Path(self.cfg.midway_csv)
        if not path.exists():
            raise FileNotFoundError(
                f"Midway CSV not found: {path!r}. "
                "Generate it via the metrics pipeline before calling the loader."
            )

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"Midway CSV is empty: {path!r}")

        if "timestamp" not in df.columns:
            raise KeyError(
                f"Midway CSV is missing the required 'timestamp' column: {path!r}"
            )

        # Timestamps are stored as Unix seconds (int); parse to UTC datetime
        # so window extraction can reconstruct them faithfully later.
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Verify that every SEQ_FEATURE_COL is present before going further
        missing = [c for c in SEQ_FEATURE_COLS if c not in df.columns]
        if missing:
            raise KeyError(
                f"Midway CSV is missing expected feature columns: {missing}. "
                "Re-run the midway generation pipeline."
            )

        log.info(
            "Midway CSV loaded: %d rows × %d columns from %s",
            len(df), len(df.columns), path.name,
        )
        return df

    # ── Step 2: validate ──────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check null fractions per SEQ_FEATURE_COLS column.

        Raises  ValueError  for any column above null_fail_threshold.
        Logs    WARNING     for columns between warn and fail thresholds.
        """
        null_ratios = df[list(SEQ_FEATURE_COLS)].isnull().mean()

        # 100% null → data source not yet in the midway CSV.
        # Warn loudly so the gap is visible; imputation fills with 0.0.
        completely_empty = null_ratios[null_ratios >= 1.0]
        for col in completely_empty.index:
            log.warning(
                "Column '%s' is 100%% null — its data source is not in the midway CSV "
                "and will be imputed with 0.0 during windowing.",
                col,
            )

        # Partially null above the hard-fail threshold signals a corrupt CSV.
        hard_fail = null_ratios[
            (null_ratios > self.cfg.null_fail_threshold) & (null_ratios < 1.0)
        ]
        if not hard_fail.empty:
            raise ValueError(
                f"Features exceed {self.cfg.null_fail_threshold:.0%} null threshold "
                "(corrupted or truncated midway CSV):\n"
                + "\n".join(f"  {col}: {r:.1%}" for col, r in hard_fail.items())
            )

        soft_warn = null_ratios[
            (null_ratios > self.cfg.null_warn_threshold)
            & (null_ratios <= self.cfg.null_fail_threshold)
        ]
        for col, ratio in soft_warn.items():
            log.warning("High null ratio %.1f%% in feature '%s'", ratio * 100, col)

        total_null_rows = df[list(SEQ_FEATURE_COLS)].isnull().any(axis=1).sum()
        log.info(
            "Validation passed: %d rows, %d (%.1f%%) contain at least one NaN",
            len(df), total_null_rows, 100 * total_null_rows / max(len(df), 1),
        )
        return df

    # ── Step 3: sliding-window construction ──────────────────────────────

    def _build_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Build a strided sliding window over the feature columns.

        Returns
        -------
        x_seq      : float32 [N, window_size, seq_feature_dim]
        timestamps : int64   [N]  — Unix-second of the LAST row in each window

        NaN values are imputed with the per-column median before windowing so
        the model never receives NaN inputs. The imputation count is logged as
        DEBUG so it surfaces during debugging but doesn't clutter training logs.
        """
        T = self.cfg.window_size
        S = self.cfg.step_size

        feature_matrix: np.ndarray = (
            df[list(SEQ_FEATURE_COLS)].astype("float64").values  # [R, F]
        )
        ts_unix: np.ndarray = (
            df["timestamp"].astype("int64").values // 1_000_000_000  # [R]
        )

        n_rows, n_feats = feature_matrix.shape
        if n_rows < T:
            raise ValueError(
                f"Insufficient rows ({n_rows}) to build windows of size {T}. "
                "Reduce window_size or extend the date range."
            )

        # Impute NaN with per-column median before constructing the strided view.
        # If the median is itself NaN (column is 100% null), fall back to 0.0 —
        # a neutral value for log-transformed features.
        nan_col_indices = np.where(np.isnan(feature_matrix).any(axis=0))[0]
        if nan_col_indices.size > 0:
            col_medians = np.nanmedian(feature_matrix, axis=0)
            for ci in nan_col_indices:
                fill_value = col_medians[ci] if not np.isnan(col_medians[ci]) else 0.0
                nan_mask = np.isnan(feature_matrix[:, ci])
                feature_matrix[nan_mask, ci] = fill_value
                log.debug(
                    "Imputed %d NaN in column '%s' with %.4g",
                    nan_mask.sum(), SEQ_FEATURE_COLS[ci], fill_value,
                )

        # sliding_window_view produces a zero-copy strided view: [R-T+1, F, T]
        windows = np.lib.stride_tricks.sliding_window_view(
            feature_matrix, window_shape=T, axis=0
        )                                      # [R-T+1, F, T]
        windows = windows.transpose(0, 2, 1)   # [R-T+1, T, F]

        x_seq = windows[::S].astype("float32")            # [N, T, F]
        timestamps = ts_unix[T - 1 :: S].astype("int64")  # [N]

        log.info(
            "Windows built: N=%d, T=%d, F=%d  (step=%d)",
            len(x_seq), T, n_feats, S,
        )
        return x_seq, timestamps

    # ── Portfolio state placeholder ───────────────────────────────────────

    def _build_port(self, n_windows: int) -> np.ndarray:
        """Placeholder portfolio state [N, port_dim] — zeros until TBD.

        When the portfolio state is finalised:
          1. Update XPortSchema.feature_cols in schema.py.
          2. Replace the body of this method with the actual construction.
        """
        return np.zeros((n_windows, self._p_schema.feature_dim), dtype="float32")


__all__ = [
    "DatasetConfig",
    "EthDataset",
    "EthDatasetLoader",
]
