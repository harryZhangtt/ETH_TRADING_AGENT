"""
datasets_loader/test.py
=======================
Diagnostic script: load eth_dataset.npz and verify structural integrity.

Checks
------
1.  Array shapes match expectations (x_seq, x_port, timestamps, feature_names)
2.  Timestamp count, monotonicity, and human-readable date range
3.  NaN audit — per-feature across all windows and per-window totals
4.  Portfolio state is all-zeros (current placeholder)-> TODO: replace this with actual when env implementation is done
5.  Per-feature descriptive statistics (mean / std / min / max)

Visualisations (saved to data/dataset_loader/diagnostics/)
-----------------------------------------------------------
  fig1_sample_window_heatmap.png  — x_seq[0] as a [T × F] colour map
  fig2_nan_per_feature.png        — NaN count per feature column
  fig3_timestamp_gap.png          — hourly gaps between consecutive timestamps
  fig4_feature_means.png          — mean value per feature over all windows

Usage
-----
    python datasets_loader/test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets_loader import EthDataset, SEQ_FEATURE_COLS, SEQ_FEATURE_DIM, PORT_DIM

# ── paths ──────────────────────────────────────────────────────────────────
DATASET_PATH = "data/dataset_loader/eth_dataset.npz"
OUT_DIR      = Path("data/dataset_loader/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"


def _check(label: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    print(f"  {tag}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        raise AssertionError(label)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 1. Loading dataset ──────────────────────────────────────────────")
ds = EthDataset.load(DATASET_PATH)
print(f"  {ds}")

x_seq         : np.ndarray = ds.x_seq          # [N, T, F]
x_port        : np.ndarray = ds.x_port         # [N, P]
timestamps    : np.ndarray = ds.timestamps      # [N]  unix seconds
feature_names : np.ndarray = ds.feature_names  # [F]

N, T, F = x_seq.shape
P = x_port.shape[1]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Shape & schema checks
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 2. Shape & schema ───────────────────────────────────────────────")
_check("x_seq is 3-D",               x_seq.ndim == 3,                  f"got {x_seq.shape}")
_check("x_port is 2-D",              x_port.ndim == 2,                 f"got {x_port.shape}")
_check("x_seq dtype float32",        x_seq.dtype == np.float32,        str(x_seq.dtype))
_check("x_port dtype float32",       x_port.dtype == np.float32,       str(x_port.dtype))
_check("timestamps dtype int64",     timestamps.dtype == np.int64,     str(timestamps.dtype))
_check("N consistent across arrays",
       x_port.shape[0] == N and timestamps.shape[0] == N,
       f"x_port N={x_port.shape[0]}, ts N={timestamps.shape[0]}")
_check(f"feature_dim == SEQ_FEATURE_DIM ({SEQ_FEATURE_DIM})",
       F == SEQ_FEATURE_DIM, f"got {F}")
_check(f"port_dim == PORT_DIM ({PORT_DIM})",
       P == PORT_DIM, f"got {P}")
_check("feature_names length matches F",
       len(feature_names) == F, f"got {len(feature_names)}")
_check("feature_names match SEQ_FEATURE_COLS",
       list(feature_names) == list(SEQ_FEATURE_COLS))

print(f"\n  N={N:,}  T={T}  F={F}  P={P}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Timestamp checks
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 3. Timestamps ───────────────────────────────────────────────────")
ts_dt = pd.to_datetime(timestamps, unit="s", utc=True)

_check("timestamps monotonically increasing",
       bool(np.all(np.diff(timestamps) > 0)))

start_str = str(ts_dt[0])
end_str   = str(ts_dt[-1])
print(f"  range  : {start_str}  →  {end_str}")
print(f"  count  : {N:,} windows")

gaps_hrs = np.diff(timestamps) / 3600
expected_gap = 1.0   # hourly data, step_size=1
non_unit = np.where(gaps_hrs != expected_gap)[0]
if non_unit.size:
    print(f"  {WARN}  {non_unit.size} non-1h gaps between consecutive window tails:")
    for idx in non_unit[:5]:
        print(f"       window {idx}→{idx+1}: {gaps_hrs[idx]:.1f} h  "
              f"({ts_dt[idx]} → {ts_dt[idx+1]})")
    if non_unit.size > 5:
        print(f"       … and {non_unit.size - 5} more")
else:
    print(f"  {PASS}  all consecutive window tail-timestamps are 1 h apart")


# ══════════════════════════════════════════════════════════════════════════════
# 4. NaN audit — x_seq
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 4. NaN audit — x_seq ────────────────────────────────────────────")

nan_mask_seq  = np.isnan(x_seq)                      # [N, T, F]
total_nan_seq = int(nan_mask_seq.sum())
_check("zero NaN values in x_seq", total_nan_seq == 0,
       f"{total_nan_seq:,} NaN found")

# Per-feature NaN count (summed over N and T)
nan_per_feat = nan_mask_seq.sum(axis=(0, 1))          # [F]
if nan_per_feat.max() > 0:
    print("  NaN per feature (non-zero only):")
    for fi, count in enumerate(nan_per_feat):
        if count > 0:
            print(f"    {feature_names[fi]:45s}  {count:>8,}  "
                  f"({100*count/(N*T):.1f}%)")

# Per-window NaN count (any NaN in that window's [T, F] slice)
nan_per_window = nan_mask_seq.any(axis=(1, 2))        # [N] bool
print(f"  windows with any NaN : {nan_per_window.sum():,} / {N:,}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Portfolio state check
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 5. Portfolio state (x_port) ─────────────────────────────────────")
nan_port = int(np.isnan(x_port).sum())
_check("zero NaN in x_port", nan_port == 0, f"{nan_port:,} NaN")
_check("x_port is all-zeros (placeholder)",
       bool(np.all(x_port == 0.0)),
       f"non-zero entries: {int((x_port != 0).sum())}")
print(f"  shape {x_port.shape}  — zeros confirmed (TBD)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Per-feature statistics
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6. Per-feature statistics (over all N×T values) ─────────────────")
flat = x_seq.reshape(-1, F)    # [N*T, F]
means  = flat.mean(axis=0)
stds   = flat.std(axis=0)
mins   = flat.min(axis=0)
maxs   = flat.max(axis=0)

header = f"  {'feature':<45}  {'mean':>9}  {'std':>9}  {'min':>9}  {'max':>9}"
print(header)
print("  " + "-" * (len(header) - 2))
for fi, name in enumerate(feature_names):
    print(f"  {name:<45}  {means[fi]:>9.3f}  {stds[fi]:>9.3f}  "
          f"{mins[fi]:>9.3f}  {maxs[fi]:>9.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Spot-check three individual windows
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 7. Spot-check: three individual windows ──────────────────────────")
indices = [0, N // 2, N - 1]
for idx in indices:
    w   = x_seq[idx]          # [T, F]
    ts  = pd.Timestamp(timestamps[idx], unit="s", tz="UTC")
    win_nan = int(np.isnan(w).sum())
    win_inf = int(np.isinf(w).sum())
    print(f"  window[{idx:>6,}]  last_ts={ts}  "
          f"shape={w.shape}  NaN={win_nan}  Inf={win_inf}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Visualisations
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 8. Saving diagnostic plots ───────────────────────────────────────")

# ── fig 1: heatmap of the first window ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(x_seq[0].T, aspect="auto", interpolation="none",
               cmap="RdBu_r", vmin=-3, vmax=3)
ax.set_xlabel("Time step (hours, 0 = oldest)")
ax.set_ylabel("Feature index")
ax.set_yticks(range(F))
ax.set_yticklabels(feature_names, fontsize=7)
ax.set_title(f"x_seq[0] — window ending {pd.Timestamp(timestamps[0], unit='s', tz='UTC')}")
fig.colorbar(im, ax=ax, label="value")
fig.tight_layout()
p1 = OUT_DIR / "fig1_sample_window_heatmap.png"
fig.savefig(p1, dpi=120)
plt.close(fig)
print(f"  saved  {p1}")

# ── fig 2: NaN count per feature across all windows ───────────────────────
fig, ax = plt.subplots(figsize=(10, max(4, F * 0.28)))
colors = ["#d62728" if c > 0 else "#2ca02c" for c in nan_per_feat]
ax.barh(range(F), nan_per_feat, color=colors)
ax.set_yticks(range(F))
ax.set_yticklabels(feature_names, fontsize=7)
ax.set_xlabel("NaN count  (N × T cells)")
ax.set_title("NaN per feature in x_seq")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
p2 = OUT_DIR / "fig2_nan_per_feature.png"
fig.savefig(p2, dpi=120)
plt.close(fig)
print(f"  saved  {p2}")

# ── fig 3: timestamp gap distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(gaps_hrs, bins=50, color="#1f77b4", edgecolor="white")
ax.set_xlabel("Gap between consecutive window tail-timestamps (hours)")
ax.set_ylabel("Count")
ax.set_title("Timestamp gap distribution")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
fig.tight_layout()
p3 = OUT_DIR / "fig3_timestamp_gap.png"
fig.savefig(p3, dpi=120)
plt.close(fig)
print(f"  saved  {p3}")

# ── fig 4: per-feature mean value ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(4, F * 0.28)))
colors_m = ["#1f77b4" if m >= 0 else "#d62728" for m in means]
ax.barh(range(F), means, xerr=stds, color=colors_m,
        error_kw={"elinewidth": 0.8, "alpha": 0.6})
ax.set_yticks(range(F))
ax.set_yticklabels(feature_names, fontsize=7)
ax.set_xlabel("Mean ± std  (all N×T cells)")
ax.set_title("Per-feature mean value in x_seq")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
fig.tight_layout()
p4 = OUT_DIR / "fig4_feature_means.png"
fig.savefig(p4, dpi=120)
plt.close(fig)
print(f"  saved  {p4}")

print(f"\n{PASS}  All checks passed. Diagnostics → {OUT_DIR}/\n")
