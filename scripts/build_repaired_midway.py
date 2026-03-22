"""
Build model-ready repaired midway datasets with all engineered log columns.

This is a two-stage wrapper:
1. Repair the base combined+macro CSV with one or more gap-fill methods.
2. Recompute the midway features from each repaired base CSV so the derived
   log/return columns stay internally consistent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.repair_hourly_gaps import repair_csv
from data_preprocessing.further_preprocessing import build_midway_dataset


DEFAULT_BASE_INPUT = Path("data/metrics/eth_metrics_combined_macro.csv")
DEFAULT_OUTPUT_DIR = Path("data/gap_repair")
DEFAULT_REFERENCE_MIDWAY = Path("data/midway/eth_metrics_midway.csv")


def _resolve_trim_before(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair the base hourly CSV and regenerate midway datasets that retain "
            "all engineered feature columns."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_BASE_INPUT),
        help="Base combined+macro CSV to repair.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the repaired midway CSV outputs.",
    )
    parser.add_argument(
        "--trim-before",
        default="2016-05-26T00:00:00Z",
        help="Optional UTC timestamp to cut the sparse launch window before repair.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["average", "conservative"],
        choices=["average", "conservative"],
        help="One or both repair methods to run.",
    )
    parser.add_argument(
        "--multi-day-lag",
        type=int,
        default=7,
        help="Lag used when regenerating multi-day log returns.",
    )
    parser.add_argument(
        "--reference-midway",
        default=str(DEFAULT_REFERENCE_MIDWAY),
        help="Existing midway CSV whose column order should be preserved when possible.",
    )
    return parser


def _reorder_like_reference(output_csv: Path, reference_csv: Path) -> None:
    if not reference_csv.exists() or not output_csv.exists():
        return

    ref_df = pd.read_csv(reference_csv, nrows=0)
    out_df = pd.read_csv(output_csv)
    ref_cols = list(ref_df.columns)

    if set(ref_cols) != set(out_df.columns):
        return

    out_df = out_df[ref_cols]
    out_df.to_csv(output_csv, index=False)


def main() -> int:
    args = _build_parser().parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    trim_before = _resolve_trim_before(args.trim_before)
    reference_midway = Path(args.reference_midway)

    base_dir = output_dir / "_repaired_base"
    summaries = repair_csv(
        input_csv=input_csv,
        output_dir=base_dir,
        methods=args.methods,
        trim_before=trim_before,
    )

    written: List[Path] = []
    for summary in summaries:
        midway_name = f"eth_metrics_midway_{summary.method}_fill.csv"
        midway_csv, _ = build_midway_dataset(
            input_csv=summary.output_path,
            output_dir=output_dir,
            output_name=midway_name,
            multi_day_lag=args.multi_day_lag,
        )
        _reorder_like_reference(midway_csv, reference_midway)
        written.append(midway_csv)

    for path in written:
        print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
