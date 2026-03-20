import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_CSV = Path("data/metrics/eth_metrics_combined.csv")
ONE_HOUR_SECONDS = 3600


def _to_float(value: Optional[str]) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _normalize_unix_seconds(ts: float) -> Optional[int]:
    if math.isnan(ts):
        return None
    value = float(ts)
    abs_value = abs(value)

    if abs_value >= 1e18:
        value = value / 1e9
    elif abs_value >= 1e14:
        value = value / 1e9
    elif abs_value >= 1e12:
        value = value / 1000.0
    elif 1e6 <= abs_value < 1e9:
        value = value * 1000.0

    return int(value)


def _parse_timestamp(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    numeric = _to_float(text)
    if not math.isnan(numeric):
        return _normalize_unix_seconds(numeric)

    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "timestamp" not in reader.fieldnames:
            raise ValueError(f"{csv_path} is missing a 'timestamp' column.")
        return list(reader)


def _find_non_hourly_intervals(rows: List[Dict[str, str]]) -> Tuple[int, List[dict]]:
    parsed_rows = []
    for index, row in enumerate(rows, start=2):
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None:
            raise ValueError(f"Invalid timestamp at CSV line {index}: {row.get('timestamp')}")
        parsed_rows.append((index, timestamp, row))

    parsed_rows.sort(key=lambda item: item[1])

    mismatches = []
    for previous, current in zip(parsed_rows, parsed_rows[1:]):
        prev_line, prev_ts, _ = previous
        curr_line, curr_ts, _ = current
        delta = curr_ts - prev_ts
        if delta != ONE_HOUR_SECONDS:
            mismatches.append(
                {
                    "previous_line": prev_line,
                    "current_line": curr_line,
                    "previous_timestamp": prev_ts,
                    "current_timestamp": curr_ts,
                    "delta_seconds": delta,
                }
            )
    return len(parsed_rows), mismatches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether consecutive timestamps in a CSV are strictly 1 hour apart."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV),
        help="CSV file to check. Defaults to data/metrics/eth_metrics_combined.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of interval mismatches to print.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    rows = _load_rows(csv_path)
    row_count, mismatches = _find_non_hourly_intervals(rows)

    if not mismatches:
        print(f"PASS: {csv_path} has {row_count} rows with strictly 1-hour intervals.")
        return 0

    print(
        f"FAIL: {csv_path} has {len(mismatches)} non-1-hour intervals out of {max(row_count - 1, 0)} gaps."
    )
    for item in mismatches[: max(args.limit, 0)]:
        print(
            "line "
            f"{item['previous_line']} -> {item['current_line']}: "
            f"delta={item['delta_seconds']} seconds "
            f"({item['previous_timestamp']} -> {item['current_timestamp']})"
        )

    if len(mismatches) > max(args.limit, 0):
        remaining = len(mismatches) - max(args.limit, 0)
        print(f"... {remaining} more mismatches not shown")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
