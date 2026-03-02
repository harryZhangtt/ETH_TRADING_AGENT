import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_INPUT = Path("data_preprocessing/data/metrics/eth_metrics_combined.csv")
DEFAULT_OUTPUT_DIR = Path("data_preprocessing/data/midway")
DEFAULT_OUTPUT_NAME = "eth_metrics_midway.csv"
METRICS_DIR = Path("data_preprocessing/data/metrics")


METRIC_ALIASES: Dict[str, List[str]] = {
    "close": ["close", "Close", "eth_close"],
    "high": ["high", "High"],
    "low": ["low", "Low"],
    "volume": ["volume", "Volume", "eth_volume"],
    "supply": ["supply", "eth_supply"],
    "btc_close": ["btc_close", "btcClose", "bitcoin_close"],
    "daily_txn": ["eth_daily_tx", "daily_txn", "eth_daily_txn", "daily_transactions"],
    "google_trend": ["google_trend", "eth_google_trend"],
    "market_beta": ["eth_rolling_beta", "market_beta", "eth_market_beta", "rolling_beta"],
    "tweet_volume": ["tweet_volume", "eth_tweet_volume"],
    "network_tx_fee": ["network_tx_fee", "eth_network_tx_fee", "network_txfee"],
    "btc_hashrate": ["btc_hashrate", "bitcoin_hashrate"],
    "spread": ["spread", "order_book_spread", "bid_ask_spread"],
    "ask_depth": ["ask_depth", "order_book_ask_depth"],
    "bid_depth": ["bid_depth", "order_book_bid_depth"],
    "order_book_imbalance": ["order_book_imbalance", "ob_imbalance", "imbalance"],
    "unique_addr": ["unique_addr", "eth_unique_addresses", "unique_addresses"],
    "avg_txn_fee": ["avg_txn_fee", "avg_tx_fee", "eth_avg_txfee", "avg_transaction_fee"],
    "avg_block_size": ["avg_block_size", "eth_avg_block_size"],
    "daily_token_transfer": [
        "daily_token_transfer",
        "daily_token_transfers",
        "eth_erc20_daily_token_transfers",
    ],
    "macro_economics": [
        "macro_economics",
        "macro_index",
        "macro_factor",
    ],
    "daily_active_ethereum_address": [
        "daily_active_ethereum_address",
        "daily_active_eth_address",
    ],
}

SUPPLEMENTAL_SOURCES: Dict[str, Dict[str, object]] = {
    "tweet_volume": {
        "path": METRICS_DIR / "eth_tweet_volume.csv",
        "value_columns": ["tweet_volume"],
    },
    "network_tx_fee": {
        "path": METRICS_DIR / "eth_avg_txfee.csv",
        "value_columns": ["avg_txfee_eth", "network_tx_fee"],
    },
    "avg_txn_fee": {
        "path": METRICS_DIR / "eth_avg_txfee.csv",
        "value_columns": ["avg_txfee_usd", "avg_txn_fee"],
    },
    "btc_hashrate": {
        "path": METRICS_DIR / "btc_hashrate.csv",
        "value_columns": ["btc_hashrate_thps", "btc_hashrate"],
    },
    "spread": {
        "path": METRICS_DIR / "eth_order_book_metrics.csv",
        "value_columns": ["spread"],
    },
    "ask_depth": {
        "path": METRICS_DIR / "eth_order_book_metrics.csv",
        "value_columns": ["ask_depth"],
    },
    "bid_depth": {
        "path": METRICS_DIR / "eth_order_book_metrics.csv",
        "value_columns": ["bid_depth"],
    },
    "order_book_imbalance": {
        "path": METRICS_DIR / "eth_order_book_metrics.csv",
        "value_columns": ["order_book_imbalance"],
    },
    "unique_addr": {
        "path": METRICS_DIR / "eth_unique_addresses.csv",
        "value_columns": ["total_unique_addresses", "unique_addr"],
    },
    "daily_active_ethereum_address": {
        "path": METRICS_DIR / "eth_unique_addresses.csv",
        "value_columns": ["daily_increase", "daily_active_ethereum_address"],
    },
    "avg_block_size": {
        "path": METRICS_DIR / "eth_avg_block_size.csv",
        "value_columns": ["avg_block_size_bytes", "avg_block_size"],
    },
    "daily_token_transfer": {
        "path": METRICS_DIR / "eth_erc20_daily_token_transfers.csv",
        "value_columns": [
            "erc20_daily_token_transfers",
            "daily_token_transfer",
            "daily_token_transfers",
        ],
    },
}


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

    if abs_value >= 1e18:  # very unlikely, but guard absurd precision
        value = value / 1e9
    elif abs_value >= 1e14:  # nanoseconds
        value = value / 1e9
    elif abs_value >= 1e12:  # milliseconds
        value = value / 1000.0
    elif 1e6 <= abs_value < 1e9:
        # Some repo files appear to have stripped 3 trailing zeros.
        value = value * 1000.0

    return int(value)


def _to_day_bucket(ts_seconds: int) -> int:
    return (ts_seconds // 86400) * 86400


def _parse_timestamp_to_unix(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    numeric = _to_float(text)
    if not math.isnan(numeric):
        return _normalize_unix_seconds(numeric)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except ValueError:
        return None


def _safe_log(value: float) -> float:
    if math.isnan(value) or value <= 0:
        return math.nan
    return math.log(value)


def _log_series(series: List[float]) -> List[float]:
    return [_safe_log(v) for v in series]


def _log_ratio(series: List[float], lag: int) -> List[float]:
    out = [math.nan] * len(series)
    for i in range(lag, len(series)):
        cur = series[i]
        prev = series[i - lag]
        if math.isnan(cur) or math.isnan(prev) or cur <= 0 or prev <= 0:
            continue
        out[i] = math.log(cur / prev)
    return out


def _log_ratio_two(numer: List[float], denom: List[float]) -> List[float]:
    out = [math.nan] * len(numer)
    for i in range(len(numer)):
        a = numer[i]
        b = denom[i]
        if math.isnan(a) or math.isnan(b) or a <= 0 or b <= 0:
            continue
        out[i] = math.log(a / b)
    return out


def _resolve_column(fieldnames: List[str], metric_key: str) -> Optional[str]:
    aliases = METRIC_ALIASES.get(metric_key, [metric_key])
    lowered = {name.lower(): name for name in fieldnames}
    for alias in aliases:
        if alias in fieldnames:
            return alias
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _read_rows(input_csv: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not input_csv.exists() or input_csv.stat().st_size == 0:
        return [], []
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def _sort_rows_by_timestamp(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def ts_key(row: Dict[str, str]) -> float:
        ts = _parse_timestamp_to_unix(row.get("timestamp"))
        if ts is None:
            return math.inf
        return float(ts)
    return sorted(rows, key=ts_key)


def _as_str(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.16g}"


def _series_non_nan_count(series: List[float]) -> int:
    return sum(0 if math.isnan(v) else 1 for v in series)


def _series_positive_count(series: List[float]) -> int:
    return sum(1 for v in series if (not math.isnan(v) and v > 0))


def _resolve_any_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lowered = {name.lower(): name for name in fieldnames}
    for name in candidates:
        if name in fieldnames:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _load_supplemental_series(
    base_rows: List[Dict[str, str]],
    metric_key: str,
) -> Tuple[List[float], Optional[str]]:
    spec = SUPPLEMENTAL_SOURCES.get(metric_key)
    if spec is None:
        return [math.nan] * len(base_rows), None

    file_path = spec["path"]
    fieldnames, rows = _read_rows(file_path)
    if not fieldnames or not rows:
        return [math.nan] * len(base_rows), None

    ts_col = _resolve_any_column(fieldnames, ["timestamp", "Timestamp", "time", "UnixTimeStamp"])
    value_col = _resolve_any_column(fieldnames, list(spec["value_columns"]))
    if ts_col is None or value_col is None:
        return [math.nan] * len(base_rows), None

    daily_map: Dict[int, float] = {}
    for row in rows:
        ts = _parse_timestamp_to_unix(row.get(ts_col))
        if ts is None:
            continue
        day = _to_day_bucket(ts)
        val = _to_float(row.get(value_col))
        if math.isnan(val):
            continue
        daily_map[day] = val

    if not daily_map:
        return [math.nan] * len(base_rows), None

    output = [math.nan] * len(base_rows)
    last_value = math.nan
    for i, row in enumerate(base_rows):
        ts = _parse_timestamp_to_unix(row.get("timestamp"))
        if ts is None:
            output[i] = math.nan
            continue
        day = _to_day_bucket(ts)
        if day in daily_map:
            last_value = daily_map[day]
        output[i] = last_value

    source_ref = f"{file_path.name}:{value_col}"
    return output, source_ref


def build_midway_dataset(
    input_csv: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_name: str = DEFAULT_OUTPUT_NAME,
    multi_day_lag: int = 7,
) -> Tuple[Path, Path]:
    fieldnames, rows = _read_rows(input_csv)
    rows = _sort_rows_by_timestamp(rows)

    resolved_base = {key: _resolve_column(fieldnames, key) for key in METRIC_ALIASES}
    resolved: Dict[str, Optional[str]] = dict(resolved_base)
    series_cache: Dict[str, List[float]] = {}

    def get_series(metric_key: str) -> List[float]:
        if metric_key in series_cache:
            return series_cache[metric_key]

        source_col = resolved_base.get(metric_key)
        base_series = (
            [_to_float(r.get(source_col)) for r in rows]
            if source_col is not None
            else [math.nan] * len(rows)
        )
        base_count = _series_non_nan_count(base_series)
        if base_count > 0:
            if source_col is not None:
                resolved[metric_key] = source_col
            return base_series

        supplemental_series, supplemental_source = _load_supplemental_series(rows, metric_key)
        supp_count = _series_non_nan_count(supplemental_series)
        if supp_count > 0 and supplemental_source is not None:
            resolved[metric_key] = supplemental_source
            chosen_series = supplemental_series
        else:
            resolved[metric_key] = source_col
            chosen_series = base_series

        if metric_key == "network_tx_fee" and _series_positive_count(chosen_series) == 0:
            avg_txn_fee = get_series("avg_txn_fee")
            daily_txn = get_series("daily_txn")
            derived = [math.nan] * len(rows)
            for i in range(len(rows)):
                fee = avg_txn_fee[i]
                tx = daily_txn[i]
                if math.isnan(fee) or math.isnan(tx):
                    continue
                val = fee * tx
                if val > 0:
                    derived[i] = val
            if _series_positive_count(derived) > 0:
                chosen_series = derived
                resolved[metric_key] = "derived:avg_txn_fee*daily_txn"

        series_cache[metric_key] = chosen_series
        return chosen_series

    close = get_series("close")
    high = get_series("high")
    low = get_series("low")
    btc_close = get_series("btc_close")

    # Core returns/range.
    single_day_log_return = _log_ratio(close, lag=1)
    multi_day_log_return = _log_ratio(close, lag=multi_day_lag)
    intraday_range_log = _log_ratio_two(high, low)
    btc_single_day_log_return = _log_ratio(btc_close, lag=1)

    # Requested log and delta-log features.
    feature_specs = [
        ("volume", "log_volume", "log_volume_change"),
        ("supply", "log_supply", "log_supply_change"),
        ("daily_txn", "log_daily_txn", None),
        ("google_trend", "log_google_trend", "log_google_trend_change"),
        ("market_beta", "log_market_beta", None),
        ("tweet_volume", "log_tweet_volume", "log_tweet_volume_change"),
        ("network_tx_fee", "log_network_tx_fee", "log_network_tx_fee_change"),
        ("btc_hashrate", "log_btc_hashrate", "log_btc_hashrate_change"),
        ("spread", "log_spread", "log_spread_change"),
        ("ask_depth", "log_ask_depth", "log_ask_depth_change"),
        ("bid_depth", "log_bid_depth", "log_bid_depth_change"),
        (
            "order_book_imbalance",
            "log_order_book_imbalance",
            "log_order_book_imbalance_change",
        ),
        ("unique_addr", "log_unique_addr", "log_unique_addr_change"),
        ("avg_txn_fee", "log_avg_txn_fee", "log_avg_txn_fee_change"),
        ("avg_block_size", "log_avg_block_size", "log_avg_block_size_change"),
        (
            "daily_token_transfer",
            "log_daily_token_transfer",
            "log_daily_token_transfer_change",
        ),
        ("macro_economics", "log_macro_economics", "log_macro_economics_change"),
    ]

    computed: Dict[str, List[float]] = {
        "single_day_log_return": single_day_log_return,
        f"multi_day_log_return_{multi_day_lag}": multi_day_log_return,
        "intraday_range_log": intraday_range_log,
        "btc_single_day_log_return": btc_single_day_log_return,
    }

    for metric_key, log_name, change_name in feature_specs:
        series = get_series(metric_key)
        computed[log_name] = _log_series(series)
        if change_name is not None:
            computed[change_name] = _log_ratio(series, lag=1)

    # Daily Active Ethereum Address (delta_log), if present.
    daily_active_series = get_series("daily_active_ethereum_address")
    computed["daily_active_ethereum_address_delta_log"] = _log_ratio(
        daily_active_series, lag=1
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / output_name
    output_json = output_dir / "eth_metrics_midway_metadata.json"

    out_fields = list(fieldnames) + list(computed.keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            out_row = dict(row)
            for col_name, values in computed.items():
                out_row[col_name] = _as_str(values[i])
            writer.writerow(out_row)

    metadata = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "multi_day_lag": multi_day_lag,
        "resolved_source_columns": resolved,
        "supplemental_metric_files": {
            key: str(spec["path"]) for key, spec in SUPPLEMENTAL_SOURCES.items()
        },
        "missing_source_metrics": [k for k, v in resolved.items() if v is None],
        "row_count": len(rows),
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_csv, output_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct midway ETH feature set with log and delta-log transforms."
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT),
        help="Path to source combined metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for midway outputs.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Filename for midway CSV output.",
    )
    parser.add_argument(
        "--multi-day-lag",
        type=int,
        default=7,
        help="Lag k for multi-day log return: log(c_t / c_{t-k}).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.multi_day_lag < 1:
        raise ValueError("--multi-day-lag must be >= 1.")
    output_csv, output_json = build_midway_dataset(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        output_name=args.output_name,
        multi_day_lag=args.multi_day_lag,
    )
    print(f"Saved midway dataset: {output_csv}")
    print(f"Saved metadata: {output_json}")


if __name__ == "__main__":
    main()
