from typing import Optional, Union

import pandas as pd
import requests

try:
    from .common.config import PipelineConfig
    from .common.etherscan_chart import fetch_chart_csv
    from .common.io_utils import maybe_save_csv
    from .common.time_utils import resolve_time_range
    from .common.transforms import build_daily_series, to_unix_timestamp
except ImportError:  # Allow running as a script without package context.
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from data_preprocessing.metrics.common.config import PipelineConfig
    from data_preprocessing.metrics.common.etherscan_chart import fetch_chart_csv
    from data_preprocessing.metrics.common.io_utils import maybe_save_csv
    from data_preprocessing.metrics.common.time_utils import resolve_time_range
    from data_preprocessing.metrics.common.transforms import build_daily_series, to_unix_timestamp


SUPPLY_GROWTH_URL = "https://etherscan.io/chart/ethersupplygrowth"


def fetch_eth_supply(config: PipelineConfig = PipelineConfig()) -> Optional[float]:
    params = {
        "chainid": str(config.etherscan_chain_id),
        "module": "stats",
        "action": "ethsupply",
    }
    if config.etherscan_api_key:
        params["apikey"] = config.etherscan_api_key

    try:
        response = requests.get(config.etherscan_api_base, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        if config.debug:
            print(f"[DEBUG] eth_supply request failed: {exc}")
        return None

    if str(payload.get("status")) != "1":
        if config.debug:
            print(f"[DEBUG] eth_supply bad status: {payload}")
        return None

    result = payload.get("result")
    if result is None:
        if config.debug:
            print("[DEBUG] eth_supply missing result")
        return None

    try:
        supply_wei = int(result)
    except (TypeError, ValueError) as exc:
        if config.debug:
            print(f"[DEBUG] eth_supply parse error: {exc}")
        return None

    return supply_wei / 1_000_000_000_000_000_000


def fetch_eth_supply_daily(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
    as_unix: bool = True,
) -> pd.DataFrame:
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)
    supply_value = fetch_eth_supply(config=config)
    series = build_daily_series(start_ts, end_ts, supply_value)
    df = series.reset_index()
    df.columns = ["timestamp", "supply"]

    output = df.copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "eth_supply_daily.csv", enabled=save)
    return output


def fetch_eth_supply_growth(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
    as_unix: bool = True,
) -> pd.DataFrame:
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)
    df = fetch_chart_csv(
        url=SUPPLY_GROWTH_URL,
        api_key=config.etherscan_api_key,
        debug=config.debug,
    )
    if df.empty:
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "eth_supply_growth.csv", enabled=save)
        return output

    df = _normalize_timestamp(df)
    df = _rename_value_columns(df)
    df = _filter_range(df, start_ts, end_ts)
    df = df.sort_values("timestamp")

    output = df.copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "eth_supply_growth.csv", enabled=save)
    return output


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "UnixTimeStamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["UnixTimeStamp"], errors="coerce")
    elif "Date(UTC)" in df.columns:
        dt = pd.to_datetime(df["Date(UTC)"], utc=True, errors="coerce")
        df["timestamp"] = dt.view("int64") // 1_000_000_000
    else:
        raise RuntimeError(
            f"Missing both UnixTimeStamp and Date(UTC). Columns={list(df.columns)}"
        )
    return df


def _rename_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col in ("timestamp", "Date(UTC)", "UnixTimeStamp", "DateTime"):
            continue
        new = (
            col.strip()
            .lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace("/", "_")
        )
        while "__" in new:
            new = new.replace("__", "_")
        rename_map[col] = new
    return df.rename(columns=rename_map)


def _filter_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    mask = (ts >= start_ts) & (ts < end_ts)
    return df.loc[mask].copy()


if __name__ == "__main__":
    config = PipelineConfig(debug=True)
    daily = fetch_eth_supply_daily(period="30d", config=config)
    print(daily.head())
    growth = fetch_eth_supply_growth(period="365d", config=config)
    print(growth.head())
