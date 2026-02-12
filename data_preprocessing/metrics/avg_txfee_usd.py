from typing import Optional, Union

import pandas as pd

from .common.config import PipelineConfig
from .common.etherscan_chart import fetch_chart_csv
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range
from .common.transforms import to_unix_timestamp


def fetch_avg_txfee(
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
        url="https://etherscan.io/chart/avg-txfee-usd",
        api_key=config.etherscan_api_key,
        debug=config.debug,
    )
    if df.empty:
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "eth_avg_txfee.csv", enabled=save)
        return output

    if "UnixTimeStamp" not in df.columns:
        raise RuntimeError("UnixTimeStamp column not found in avg tx fee chart.")

    df["timestamp"] = pd.to_numeric(df["UnixTimeStamp"], errors="coerce")

    rename_map = {}
    if "Average Txn Fee (USD)" in df.columns:
        rename_map["Average Txn Fee (USD)"] = "avg_txfee_usd"
    if "Average Txn Fee (Ether)" in df.columns:
        rename_map["Average Txn Fee (Ether)"] = "avg_txfee_eth"
    if not rename_map:
        raise RuntimeError("Expected fee columns not found.")

    df = df.rename(columns=rename_map)
    df = _filter_range(df, start_ts, end_ts)
    df = df.sort_values("timestamp")

    output_cols = ["timestamp"] + list(rename_map.values())
    output = df[output_cols].copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "eth_avg_txfee.csv", enabled=save)
    return output


def _filter_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    mask = (ts >= start_ts) & (ts < end_ts)
    return df.loc[mask].copy()
