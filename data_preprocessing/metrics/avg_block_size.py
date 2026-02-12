from typing import Optional, Union

import pandas as pd

from .common.config import PipelineConfig
from .common.etherscan_chart import fetch_chart_csv
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range
from .common.transforms import to_unix_timestamp


def fetch_avg_block_size(
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
        url="https://etherscan.io/chart/blocksize",
        api_key=config.etherscan_api_key,
        debug=config.debug,
    )
    if df.empty:
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "eth_avg_block_size.csv", enabled=save)
        return output

    df = _normalize_timestamp(df)
    value_col = _single_value_column(df, exclude={"Date(UTC)", "UnixTimeStamp", "DateTime", "timestamp"})
    df["avg_block_size_bytes"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.drop(columns=[value_col])

    df = _filter_range(df, start_ts, end_ts)
    df = df.sort_values("timestamp")

    output = df[["timestamp", "avg_block_size_bytes"]].copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "eth_avg_block_size.csv", enabled=save)
    return output


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "UnixTimeStamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["UnixTimeStamp"], errors="coerce")
    elif "Date(UTC)" in df.columns:
        dt = pd.to_datetime(df["Date(UTC)"], utc=True, errors="coerce")
        df["timestamp"] = dt.view("int64") // 1_000_000_000
    else:
        raise RuntimeError("No timestamp column found in block size chart.")
    return df


def _single_value_column(df: pd.DataFrame, exclude: set) -> str:
    value_candidates = [c for c in df.columns if c not in exclude]
    if len(value_candidates) != 1:
        raise RuntimeError(f"Unexpected value columns: {value_candidates}")
    return value_candidates[0]


def _filter_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    mask = (ts >= start_ts) & (ts < end_ts)
    return df.loc[mask].copy()
