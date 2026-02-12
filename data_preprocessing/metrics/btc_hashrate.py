from typing import Optional, Union

import pandas as pd
import requests

from .common.config import PipelineConfig
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range
from .common.transforms import to_unix_timestamp


def fetch_btc_hashrate(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
    as_unix: bool = True,
) -> pd.DataFrame:
    """
    Fetch BTC network hashrate (TH/s) as a daily series.

    Uses a public CSV chart endpoint configured by `config.btc_hashrate_url`.
    If the request fails or the response cannot be parsed, an empty DataFrame
    is returned and (optionally) saved.
    """
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)
    df = _get_btc_hashrate_csv(config=config)
    if df.empty:
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "btc_hashrate.csv", enabled=save)
        return output

    if "Timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["Timestamp"], unit="s", utc=True, errors="coerce"
        )
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["time"], unit="s", utc=True, errors="coerce"
        )
    else:
        # last column heuristic
        ts_col = df.columns[0]
        df["timestamp"] = pd.to_datetime(
            df[ts_col], unit="s", utc=True, errors="coerce"
        )

    # value column heuristic
    value_col = None
    for candidate in ["Value", "hashrate", "hash_rate"]:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        # assume the second column is the value
        if len(df.columns) > 1:
            value_col = df.columns[1]
        else:
            if config.debug:
                print("[DEBUG] btc_hashrate could not infer value column")
            output = pd.DataFrame()
            maybe_save_csv(output, config.output_dir, "btc_hashrate.csv", enabled=save)
            return output

    df["btc_hashrate_thps"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "btc_hashrate_thps"])

    # filter range and sort
    mask = (df["timestamp"] >= start_ts.floor("D")) & (
        df["timestamp"] < end_ts.floor("D") + pd.Timedelta(days=1)
    )
    df = df.loc[mask].sort_values("timestamp")

    output = df[["timestamp", "btc_hashrate_thps"]].copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "btc_hashrate.csv", enabled=save)
    return output


def _get_btc_hashrate_csv(config: PipelineConfig) -> pd.DataFrame:
    try:
        response = requests.get(config.btc_hashrate_url, timeout=30)
        response.raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        if config.debug:
            print(f"[DEBUG] btc_hashrate request failed: {exc}")
        return pd.DataFrame()

    text = response.text.strip()
    if not text:
        if config.debug:
            print("[DEBUG] btc_hashrate empty response body")
        return pd.DataFrame()

    try:
        df = pd.read_csv(pd.compat.StringIO(text))
    except Exception:
        # fallback: let pandas read directly from string buffer
        from io import StringIO

        try:
            df = pd.read_csv(StringIO(text))
        except Exception as exc:
            if config.debug:
                print(f"[DEBUG] btc_hashrate CSV parse failed: {exc}")
            return pd.DataFrame()

    return df

