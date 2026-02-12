from typing import Optional, Union

import numpy as np
import pandas as pd

from .btc_price_info import fetch_btc_price_info
from .common.config import PipelineConfig
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range
from .common.transforms import to_unix_timestamp
from .ohlc_volume import fetch_eth_ohlc_volume


def fetch_eth_rolling_beta(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    caller: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    window: Optional[int] = None,
    eth_df: Optional[pd.DataFrame] = None,
    btc_df: Optional[pd.DataFrame] = None,
    save: bool = True,
    as_unix: bool = True,
) -> pd.DataFrame:
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)
    window = window or config.rolling_beta_window

    if eth_df is None:
        eth_df = fetch_eth_ohlc_volume(
            start=start_ts,
            end=end_ts,
            caller=caller,
            config=config,
            save=False,
            as_unix=False,
        )
    if btc_df is None:
        btc_df = fetch_btc_price_info(
            start=start_ts,
            end=end_ts,
            caller=caller,
            config=config,
            save=False,
            as_unix=False,
        )

    if eth_df.empty or btc_df.empty:
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "eth_rolling_beta.csv", enabled=save)
        return output

    df = _compute_beta(eth_df, btc_df, window)

    output = df.copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "eth_rolling_beta.csv", enabled=save)
    return output


def _compute_beta(eth_df: pd.DataFrame, btc_df: pd.DataFrame, window: int) -> pd.DataFrame:
    eth = eth_df.set_index("timestamp")["close"].astype(float)
    btc = btc_df.set_index("timestamp")["close"].astype(float)
    aligned = pd.concat([eth, btc], axis=1, join="inner")
    aligned.columns = ["eth_close", "btc_close"]

    eth_ret = np.log(aligned["eth_close"]).diff()
    btc_ret = np.log(aligned["btc_close"]).diff()

    cov = eth_ret.rolling(window=window, min_periods=window).cov(btc_ret)
    var = btc_ret.rolling(window=window, min_periods=window).var()
    beta = cov / var

    df = pd.DataFrame({"timestamp": aligned.index, "eth_rolling_beta": beta.values})
    return df
