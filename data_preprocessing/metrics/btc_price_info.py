from typing import Optional, Union

import pandas as pd

from .common.coinbase_client import fetch_coinbase_candles, format_ohlc_frame
from .common.config import PipelineConfig
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range
from .common.transforms import to_unix_timestamp


def fetch_btc_price_info(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    caller: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
    as_unix: bool = True,
) -> pd.DataFrame:
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)
    raw = fetch_coinbase_candles(
        product_id=config.btc_product_id,
        start=start_ts,
        end=end_ts,
        interval_seconds=config.interval_seconds,
        max_candles=config.max_candles_per_call,
        api_base=config.coinbase_api_base,
    )

    if raw.empty:
        df = pd.DataFrame()
    else:
        df = format_ohlc_frame(raw, config.btc_product_id, caller)

    output = df.copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "btc_price_info.csv", enabled=save)
    return output
