from typing import Optional, Union

import pandas as pd

from .btc_price_info import fetch_btc_price_info
from .common.config import PipelineConfig
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range
from .common.transforms import attach_daily_metric, build_daily_series, null_value_check
from .eth_daily_txn import fetch_eth_daily_txn
from .ohlc_volume import fetch_eth_ohlc_volume
from .supply import fetch_eth_supply

OUTPUT_COLUMNS = [
    "ticker",
    "caller",
    "timestamp",
    "Open",
    "high",
    "close",
    "low",
    "supply",
    "eth_daily_tx",
    "market_cap",
    "volume",
    "btc_open",
    "btc_high",
    "btc_low",
    "btc_close",
    "btc_volume",
]


def build_universal_metrics(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    caller: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
) -> pd.DataFrame:
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)

    eth_ohlc = fetch_eth_ohlc_volume(
        start=start_ts,
        end=end_ts,
        caller=caller,
        config=config,
        save=save,
    )
    if eth_ohlc.empty:
        return _empty_output()

    supply_value = fetch_eth_supply(config=config)
    supply_daily = build_daily_series(start_ts, end_ts, supply_value)
    eth_ohlc = attach_daily_metric(eth_ohlc, supply_daily, "supply")

    daily_tx = fetch_eth_daily_txn(
        start=start_ts,
        end=end_ts,
        config=config,
        save=save,
    )
    if daily_tx.empty:
        daily_series = pd.Series(dtype="float64")
    else:
        daily_series = daily_tx.set_index("timestamp")["eth_daily_tx"]
    eth_ohlc = attach_daily_metric(eth_ohlc, daily_series, "eth_daily_tx")

    eth_ohlc["market_cap"] = eth_ohlc["close"].astype("float64") * eth_ohlc[
        "supply"
    ]

    btc_info = fetch_btc_price_info(
        start=start_ts,
        end=end_ts,
        caller=caller,
        config=config,
        save=save,
    )
    eth_ohlc = _attach_btc_metrics(eth_ohlc, btc_info)

    eth_ohlc = null_value_check(eth_ohlc)

    maybe_save_csv(eth_ohlc, config.output_dir, "eth_metrics_combined.csv", enabled=save)
    return eth_ohlc[OUTPUT_COLUMNS]


def _attach_btc_metrics(
    eth_ohlc: pd.DataFrame,
    btc_info: pd.DataFrame,
) -> pd.DataFrame:
    df = eth_ohlc.copy()
    if btc_info.empty:
        for col in ["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]:
            df[col] = pd.Series(dtype="float64")
        return df

    btc = btc_info.set_index("timestamp")[
        ["Open", "high", "low", "close", "volume"]
    ].rename(
        columns={
            "Open": "btc_open",
            "high": "btc_high",
            "low": "btc_low",
            "close": "btc_close",
            "volume": "btc_volume",
        }
    )
    return df.join(btc, on="timestamp")


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype="object"),
            "caller": pd.Series(dtype="object"),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "Open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "supply": pd.Series(dtype="float64"),
            "eth_daily_tx": pd.Series(dtype="float64"),
            "market_cap": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "btc_open": pd.Series(dtype="float64"),
            "btc_high": pd.Series(dtype="float64"),
            "btc_low": pd.Series(dtype="float64"),
            "btc_close": pd.Series(dtype="float64"),
            "btc_volume": pd.Series(dtype="float64"),
        }
    )
