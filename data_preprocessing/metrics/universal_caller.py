from typing import Optional, Union

import pandas as pd

try:
    from .btc_price_info import fetch_btc_price_info
    from .common.config import PipelineConfig
    from .common.io_utils import maybe_save_csv
    from .common.time_utils import resolve_time_range
    from .common.transforms import attach_daily_metric, null_value_check, to_unix_timestamp
    from .eth_rolling_beta import fetch_eth_rolling_beta
    from .eth_daily_txn import fetch_eth_daily_txn
    from .google_trend import fetch_google_trend
    from .ohlc_volume import fetch_eth_ohlc_volume
    from .supply import fetch_eth_supply_growth
except ImportError:  # Allow running as a script without package context.
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from data_preprocessing.metrics.btc_price_info import fetch_btc_price_info
    from data_preprocessing.metrics.common.config import PipelineConfig
    from data_preprocessing.metrics.common.io_utils import maybe_save_csv
    from data_preprocessing.metrics.common.time_utils import resolve_time_range
    from data_preprocessing.metrics.common.transforms import (
        attach_daily_metric,
        null_value_check,
        to_unix_timestamp,
    )
    from data_preprocessing.metrics.eth_rolling_beta import fetch_eth_rolling_beta
    from data_preprocessing.metrics.eth_daily_txn import fetch_eth_daily_txn
    from data_preprocessing.metrics.google_trend import fetch_google_trend
    from data_preprocessing.metrics.ohlc_volume import fetch_eth_ohlc_volume
    from data_preprocessing.metrics.supply import fetch_eth_supply_growth

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
    "google_trend",
    "eth_rolling_beta",
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
        as_unix=False,
    )
    if eth_ohlc.empty:
        return _empty_output()

    supply_daily_df = fetch_eth_supply_growth(
        start=start_ts,
        end=end_ts,
        config=config,
        save=save,
        as_unix=False,
    )
    if supply_daily_df.empty or "supply" not in supply_daily_df.columns:
        supply_series = pd.Series(dtype="float64")
    else:
        supply_series = supply_daily_df.set_index("timestamp")["supply"]
    eth_ohlc = attach_daily_metric(eth_ohlc, supply_series, "supply")

    daily_tx = fetch_eth_daily_txn(
        start=start_ts,
        end=end_ts,
        config=config,
        save=save,
        as_unix=False,
    )
    if daily_tx.empty:
        daily_series = pd.Series(dtype="float64")
    else:
        daily_series = daily_tx.set_index("timestamp")["eth_daily_tx"]
    eth_ohlc = attach_daily_metric(eth_ohlc, daily_series, "eth_daily_tx")

    google_df = fetch_google_trend(
        start=start_ts,
        end=end_ts,
        config=config,
        save=save,
        as_unix=False,
    )
    if google_df.empty:
        google_series = pd.Series(dtype="float64")
    else:
        google_series = google_df.set_index("timestamp")["google_trend"]
    eth_ohlc = attach_daily_metric(eth_ohlc, google_series, "google_trend")

    eth_ohlc["market_cap"] = eth_ohlc["close"].astype("float64") * eth_ohlc[
        "supply"
    ]

    btc_info = fetch_btc_price_info(
        start=start_ts,
        end=end_ts,
        caller=caller,
        config=config,
        save=save,
        as_unix=False,
    )
    eth_ohlc = _attach_btc_metrics(eth_ohlc, btc_info)

    beta_df = fetch_eth_rolling_beta(
        start=start_ts,
        end=end_ts,
        caller=caller,
        config=config,
        window=config.rolling_beta_window,
        eth_df=eth_ohlc,
        btc_df=btc_info,
        save=save,
        as_unix=False,
    )
    if beta_df.empty:
        eth_ohlc["eth_rolling_beta"] = pd.Series(dtype="float64")
    else:
        eth_ohlc = eth_ohlc.merge(beta_df, on="timestamp", how="left")

    eth_ohlc = null_value_check(eth_ohlc)
    output = to_unix_timestamp(eth_ohlc.copy(), "timestamp")

    maybe_save_csv(output, config.output_dir, "eth_metrics_combined.csv", enabled=save)
    return output[OUTPUT_COLUMNS]


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
            "google_trend": pd.Series(dtype="float64"),
            "eth_rolling_beta": pd.Series(dtype="float64"),
            "market_cap": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "btc_open": pd.Series(dtype="float64"),
            "btc_high": pd.Series(dtype="float64"),
            "btc_low": pd.Series(dtype="float64"),
            "btc_close": pd.Series(dtype="float64"),
            "btc_volume": pd.Series(dtype="float64"),
        }
    )
