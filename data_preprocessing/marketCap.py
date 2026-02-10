##get ETH hourly OHLC data and volume, plus market cap

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class MarketCapConfig:
    coinbase_product_id: str = "ETH-USD"
    interval_seconds: int = 3600
    max_candles_per_call: int = 300
    coinbase_api_base: str = "https://api.exchange.coinbase.com"
    etherscan_api_base: str = "https://api.etherscan.io/v2/api"
    etherscan_api_key: str = '7K814DY5AXIQCHEH9VKWBBIP1AAMHU2VIS'
    etherscan_chain_id: int = 1


OUTPUT_COLUMNS = [
    "ticker",
    "caller",
    "timestamp",
    "Open",
    "high",
    "close",
    "low",
    "market_cap",
    "volume",
]


def get_eth_hourly_ohlc(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    caller: Optional[str] = None,
    config: MarketCapConfig = MarketCapConfig(),
) -> pd.DataFrame:
    """
    Fetch ETH hourly OHLC data (Coinbase) with market cap from Etherscan supply.

    Args:
        start: Start timestamp (inclusive). Requires end if provided.
        end: End timestamp (exclusive). If omitted with start, defaults to now (UTC).
        period: Time span like "60d". Use either period or start/end.
        caller: Optional caller label to include in output.
        config: MarketCapConfig for API endpoints and Etherscan settings.

    Returns:
        DataFrame with columns:
        ticker, caller, timestamp, Open, high, close, low, market_cap, volume
    """
    return get_hourly_ohlc_with_market_cap(
        product_id=config.coinbase_product_id,
        interval_seconds=config.interval_seconds,
        max_candles_per_call=config.max_candles_per_call,
        start=start,
        end=end,
        period=period,
        coinbase_api_base=config.coinbase_api_base,
        caller=caller,
        etherscan_api_base=config.etherscan_api_base,
        etherscan_api_key=config.etherscan_api_key,
        etherscan_chain_id=config.etherscan_chain_id,
    )


def get_hourly_ohlc_with_market_cap(
    product_id: str,
    interval_seconds: int = 3600,
    max_candles_per_call: int = 300,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    caller: Optional[str] = None,
    coinbase_api_base: str = "https://api.exchange.coinbase.com",
    etherscan_api_base: str = "https://api.etherscan.io/v2/api",
    etherscan_api_key: Optional[str] = None,
    etherscan_chain_id: int = 1,
) -> pd.DataFrame:
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = _resolve_time_range(start, end, period)
    candles = _fetch_coinbase_candles(
        product_id=product_id,
        start=start_ts,
        end=end_ts,
        interval_seconds=interval_seconds,
        max_candles=max_candles_per_call,
        api_base=coinbase_api_base,
    )

    if candles.empty:
        return _empty_output()

    candles = _format_candles(candles, product_id, caller)
    supply_value = _fetch_eth_supply_etherscan(
        api_base=etherscan_api_base,
        api_key=etherscan_api_key,
        chain_id=etherscan_chain_id,
    )
    supply_daily = _build_daily_supply_series(start_ts, end_ts, supply_value)
    candles = _attach_supply(candles, supply_daily)
    candles["market_cap"] = candles["close"].astype("float64") * candles["supply"]
    candles = candles.drop(columns=["supply"])

    return candles[OUTPUT_COLUMNS]


def _resolve_time_range(
    start: Optional[Union[str, pd.Timestamp]],
    end: Optional[Union[str, pd.Timestamp]],
    period: Optional[str],
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if start is None and period is None:
        period = "60d"

    if start is None:
        end_ts = _to_utc(end) or _now_utc()
        start_ts = end_ts - pd.Timedelta(period)
        return start_ts, end_ts

    start_ts = _to_utc(start)
    end_ts = _to_utc(end) or _now_utc()
    if end_ts <= start_ts:
        raise ValueError("end must be after start.")
    return start_ts, end_ts


def _to_utc(value: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _fetch_coinbase_candles(
    product_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_seconds: int,
    max_candles: int,
    api_base: str,
) -> pd.DataFrame:
    session = requests.Session()
    frames: List[pd.DataFrame] = []
    for chunk_start, chunk_end in _iter_candle_ranges(
        start, end, interval_seconds, max_candles
    ):
        params = {
            "start": chunk_start.isoformat(),
            "end": chunk_end.isoformat(),
            "granularity": interval_seconds,
        }
        url = f"{api_base}/products/{product_id}/candles"
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue
        frame = pd.DataFrame(
            data,
            columns=["time", "low", "high", "open", "close", "volume"],
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["time"], unit="s", utc=True)
    combined = combined.drop(columns=["time"])
    combined = combined[
        (combined["timestamp"] >= start) & (combined["timestamp"] < end)
    ]
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def _iter_candle_ranges(
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_seconds: int,
    max_candles: int,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    step = pd.Timedelta(seconds=interval_seconds * max_candles)
    current = start
    while current < end:
        chunk_end = min(current + step, end)
        yield current, chunk_end
        current = chunk_end


def _format_candles(
    raw: pd.DataFrame, product_id: str, caller: Optional[str]
) -> pd.DataFrame:
    df = raw.copy()
    df = df.rename(
        columns={
            "open": "Open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    df.insert(0, "ticker", product_id)
    df.insert(1, "caller", caller)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _fetch_eth_supply_etherscan(
    api_base: str,
    api_key: Optional[str],
    chain_id: int,
) -> Optional[float]:
    params = {
        "chainid": str(chain_id),
        "module": "stats",
        "action": "ethsupply",
    }
    if api_key:
        params["apikey"] = api_key

    try:
        response = requests.get(api_base, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    if str(payload.get("status")) != "1":
        return None

    result = payload.get("result")
    if result is None:
        print('Warning: no circularting supply return')
        return None

    try:
        supply_wei = int(result)
        print(f"Fetched circulating supply from Etherscan: {supply_wei} wei")
    except (TypeError, ValueError):
        print('Warning: invalid circulating supply value')
        return None

    return supply_wei / 1_000_000_000_000_000_000


def _build_daily_supply_series(
    start: pd.Timestamp,
    end: pd.Timestamp,
    supply_value: Optional[float],
) -> pd.Series:
    if supply_value is None:
        return pd.Series(dtype="float64")

    start_day = start.floor("D")
    end_day = end.floor("D")
    dates = pd.date_range(start=start_day, end=end_day, freq="D", tz="UTC")
    return pd.Series(supply_value, index=dates, dtype="float64")


def _attach_supply(candles: pd.DataFrame, daily_supply: pd.Series) -> pd.DataFrame:
    df = candles.sort_values("timestamp").copy()
    if daily_supply.empty:
        df["supply"] = pd.Series(np.nan, index=df.index, dtype="float64")
        return df

    supply_map = daily_supply.copy()
    supply_map.index = supply_map.index.floor("D")
    df["supply"] = df["timestamp"].dt.floor("D").map(supply_map)
    df["supply"] = df["supply"].ffill()
    if df["supply"].isna().any():
        df["supply"] = df["supply"].ffill()
    return df


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
            "market_cap": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )
    
    


df = get_eth_hourly_ohlc(
    period="30d",
    caller="backtest_v2",
    config=MarketCapConfig(etherscan_api_key="YOUR_KEY"),
)
print(df.head(200))