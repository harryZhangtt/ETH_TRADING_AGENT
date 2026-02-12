from typing import List, Optional

import pandas as pd
import requests

from .time_utils import iter_candle_ranges


def fetch_coinbase_candles(
    product_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_seconds: int,
    max_candles: int,
    api_base: str,
) -> pd.DataFrame:
    session = requests.Session()
    frames: List[pd.DataFrame] = []
    for chunk_start, chunk_end in iter_candle_ranges(
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


def format_ohlc_frame(
    raw: pd.DataFrame,
    product_id: str,
    caller: Optional[str],
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
