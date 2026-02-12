from typing import Iterable, Optional, Tuple, Union

import pandas as pd


def to_utc(value: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def resolve_time_range(
    start: Optional[Union[str, pd.Timestamp]],
    end: Optional[Union[str, pd.Timestamp]],
    period: Optional[str],
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if start is None and period is None:
        period = "60d"

    if start is None:
        end_ts = to_utc(end) or now_utc()
        start_ts = end_ts - pd.Timedelta(period)
        return start_ts, end_ts

    start_ts = to_utc(start)
    end_ts = to_utc(end) or now_utc()
    if end_ts <= start_ts:
        raise ValueError("end must be after start.")
    return start_ts, end_ts


def iter_candle_ranges(
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
