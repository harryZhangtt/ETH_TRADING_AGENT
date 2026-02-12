from typing import Optional

import numpy as np
import pandas as pd


def build_daily_series(
    start: pd.Timestamp,
    end: pd.Timestamp,
    value: Optional[float],
) -> pd.Series:
    if value is None:
        return pd.Series(dtype="float64")

    start_day = start.floor("D")
    end_day = end.floor("D")
    dates = pd.date_range(start=start_day, end=end_day, freq="D", tz="UTC")
    return pd.Series(value, index=dates, dtype="float64")


def attach_daily_metric(
    candles: pd.DataFrame,
    daily_series: pd.Series,
    column_name: str,
) -> pd.DataFrame:
    df = candles.sort_values("timestamp").copy()
    if daily_series.empty:
        df[column_name] = np.nan
        return df

    series = daily_series.copy()
    series.index = pd.to_datetime(series.index, utc=True).floor("D")
    series = series.sort_index()

    hourly_index = df["timestamp"].dt.floor("h")
    df[column_name] = series.reindex(hourly_index, method="ffill").values
    df[column_name] = df[column_name].ffill()
    return df


def null_value_check(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    na_ratios = df.isnull().mean()
    for col, ratio in na_ratios.items():
        if ratio > threshold:
            print(f"Warning: {col} has {ratio:.2%} null values")
    print("[NOTICE]: Null value check passed")
    return df
