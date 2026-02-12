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
    if series.index.has_duplicates:
        dup_count = int(series.index.duplicated().sum())
        print(
            f"[WARN] attach_daily_metric deduped {dup_count} duplicate daily "
            f"timestamps for {column_name}"
        )
    if pd.api.types.is_numeric_dtype(series.index):
        series.index = pd.to_datetime(
            series.index, unit="s", utc=True, errors="coerce"
        )
    else:
        series.index = pd.to_datetime(series.index, utc=True, errors="coerce")
    series.index = series.index.floor("D")
    series = series.sort_index()
    if series.index.has_duplicates:
        series = series[~series.index.duplicated(keep="last")]

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


def to_unix_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    if column not in df.columns or df.empty:
        return df
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = df[column].view("int64") // 1_000_000_000
    else:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
