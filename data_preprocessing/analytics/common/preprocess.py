from typing import List, Tuple

import numpy as np
import pandas as pd

from metrics.universal_caller import OUTPUT_COLUMNS as METRIC_COLUMNS


def validate_schema(df: pd.DataFrame, required: List[str] = None) -> List[str]:
    required = required or METRIC_COLUMNS
    missing = [col for col in required if col not in df.columns]
    return missing


def ensure_timestamp_datetime(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    if column not in df.columns:
        return df
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], unit="s", utc=True, errors="coerce")
    else:
        df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    return df


def sort_dedup(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    if column in df.columns:
        df = df.sort_values(column)
        df = df.drop_duplicates(subset=[column], keep="last")
    return df


def cast_numeric(df: pd.DataFrame, exclude: List[str] = None) -> pd.DataFrame:
    exclude = set(exclude or ["ticker", "caller", "timestamp"])
    for col in df.columns:
        if col in exclude:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def impute_missing(
    df: pd.DataFrame,
    method: str = "ffill",
    limit: int = None,
) -> pd.DataFrame:
    if method == "ffill":
        return df.ffill(limit=limit)
    if method == "bfill":
        return df.bfill(limit=limit)
    if method == "interpolate":
        return df.interpolate(limit=limit)
    return df


def winsorize(df: pd.DataFrame, limits: float = 0.01) -> pd.DataFrame:
    if limits <= 0:
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lower = df[col].quantile(limits)
        upper = df[col].quantile(1 - limits)
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    if "close" in df.columns:
        df["eth_return"] = np.log(df["close"]).diff()
    if "btc_close" in df.columns:
        df["btc_return"] = np.log(df["btc_close"]).diff()
    return df


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({"missing_ratio": df.isna().mean()})
    report["missing_count"] = df.isna().sum()
    return report


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "timestamp" in numeric_cols:
        numeric_cols.remove("timestamp")
    return df[numeric_cols].copy(), numeric_cols
