from typing import List, Optional, Union

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


def append_upon_universal_metrics(
    datas: List[str],
    fields_to_append: List[str],
    combined_csv: str = "data/metrics/eth_metrics_combined.csv",
    save: bool = True,
    output_dir: str = "data/metrics",
    output_filename: str = "eth_metrics_combined_macro.csv",
) -> pd.DataFrame:
    """Enrich the cached hourly ETH metrics CSV with daily macro-economic data.

    Reads *combined_csv* (hourly, Unix-second timestamps) and forward-fills
    each daily macro series onto every hourly row, then optionally saves the
    result.

    Parameters
    ----------
    datas:
        Ordered list of paths to macro CSV files.  Each file must contain a
        ``Date`` column (YYYY-MM-DD, business-day cadence) and a ``Close``
        column.
    fields_to_append:
        Column names to assign to the corresponding entry in *datas*.  Must
        be the same length as *datas*.
    combined_csv:
        Path to the base ``eth_metrics_combined.csv`` produced by
        ``build_universal_metrics``.
    save:
        Persist the enriched DataFrame to *output_dir* / *output_filename*.
    output_dir:
        Directory for the saved CSV.
    output_filename:
        Filename for the saved output.

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with macro columns appended.  The ``timestamp``
        column is Unix seconds (int64).  Column order: all OUTPUT_COLUMNS
        that are present, followed by the new macro columns.

    Raises
    ------
    ValueError
        If *datas* and *fields_to_append* have different lengths.
    FileNotFoundError
        If *combined_csv* does not exist.
    """
    if len(datas) != len(fields_to_append):
        raise ValueError(
            f"datas and fields_to_append must have equal length "
            f"(got {len(datas)} vs {len(fields_to_append)})"
        )

    from pathlib import Path

    if not Path(combined_csv).exists():
        raise FileNotFoundError(
            f"Base metrics CSV not found: {combined_csv!r}.  "
            "Run build_universal_metrics first."
        )

    # ── Load base hourly DataFrame ─────────────────────────────────────────
    df = pd.read_csv(combined_csv)
    if df.empty:
        return df

    # Unix-second ints → UTC datetime so .dt accessor works inside
    # attach_daily_metric (which calls df["timestamp"].dt.floor("h")).
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # ── Drop stale macro columns so repeated calls replace, not accumulate ─
    # Handles the case where combined_csv already contains a previous run's
    # output (e.g. eth_metrics_combined_macro.csv fed back as combined_csv).
    stale = [f for f in fields_to_append if f in df.columns]
    if stale:
        df = df.drop(columns=stale)

    # ── Attach each macro series via daily → hourly forward-fill ──────────
    for csv_path, field_name in zip(datas, fields_to_append):
        raw = pd.read_csv(csv_path)

        if raw.empty or "Date" not in raw.columns or "Close" not in raw.columns:
            print(f"[WARN] append_upon_universal_metrics: skipping {csv_path!r} "
                  "(missing Date or Close column)")
            df[field_name] = pd.NA
            continue

        # Build a UTC-indexed daily Series (business-day cadence → gaps are
        # fine; attach_daily_metric will ffill across weekends automatically).
        daily_series: pd.Series = (
            pd.Series(
                raw["Close"].astype("float64").values,
                index=pd.to_datetime(raw["Date"], utc=True),
            )
            .sort_index()
        )
        if daily_series.index.has_duplicates:
            daily_series = daily_series[~daily_series.index.duplicated(keep="last")]

        df = attach_daily_metric(df, daily_series, field_name)

    # ── Convert timestamp back to Unix seconds (matches pipeline convention) ─
    df = to_unix_timestamp(df, "timestamp")

    # ── Stable column ordering: base schema first, then new macro columns ──
    base_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    macro_cols = [f for f in fields_to_append if f in df.columns and f not in base_cols]
    df = df[base_cols + macro_cols]
    
    
    #null value check 
    df= null_value_check(df)
    

    maybe_save_csv(df, output_dir, output_filename, enabled=save)
    return df


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
