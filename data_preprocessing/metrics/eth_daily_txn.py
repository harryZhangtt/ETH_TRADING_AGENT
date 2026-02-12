import io
from typing import Optional, Union

import pandas as pd
import requests

from .common.config import PipelineConfig
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range


def fetch_eth_daily_txn(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
) -> pd.DataFrame:
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)
    series = _get_eth_daily_transaction(
        api_url=config.etherscan_chart_tx_url,
        api_key=config.etherscan_api_key,
        start=start_ts,
        end=end_ts,
        debug=config.debug,
    )
    df = series.reset_index()
    df.columns = ["timestamp", "eth_daily_tx"]

    maybe_save_csv(df, config.output_dir, "eth_daily_txn.csv", enabled=save)
    return df


def _get_eth_daily_transaction(
    api_url: str,
    api_key: Optional[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    debug: bool = False,
) -> pd.Series:
    def _dbg(message: str) -> None:
        if debug:
            print(message)

    params = {"output": "csv"}
    if api_key:
        params["apikey"] = api_key

    safe_params = dict(params)
    if "apikey" in safe_params:
        safe_params["apikey"] = "***"
    _dbg(f"[DEBUG] eth_daily_tx request: url={api_url} params={safe_params}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=30)
    except (requests.RequestException, ValueError) as exc:
        _dbg(f"[DEBUG] eth_daily_tx request failed: {exc}")
        return pd.Series(dtype="float64")

    _dbg(
        "[DEBUG] eth_daily_tx response: "
        f"status={response.status_code} content_type={response.headers.get('Content-Type')} "
        f"len={len(response.text)}"
    )

    if not response.ok:
        snippet = response.text[:300].replace("\n", " ")
        _dbg(f"[DEBUG] eth_daily_tx non-200 body snippet: {snippet}")
        return pd.Series(dtype="float64")

    text = response.text.strip()
    if not text:
        _dbg("[DEBUG] eth_daily_tx empty response body")
        return pd.Series(dtype="float64")

    _dbg(f"[DEBUG] eth_daily_tx raw snippet: {text[:200].replace('\\n', ' ')}")

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        _dbg(f"[DEBUG] eth_daily_tx CSV parse failed: {exc}")
        return pd.Series(dtype="float64")

    if df.empty:
        _dbg("[DEBUG] eth_daily_tx CSV empty after parse")
        return pd.Series(dtype="float64")

    _dbg(f"[DEBUG] eth_daily_tx columns: {list(df.columns)} rows={len(df)}")

    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    if len(df.columns) > 1:
        value_col = next(
            (c for c in df.columns if "value" in c.lower() or "tx" in c.lower()),
            df.columns[1],
        )
    else:
        value_col = df.columns[0]

    df = df[[date_col, value_col]].rename(
        columns={date_col: "timestamp", value_col: "eth_daily_tx"}
    )
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    ).dt.floor("D")
    df["eth_daily_tx"] = pd.to_numeric(df["eth_daily_tx"], errors="coerce")
    df = df.dropna(subset=["timestamp", "eth_daily_tx"]).sort_values("timestamp")

    if df.empty:
        _dbg("[DEBUG] eth_daily_tx empty after cleanup")
        return pd.Series(dtype="float64")

    if start is not None:
        df = df[df["timestamp"] >= start.floor("D")]
    if end is not None:
        df = df[df["timestamp"] < end.floor("D") + pd.Timedelta(days=1)]

    if df.empty:
        _dbg("[DEBUG] eth_daily_tx empty after date filtering")
        return pd.Series(dtype="float64")

    _dbg(
        "[DEBUG] eth_daily_tx final range: "
        f"{df['timestamp'].min()} -> {df['timestamp'].max()} rows={len(df)}"
    )

    return df.set_index("timestamp")["eth_daily_tx"].astype("float64")
