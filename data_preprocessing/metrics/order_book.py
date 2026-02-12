from typing import Optional, Union

import pandas as pd
import requests

from .common.config import PipelineConfig
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range
from .common.transforms import build_daily_series, to_unix_timestamp


def fetch_order_book_metrics(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
    as_unix: bool = True,
) -> pd.DataFrame:
    """
    Fetch simple ETH-USD order book metrics from Coinbase and expand them
    as a daily series over the requested range.

    Notes
    -----
    Coinbase does not expose historical order book snapshots, so this
    implementation samples a *current* snapshot and replicates it across
    the requested time range. This keeps the interface consistent with
    other metrics modules and produces a CSV you can later replace with a
    true historical source if desired.
    """
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)

    snapshot = _get_current_order_book(
        eth_product_id=config.eth_product_id,
        api_base=config.coinbase_api_base,
        debug=config.debug,
    )
    if snapshot is None:
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "eth_order_book_metrics.csv", enabled=save)
        return output

    # replicate snapshot as a daily series over the requested range
    index = build_daily_series(start_ts, end_ts, value=1.0).index
    df = pd.DataFrame(index=index).reset_index().rename(columns={"index": "timestamp"})
    for key, value in snapshot.items():
        df[key] = float(value)

    output = df.copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "eth_order_book_metrics.csv", enabled=save)
    return output


def _get_current_order_book(
    eth_product_id: str,
    api_base: str,
    debug: bool = False,
) -> Optional[dict]:
    url = f"{api_base}/products/{eth_product_id}/book"
    params = {"level": 2}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        if debug:
            print(f"[DEBUG] order_book request failed: {exc}")
        return None

    try:
        data = response.json()
    except ValueError as exc:
        if debug:
            print(f"[DEBUG] order_book JSON parse failed: {exc}")
        return None

    bids = data.get("bids", [])
    asks = data.get("asks", [])
    if not bids or not asks:
        if debug:
            print("[DEBUG] order_book empty bids/asks")
        return None

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = best_ask - best_bid
    mid_price = (best_ask + best_bid) / 2.0

    bid_depth = sum(float(level[1]) for level in bids)
    ask_depth = sum(float(level[1]) for level in asks)
    total_depth = bid_depth + ask_depth if (bid_depth + ask_depth) > 0 else 1.0
    imbalance = (bid_depth - ask_depth) / total_depth

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "mid_price": mid_price,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "order_book_imbalance": imbalance,
    }

