from typing import Optional, Union

import pandas as pd
import requests

from .common.config import PipelineConfig
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range
from .common.transforms import build_daily_series, to_unix_timestamp


def fetch_eth_supply(
    config: PipelineConfig = PipelineConfig(),
) -> Optional[float]:
    params = {
        "chainid": str(config.etherscan_chain_id),
        "module": "stats",
        "action": "ethsupply",
    }
    if config.etherscan_api_key:
        params["apikey"] = config.etherscan_api_key

    try:
        response = requests.get(config.etherscan_api_base, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        if config.debug:
            print(f"[DEBUG] eth_supply request failed: {exc}")
        return None

    if str(payload.get("status")) != "1":
        if config.debug:
            print(f"[DEBUG] eth_supply bad status: {payload}")
        return None

    result = payload.get("result")
    if result is None:
        if config.debug:
            print("[DEBUG] eth_supply missing result")
        return None

    try:
        supply_wei = int(result)
    except (TypeError, ValueError) as exc:
        if config.debug:
            print(f"[DEBUG] eth_supply parse error: {exc}")
        return None

    return supply_wei / 1_000_000_000_000_000_000


def fetch_eth_supply_daily(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
    as_unix: bool = True,
) -> pd.DataFrame:
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)
    supply_value = fetch_eth_supply(config=config)
    series = build_daily_series(start_ts, end_ts, supply_value)
    df = series.reset_index()
    df.columns = ["timestamp", "supply"]

    output = df.copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "eth_supply_daily.csv", enabled=save)
    return output
