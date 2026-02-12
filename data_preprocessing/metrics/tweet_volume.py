from typing import Optional, Union

import pandas as pd
import requests

from .common.config import PipelineConfig
from .common.io_utils import maybe_save_csv
from .common.time_utils import resolve_time_range
from .common.transforms import to_unix_timestamp


def fetch_tweet_volume(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    period: Optional[str] = None,
    config: PipelineConfig = PipelineConfig(),
    save: bool = True,
    as_unix: bool = True,
) -> pd.DataFrame:
    """
    Fetch ETH-related tweet volume using the Twitter counts endpoint.

    This implementation assumes you supply a valid bearer token in
    `config.twitter_bearer_token`. If no token is provided or the request
    fails, an empty DataFrame is returned and (optionally) saved.
    """
    if start is not None and period is not None:
        raise ValueError("Use either start/end or period, not both.")

    start_ts, end_ts = resolve_time_range(start, end, period)

    if not config.twitter_bearer_token:
        if config.debug:
            print("[DEBUG] twitter_bearer_token not set; tweet volume skipped")
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "eth_tweet_volume.csv", enabled=save)
        return output

    df = _get_tweet_counts(
        start_ts=start_ts,
        end_ts=end_ts,
        config=config,
    )

    if df.empty:
        output = pd.DataFrame()
        maybe_save_csv(output, config.output_dir, "eth_tweet_volume.csv", enabled=save)
        return output

    output = df.copy()
    if as_unix:
        output = to_unix_timestamp(output, "timestamp")

    maybe_save_csv(output, config.output_dir, "eth_tweet_volume.csv", enabled=save)
    return output


def _get_tweet_counts(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    config: PipelineConfig,
) -> pd.DataFrame:
    headers = {
        "Authorization": f"Bearer {config.twitter_bearer_token}",
    }
    params = {
        "query": config.twitter_query,
        "granularity": "day",
        "start_time": start_ts.isoformat(),
        "end_time": end_ts.isoformat(),
    }

    try:
        response = requests.get(
            config.twitter_search_url,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        if config.debug:
            print(f"[DEBUG] tweet_volume request failed: {exc}")
        return pd.DataFrame()

    try:
        data = response.json()
    except ValueError as exc:
        if config.debug:
            print(f"[DEBUG] tweet_volume JSON parse failed: {exc}")
        return pd.DataFrame()

    buckets = data.get("data", [])
    if not buckets:
        return pd.DataFrame()

    rows = []
    for item in buckets:
        # Twitter returns ISO8601 start time for each bucket
        ts = pd.to_datetime(item.get("start"), utc=True, errors="coerce")
        count = item.get("tweet_count")
        rows.append({"timestamp": ts, "tweet_volume": count})

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["timestamp"])
    df["tweet_volume"] = pd.to_numeric(df["tweet_volume"], errors="coerce")
    df = df.dropna(subset=["tweet_volume"])
    df = df.sort_values("timestamp")
    return df

