from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    eth_product_id: str = "ETH-USD"
    btc_product_id: str = "BTC-USD"
    interval_seconds: int = 3600
    max_candles_per_call: int = 300
    coinbase_api_base: str = "https://api.exchange.coinbase.com"
    etherscan_api_base: str = "https://api.etherscan.io/v2/api"
    etherscan_api_key: Optional[str] = None
    etherscan_chain_id: int = 1
    etherscan_chart_tx_url: str = "https://etherscan.io/chart/tx"
    output_dir: str = "data_preprocessing/data/metrics"
    rolling_beta_window: int = 24
    google_trend_keyword: str = "Ethereum"
    google_trend_geo: str = ""
    google_trend_hl: str = "en-US"
    google_trend_tz: int = 0
    google_trend_category: int = 0
    google_trend_chunk_days: int = 180
    google_trend_overlap_days: int = 7
    google_trend_anchor_days: int = 90
    google_trend_max_retries: int = 5
    google_trend_backoff_seconds: float = 5.0
    debug: bool = False

    # Twitter / social metrics (optional)
    twitter_bearer_token: Optional[str] = None
    twitter_search_url: str = "https://api.twitter.com/2/tweets/counts/recent"
    twitter_query: str = "ethereum OR eth -is:retweet"

    # BTC hashrate (optional)
    btc_hashrate_url: str = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=csv"
