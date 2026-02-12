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
    debug: bool = False
