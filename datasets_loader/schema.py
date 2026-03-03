"""
datasets_loader/schema.py
=========================
Single source of truth for the feature layout consumed by the loader and the model.

The primary data source is ``data/midway/eth_metrics_midway.csv``, which already
contains every derived feature listed in SEQ_FEATURE_COLS. This file is NOT responsible
for computing those features — it only declares their names and order.

To add or remove a feature:
  1. Update SEQ_FEATURE_COLS below.
  2. Ensure the midway CSV is regenerated with the matching column.
  3. Update ModelConfig.seq_feature_dim in models/model.py to match SEQ_FEATURE_DIM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Tuple


# ── Sequential feature columns (axis-2 of x_seq) ─────────────────────────
# Order is load-order stable — changing order changes the model's input layout.

SEQ_FEATURE_COLS: Final[Tuple[str, ...]] = (
    # price returns
    "single_day_log_return",
    "multi_day_log_return_7",
    "intraday_range_log",
    "btc_single_day_log_return",
    # volume
    "log_volume",
    "log_volume_change",
    # on-chain supply
    "log_supply",
    "log_supply_change",
    # on-chain transactions
    "log_daily_txn",
    # sentiment / search
    "log_google_trend",
    "log_google_trend_change",
    # market structure
    "log_market_beta",
    # social
    "log_tweet_volume",
    "log_tweet_volume_change",
    # network fees
    "log_network_tx_fee",
    "log_network_tx_fee_change",
    # mining
    "log_btc_hashrate",
    "log_btc_hashrate_change",
    # order book
    "log_spread",
    "log_spread_change",
    "log_ask_depth",
    "log_ask_depth_change",
    "log_bid_depth",
    "log_bid_depth_change",
    "log_order_book_imbalance",
    "log_order_book_imbalance_change",
    # network activity
    "log_unique_addr",
    "log_unique_addr_change",
    "log_avg_txn_fee",
    "log_avg_txn_fee_change",
    "log_avg_block_size",
    "log_avg_block_size_change",
    "log_daily_token_transfer",
    "log_daily_token_transfer_change",
    # macro aggregate-> TODO: change this with actual implementation when midway data is done
    "log_macro_economics",
    "log_macro_economics_change",
    # address growth
    "daily_active_ethereum_address_delta_log",
)

SEQ_FEATURE_DIM: Final[int] = len(SEQ_FEATURE_COLS)

# ── Portfolio state (TBD) ─────────────────────────────────────────────────
# Placeholder: single weight in ETH ∈ [0, 1].
# Extend PORT_FEATURE_COLS when the portfolio state design is finalised;
# the loader's _build_port will follow automatically.

PORT_FEATURE_COLS: Final[Tuple[str, ...]] = ("w_eth",)
PORT_DIM: Final[int] = len(PORT_FEATURE_COLS)


# ── Schema dataclasses ────────────────────────────────────────────────────

@dataclass(frozen=True)
class XSeqSchema:
    """Describes the per-timestep feature layout of x_seq [B, T, F]."""

    feature_cols: Tuple[str, ...] = field(default_factory=lambda: SEQ_FEATURE_COLS)
    feature_dim: int = field(default=SEQ_FEATURE_DIM)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_dim", len(self.feature_cols))

    def index_of(self, col: str) -> int:
        """Return the axis-2 index of *col*, raising KeyError if absent."""
        try:
            return self.feature_cols.index(col)
        except ValueError:
            raise KeyError(f"{col!r} is not in XSeqSchema.feature_cols") from None


@dataclass(frozen=True)
class XPortSchema:
    """Describes the portfolio state vector x_port [B, P] (TBD — placeholder)."""

    feature_cols: Tuple[str, ...] = field(default_factory=lambda: PORT_FEATURE_COLS)
    feature_dim: int = field(default=PORT_DIM)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_dim", len(self.feature_cols))


__all__ = [
    "SEQ_FEATURE_COLS",
    "SEQ_FEATURE_DIM",
    "PORT_FEATURE_COLS",
    "PORT_DIM",
    "XSeqSchema",
    "XPortSchema",
]
