"""
V_{t+1} = V_t * (1 - alpha * |w_{t+1} - w_t|) * (1 + w_{t+1} * r_simple)
r_log = log(O_{t+2} / O_{t+1})
r_simple = exp(r_log) - 1
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class PortfolioUpdate:
    V_next: float
    delta_w: float
    cost_multiplier: float
    market_multiplier: float
    r_simple: float


def update_portfolio(
    V: float,
    w: float,
    w_next: float,
    r_mkt: float,
    alpha: float,
    r_is_log: bool = True,
    clamp_cost: bool = True,
    clamp_r_simple: bool = True,
    weight_bounds: Optional[tuple[float, float]] = (0.0, 1.0),
    eps: float = 1e-12,
) -> PortfolioUpdate:
    if not (V > 0.0):
        raise ValueError(f"V must be > 0, got {V}")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")

    if weight_bounds is not None:
        lo, hi = weight_bounds
        if not (lo <= w <= hi):
            raise ValueError(f"w out of bounds {weight_bounds}: {w}")
        if not (lo <= w_next <= hi):
            raise ValueError(f"w_next out of bounds {weight_bounds}: {w_next}")

    if r_is_log:
        r_simple = math.expm1(r_mkt)
    else:
        r_simple = r_mkt

    if clamp_r_simple:
        r_simple = max(r_simple, -1.0 + eps)

    delta_w = w_next - w
    cost_multiplier = 1.0 - alpha * abs(delta_w)
    if clamp_cost:
        cost_multiplier = max(0.0, cost_multiplier)

    market_multiplier = 1.0 + w_next * r_simple
    market_multiplier = max(eps, market_multiplier)

    V_next = V * cost_multiplier * market_multiplier
    V_next = max(eps, V_next)

    return PortfolioUpdate(
        V_next=V_next,
        delta_w=delta_w,
        cost_multiplier=cost_multiplier,
        market_multiplier=market_multiplier,
        r_simple=r_simple,
    )
