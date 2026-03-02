"""
R_t = log(V_{t+1} / V_t) - lambda_risk * w_{t+1}^2 * sigma_t^2
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    log_growth: float
    risk_penalty: float


def compute_reward(
    V: float,
    V_next: float,
    w_next: float,
    sigma: float,
    lambda_risk: float,
    eps: float = 1e-12,
    sigma_is_std: bool = True,
    weight_bounds: Optional[tuple[float, float]] = (0.0, 1.0),
    clamp_values: bool = True,
) -> RewardBreakdown:
    if lambda_risk < 0.0:
        raise ValueError(f"lambda_risk must be >= 0, got {lambda_risk}")

    if weight_bounds is not None:
        lo, hi = weight_bounds
        if not (lo <= w_next <= hi):
            raise ValueError(f"w_next out of bounds {weight_bounds}: {w_next}")

    if clamp_values:
        V_safe = max(eps, float(V))
        V_next_safe = max(eps, float(V_next))
        sigma_safe = max(0.0, float(sigma))
    else:
        if not (V > 0.0):
            raise ValueError(f"V must be > 0, got {V}")
        if not (V_next > 0.0):
            raise ValueError(f"V_next must be > 0, got {V_next}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        V_safe = float(V)
        V_next_safe = float(V_next)
        sigma_safe = float(sigma)

    log_growth = math.log(V_next_safe / V_safe)

    sigma_term = sigma_safe * sigma_safe if sigma_is_std else sigma_safe
    risk_penalty = lambda_risk * (w_next * w_next) * sigma_term

    total = log_growth - risk_penalty
    return RewardBreakdown(
        total=total, log_growth=log_growth, risk_penalty=risk_penalty
    )


def compute_reward_from_values(
    V: float,
    V_next: float,
    w_next: float,
    sigma: float,
    lambda_risk: float,
    **kwargs,
) -> float:
    return compute_reward(
        V=V,
        V_next=V_next,
        w_next=w_next,
        sigma=sigma,
        lambda_risk=lambda_risk,
        **kwargs,
    ).total
