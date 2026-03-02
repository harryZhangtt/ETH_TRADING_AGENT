"""
Independent unit tests for rewards/portfolio.py.

Run from project root:
    pytest rewards/test_portfolio.py -v

Coverage:
  - Exact field values (V_next, delta_w, cost_multiplier, market_multiplier, r_simple)
    for every PortfolioUpdate returned
  - Log-return vs. simple-return mode (r_is_log flag)
  - Transaction cost variants: zero alpha, partial alpha, alpha that saturates to zero
  - clamp_cost=True/False
  - clamp_r_simple=True/False with a super-negative simple return
  - weight_bounds enforcement and bypass
  - V <= 0 and alpha < 0 input validation
  - Combined cost + market return scenarios
"""

from __future__ import annotations

import math
import os
import sys

import pytest

# Make `from rewards.X import ...` work when the file is run directly
# (pytest from project root already has the root on sys.path)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rewards.portfolio import PortfolioUpdate, update_portfolio

# ── Constants ──────────────────────────────────────────────────────────────
_EPS: float = 1e-12          # must match the default `eps` in update_portfolio
_REL: float = 1e-9           # relative tolerance for pytest.approx
_ABS: float = 1e-14          # absolute tolerance for quantities near zero


# ── Assertion helper ───────────────────────────────────────────────────────

def _assert_eq(actual: PortfolioUpdate, expected: PortfolioUpdate) -> None:
    """Field-by-field comparison with tight tolerances."""
    assert actual.V_next == pytest.approx(expected.V_next, rel=_REL, abs=_ABS), (
        f"V_next mismatch: actual={actual.V_next!r}  expected={expected.V_next!r}"
    )
    assert actual.delta_w == pytest.approx(expected.delta_w, rel=_REL, abs=_ABS), (
        f"delta_w mismatch: actual={actual.delta_w!r}  expected={expected.delta_w!r}"
    )
    assert actual.cost_multiplier == pytest.approx(expected.cost_multiplier, rel=_REL, abs=_ABS), (
        f"cost_multiplier mismatch: actual={actual.cost_multiplier!r}  expected={expected.cost_multiplier!r}"
    )
    assert actual.market_multiplier == pytest.approx(expected.market_multiplier, rel=_REL, abs=_ABS), (
        f"market_multiplier mismatch: actual={actual.market_multiplier!r}  expected={expected.market_multiplier!r}"
    )
    assert actual.r_simple == pytest.approx(expected.r_simple, rel=_REL, abs=_ABS), (
        f"r_simple mismatch: actual={actual.r_simple!r}  expected={expected.r_simple!r}"
    )


# ── Return-type sanity ─────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_portfolio_update_instance(self) -> None:
        result = update_portfolio(V=1000.0, w=0.5, w_next=0.5, r_mkt=0.0, alpha=0.0)
        assert isinstance(result, PortfolioUpdate)

    def test_portfolio_update_is_frozen(self) -> None:
        result = update_portfolio(V=1000.0, w=0.5, w_next=0.5, r_mkt=0.0, alpha=0.0)
        with pytest.raises((AttributeError, TypeError)):
            result.V_next = 999.0  # type: ignore[misc]


# ── Zero-cost, zero-return baseline ───────────────────────────────────────

class TestNoTradeNoReturn:
    """Hold same weight, zero log return — all multipliers are 1."""

    def test_all_fields_are_identity(self) -> None:
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.5, w_next=0.5, r_mkt=0.0, alpha=0.001
        )
        expected = PortfolioUpdate(
            V_next=1000.0,
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=1.0,
            r_simple=0.0,
        )
        _assert_eq(result, expected)

    def test_unit_portfolio_unchanged(self) -> None:
        result: PortfolioUpdate = update_portfolio(
            V=1.0, w=0.0, w_next=0.0, r_mkt=0.0, alpha=0.0
        )
        expected = PortfolioUpdate(
            V_next=1.0,
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=1.0,
            r_simple=0.0,
        )
        _assert_eq(result, expected)


# ── Transaction cost ───────────────────────────────────────────────────────

class TestTransactionCost:
    """alpha > 0 penalises weight changes."""

    def test_buy_in_exact_cost(self) -> None:
        """w: 0.0 → 0.4, alpha=0.1 ⟹ cost_multiplier=0.96, zero return."""
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.0, w_next=0.4, r_mkt=0.0, alpha=0.1
        )
        expected = PortfolioUpdate(
            V_next=960.0,        # 1000 * 0.96 * 1.0
            delta_w=0.4,
            cost_multiplier=0.96,
            market_multiplier=1.0,
            r_simple=0.0,
        )
        _assert_eq(result, expected)

    def test_sell_off_symmetric_cost(self) -> None:
        """w: 0.4 → 0.0, alpha=0.1 — |delta_w|=0.4, same magnitude as buy-in."""
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.4, w_next=0.0, r_mkt=0.0, alpha=0.1
        )
        expected = PortfolioUpdate(
            V_next=960.0,
            delta_w=-0.4,
            cost_multiplier=0.96,
            market_multiplier=1.0,
            r_simple=0.0,
        )
        _assert_eq(result, expected)

    def test_zero_alpha_no_cost_regardless_of_weight_change(self) -> None:
        """Full rebalance 0 → 1 with alpha=0 leaves cost_multiplier=1."""
        result: PortfolioUpdate = update_portfolio(
            V=500.0, w=0.0, w_next=1.0, r_mkt=0.0, alpha=0.0
        )
        expected = PortfolioUpdate(
            V_next=500.0,
            delta_w=1.0,
            cost_multiplier=1.0,
            market_multiplier=1.0,
            r_simple=0.0,
        )
        _assert_eq(result, expected)

    def test_partial_cost_small_alpha(self) -> None:
        """delta_w=0.2, alpha=0.05 ⟹ cost_multiplier=0.99."""
        result: PortfolioUpdate = update_portfolio(
            V=2000.0, w=0.3, w_next=0.5, r_mkt=0.0, alpha=0.05
        )
        expected = PortfolioUpdate(
            V_next=2000.0 * 0.99,  # cost_multiplier = 1 - 0.05*0.2 = 0.99
            delta_w=0.2,
            cost_multiplier=0.99,
            market_multiplier=1.0,
            r_simple=0.0,
        )
        _assert_eq(result, expected)


# ── Market return (log-return mode) ───────────────────────────────────────

class TestMarketReturnLog:
    """r_is_log=True (default): r_mkt is a log return."""

    def test_positive_log_return_full_exposure(self) -> None:
        """w=1.0, r_log=log(1.1) ⟹ V grows by factor 1.1."""
        r_log: float = math.log(1.1)
        r_simple: float = math.expm1(r_log)          # ≈ 0.1
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=1.0, w_next=1.0, r_mkt=r_log, alpha=0.0
        )
        market_mm: float = 1.0 + 1.0 * r_simple
        expected = PortfolioUpdate(
            V_next=1000.0 * market_mm,
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=market_mm,
            r_simple=r_simple,
        )
        _assert_eq(result, expected)

    def test_negative_log_return_full_exposure(self) -> None:
        """w=1.0, r_log=log(0.9) ⟹ V shrinks by factor 0.9."""
        r_log: float = math.log(0.9)
        r_simple: float = math.expm1(r_log)          # ≈ -0.1
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=1.0, w_next=1.0, r_mkt=r_log, alpha=0.0
        )
        market_mm: float = 1.0 + 1.0 * r_simple
        expected = PortfolioUpdate(
            V_next=1000.0 * market_mm,
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=market_mm,
            r_simple=r_simple,
        )
        _assert_eq(result, expected)

    def test_half_exposure_halves_market_impact(self) -> None:
        """w=0.5 halves the market impact relative to full exposure."""
        r_log: float = math.log(1.2)
        r_simple: float = math.expm1(r_log)
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.5, w_next=0.5, r_mkt=r_log, alpha=0.0
        )
        market_mm: float = 1.0 + 0.5 * r_simple
        expected = PortfolioUpdate(
            V_next=1000.0 * market_mm,
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=market_mm,
            r_simple=r_simple,
        )
        _assert_eq(result, expected)

    def test_zero_weight_no_market_exposure(self) -> None:
        """w_next=0 ⟹ market return does not affect V regardless of r_mkt."""
        result: PortfolioUpdate = update_portfolio(
            V=2000.0, w=0.0, w_next=0.0, r_mkt=math.log(3.0), alpha=0.0
        )
        expected = PortfolioUpdate(
            V_next=2000.0,
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=1.0,
            r_simple=2.0,              # expm1(log(3)) = 2.0
        )
        _assert_eq(result, expected)


# ── Market return (simple-return mode) ────────────────────────────────────

class TestMarketReturnSimple:
    """r_is_log=False: r_mkt is already a simple return, passed through directly."""

    def test_simple_return_positive(self) -> None:
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.5, w_next=0.5, r_mkt=0.2, alpha=0.0, r_is_log=False
        )
        expected = PortfolioUpdate(
            V_next=1100.0,   # 1000 * (1 + 0.5*0.2)
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=1.1,
            r_simple=0.2,
        )
        _assert_eq(result, expected)

    def test_simple_return_zero(self) -> None:
        result: PortfolioUpdate = update_portfolio(
            V=500.0, w=0.5, w_next=0.5, r_mkt=0.0, alpha=0.0, r_is_log=False
        )
        expected = PortfolioUpdate(
            V_next=500.0,
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=1.0,
            r_simple=0.0,
        )
        _assert_eq(result, expected)

    def test_r_simple_stored_verbatim(self) -> None:
        """r_simple field equals r_mkt exactly when r_is_log=False."""
        raw: float = 0.123456789
        result: PortfolioUpdate = update_portfolio(
            V=1.0, w=0.5, w_next=0.5, r_mkt=raw, alpha=0.0, r_is_log=False
        )
        assert result.r_simple == pytest.approx(raw, rel=1e-15)


# ── Clamping: cost_multiplier ──────────────────────────────────────────────

class TestClampCost:
    """clamp_cost=True prevents a negative cost_multiplier from flipping sign."""

    def test_over_saturated_alpha_clamps_to_zero(self) -> None:
        """alpha=2.0, delta_w=1.0 ⟹ raw=-1.0, clamped to 0 ⟹ V_next→eps."""
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.0, w_next=1.0, r_mkt=0.0, alpha=2.0, clamp_cost=True
        )
        expected = PortfolioUpdate(
            V_next=_EPS,
            delta_w=1.0,
            cost_multiplier=0.0,
            market_multiplier=1.0,
            r_simple=0.0,
        )
        _assert_eq(result, expected)

    def test_exactly_at_saturation_boundary(self) -> None:
        """alpha * |delta_w| == 1.0 ⟹ cost_multiplier == 0.0 exactly."""
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.0, w_next=0.5, r_mkt=0.0, alpha=2.0, clamp_cost=True
        )
        assert result.cost_multiplier == pytest.approx(0.0, abs=_ABS)

    def test_no_clamp_cost_exposes_raw_negative_multiplier(self) -> None:
        """clamp_cost=False: cost_multiplier = -1.0 passes through unchanged."""
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.0, w_next=1.0, r_mkt=0.0, alpha=2.0, clamp_cost=False
        )
        assert result.cost_multiplier == pytest.approx(-1.0, rel=_REL, abs=_ABS)
        assert result.delta_w == pytest.approx(1.0, abs=_ABS)
        assert result.r_simple == pytest.approx(0.0, abs=_ABS)
        assert result.market_multiplier == pytest.approx(1.0, rel=_REL, abs=_ABS)
        # V_next is still protected by the final max(eps, …) guard
        assert result.V_next == pytest.approx(_EPS, rel=1e-6)


# ── Clamping: r_simple ─────────────────────────────────────────────────────

class TestClampRSimple:
    """clamp_r_simple=True prevents r_simple from going below -1+eps."""

    def test_large_negative_simple_return_clamped(self) -> None:
        """r_mkt=-5.0 (simple), clamp ⟹ r_simple=-1+eps."""
        eps: float = _EPS
        r_clamped: float = -1.0 + eps
        market_mm: float = max(eps, 1.0 + 0.5 * r_clamped)
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.5, w_next=0.5, r_mkt=-5.0,
            alpha=0.0, r_is_log=False, clamp_r_simple=True,
        )
        expected = PortfolioUpdate(
            V_next=max(eps, 1000.0 * 1.0 * market_mm),
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=market_mm,
            r_simple=r_clamped,
        )
        _assert_eq(result, expected)

    def test_no_clamp_r_simple_passes_raw_value(self) -> None:
        """clamp_r_simple=False: r_simple=-5.0 flows into market_multiplier."""
        eps: float = _EPS
        r_raw: float = -5.0
        market_mm: float = max(eps, 1.0 + 0.5 * r_raw)   # max(eps, -1.5) = eps
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.5, w_next=0.5, r_mkt=r_raw,
            alpha=0.0, r_is_log=False, clamp_r_simple=False,
        )
        assert result.r_simple == pytest.approx(r_raw, rel=_REL, abs=_ABS)
        assert result.market_multiplier == pytest.approx(market_mm, rel=_REL, abs=_ABS)
        assert result.cost_multiplier == pytest.approx(1.0, abs=_ABS)

    def test_mildly_negative_r_simple_not_clamped(self) -> None:
        """r_simple=-0.5 > -1+eps, so clamp_r_simple has no effect."""
        r_raw: float = -0.5
        result_clamped: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.5, w_next=0.5, r_mkt=r_raw,
            alpha=0.0, r_is_log=False, clamp_r_simple=True,
        )
        result_unclamped: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.5, w_next=0.5, r_mkt=r_raw,
            alpha=0.0, r_is_log=False, clamp_r_simple=False,
        )
        assert result_clamped.r_simple == pytest.approx(result_unclamped.r_simple, rel=_REL)
        assert result_clamped.V_next == pytest.approx(result_unclamped.V_next, rel=_REL)


# ── Weight bounds ──────────────────────────────────────────────────────────

class TestWeightBounds:
    def test_w_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="w out of bounds"):
            update_portfolio(V=1000.0, w=-0.01, w_next=0.5, r_mkt=0.0, alpha=0.0)

    def test_w_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="w out of bounds"):
            update_portfolio(V=1000.0, w=1.01, w_next=0.5, r_mkt=0.0, alpha=0.0)

    def test_w_next_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="w_next out of bounds"):
            update_portfolio(V=1000.0, w=0.5, w_next=-0.01, r_mkt=0.0, alpha=0.0)

    def test_w_next_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="w_next out of bounds"):
            update_portfolio(V=1000.0, w=0.5, w_next=1.01, r_mkt=0.0, alpha=0.0)

    def test_boundary_values_w_zero_and_w_next_one_accepted(self) -> None:
        """Exact boundary values [0.0, 1.0] must not raise."""
        result: PortfolioUpdate = update_portfolio(
            V=500.0, w=0.0, w_next=1.0, r_mkt=0.0, alpha=0.0
        )
        assert result.delta_w == pytest.approx(1.0, abs=_ABS)

    def test_weight_bounds_none_bypasses_validation(self) -> None:
        """weight_bounds=None: any weight value is accepted without error."""
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=3.0, w_next=-2.0, r_mkt=0.0, alpha=0.0, weight_bounds=None
        )
        assert result.delta_w == pytest.approx(-5.0, abs=_ABS)

    def test_custom_weight_bounds_enforced(self) -> None:
        """Custom bounds [-1, 2] accept values out of [0, 1]."""
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=-0.5, w_next=1.5, r_mkt=0.0, alpha=0.0,
            weight_bounds=(-1.0, 2.0),
        )
        assert result.delta_w == pytest.approx(2.0, abs=_ABS)

    def test_custom_weight_bounds_reject_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="w_next out of bounds"):
            update_portfolio(
                V=1000.0, w=0.0, w_next=3.0, r_mkt=0.0, alpha=0.0,
                weight_bounds=(-1.0, 2.0),
            )


# ── Input validation ───────────────────────────────────────────────────────

class TestInputValidation:
    def test_zero_V_raises(self) -> None:
        with pytest.raises(ValueError, match="V must be > 0"):
            update_portfolio(V=0.0, w=0.5, w_next=0.5, r_mkt=0.0, alpha=0.0)

    def test_negative_V_raises(self) -> None:
        with pytest.raises(ValueError, match="V must be > 0"):
            update_portfolio(V=-1000.0, w=0.5, w_next=0.5, r_mkt=0.0, alpha=0.0)

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            update_portfolio(V=1000.0, w=0.5, w_next=0.5, r_mkt=0.0, alpha=-0.001)

    def test_zero_alpha_accepted(self) -> None:
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.5, w_next=0.5, r_mkt=0.0, alpha=0.0
        )
        assert result.cost_multiplier == pytest.approx(1.0, abs=_ABS)


# ── Combined cost + return ─────────────────────────────────────────────────

class TestCombinedCostAndReturn:
    """Both cost and market return are nonzero — exact field checks."""

    def test_buy_with_positive_log_return(self) -> None:
        """
        w: 0.0→0.5, alpha=0.05, r_log=log(1.1)
        cost_mult  = 1 - 0.05*0.5 = 0.975
        r_simple   = expm1(log(1.1))
        market_mult = 1 + 0.5*r_simple
        V_next     = 1000 * 0.975 * market_mult
        """
        alpha: float = 0.05
        w_next: float = 0.5
        r_log: float = math.log(1.1)
        r_simple: float = math.expm1(r_log)
        cost_mult: float = 1.0 - alpha * abs(w_next)
        mkt_mult: float = 1.0 + w_next * r_simple
        v_next: float = 1000.0 * cost_mult * mkt_mult

        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.0, w_next=w_next, r_mkt=r_log, alpha=alpha
        )
        expected = PortfolioUpdate(
            V_next=v_next,
            delta_w=w_next,
            cost_multiplier=cost_mult,
            market_multiplier=mkt_mult,
            r_simple=r_simple,
        )
        _assert_eq(result, expected)

    def test_sell_with_negative_log_return(self) -> None:
        """
        w: 1.0→0.3, alpha=0.02, r_log=log(0.95)
        delta_w = -0.7, cost_mult = 1 - 0.02*0.7 = 0.986
        """
        w: float = 1.0
        w_next: float = 0.3
        alpha: float = 0.02
        r_log: float = math.log(0.95)
        r_simple: float = math.expm1(r_log)
        delta_w: float = w_next - w         # -0.7
        cost_mult: float = 1.0 - alpha * abs(delta_w)
        mkt_mult: float = 1.0 + w_next * r_simple
        v_next: float = 1000.0 * cost_mult * mkt_mult

        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=w, w_next=w_next, r_mkt=r_log, alpha=alpha
        )
        expected = PortfolioUpdate(
            V_next=v_next,
            delta_w=delta_w,
            cost_multiplier=cost_mult,
            market_multiplier=mkt_mult,
            r_simple=r_simple,
        )
        _assert_eq(result, expected)

    def test_no_weight_change_only_market_moves(self) -> None:
        """delta_w=0 ⟹ cost_multiplier=1, only market_multiplier changes V."""
        r_log: float = math.log(1.05)
        r_simple: float = math.expm1(r_log)
        w: float = 0.6
        mkt_mult: float = 1.0 + w * r_simple

        result: PortfolioUpdate = update_portfolio(
            V=800.0, w=w, w_next=w, r_mkt=r_log, alpha=0.1
        )
        expected = PortfolioUpdate(
            V_next=800.0 * mkt_mult,
            delta_w=0.0,
            cost_multiplier=1.0,
            market_multiplier=mkt_mult,
            r_simple=r_simple,
        )
        _assert_eq(result, expected)

    def test_delta_w_sign_negative_when_selling(self) -> None:
        """Selling: w_next < w ⟹ delta_w is negative."""
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.8, w_next=0.2, r_mkt=0.0, alpha=0.0
        )
        assert result.delta_w == pytest.approx(-0.6, rel=_REL, abs=_ABS)

    def test_delta_w_sign_positive_when_buying(self) -> None:
        """Buying: w_next > w ⟹ delta_w is positive."""
        result: PortfolioUpdate = update_portfolio(
            V=1000.0, w=0.1, w_next=0.9, r_mkt=0.0, alpha=0.0
        )
        assert result.delta_w == pytest.approx(0.8, rel=_REL, abs=_ABS)
