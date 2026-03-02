"""
Independent unit tests for rewards/reward_fn.py.

Run from project root:
    pytest rewards/test_reward_fn.py -v

Coverage:
  - Exact field values (total, log_growth, risk_penalty) for every RewardBreakdown
  - lambda_risk=0, w_next=0, sigma=0 degenerate cases
  - sigma_is_std=True vs False (squared vs raw variance)
  - clamp_values=True/False — clamping of V, V_next, sigma
  - Input validation (lambda_risk<0, w_next out of bounds, V/V_next<=0, sigma<0)
  - compute_reward_from_values wrapper: return type, value, **kwargs forwarding
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

from rewards.reward_fn import (
    RewardBreakdown,
    compute_reward,
    compute_reward_from_values,
)

# ── Constants ──────────────────────────────────────────────────────────────
_EPS: float = 1e-12
_REL: float = 1e-9
_ABS: float = 1e-14


# ── Assertion helper ───────────────────────────────────────────────────────

def _assert_eq(actual: RewardBreakdown, expected: RewardBreakdown) -> None:
    """Field-by-field comparison with tight tolerances."""
    assert actual.log_growth == pytest.approx(expected.log_growth, rel=_REL, abs=_ABS), (
        f"log_growth mismatch: actual={actual.log_growth!r}  expected={expected.log_growth!r}"
    )
    assert actual.risk_penalty == pytest.approx(expected.risk_penalty, rel=_REL, abs=_ABS), (
        f"risk_penalty mismatch: actual={actual.risk_penalty!r}  expected={expected.risk_penalty!r}"
    )
    assert actual.total == pytest.approx(expected.total, rel=_REL, abs=_ABS), (
        f"total mismatch: actual={actual.total!r}  expected={expected.total!r}"
    )


# ── Return-type sanity ─────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_reward_breakdown_instance(self) -> None:
        result = compute_reward(
            V=1000.0, V_next=1100.0, w_next=0.5, sigma=0.1, lambda_risk=1.0
        )
        assert isinstance(result, RewardBreakdown)

    def test_reward_breakdown_is_frozen(self) -> None:
        result = compute_reward(
            V=1000.0, V_next=1100.0, w_next=0.5, sigma=0.1, lambda_risk=1.0
        )
        with pytest.raises((AttributeError, TypeError)):
            result.total = 0.0  # type: ignore[misc]

    def test_total_equals_log_growth_minus_risk_penalty(self) -> None:
        result = compute_reward(
            V=1000.0, V_next=1100.0, w_next=0.5, sigma=0.2, lambda_risk=2.0
        )
        assert result.total == pytest.approx(
            result.log_growth - result.risk_penalty, rel=1e-12
        )


# ── Core formula correctness ───────────────────────────────────────────────

class TestCoreFormula:
    def test_portfolio_growth_with_risk_penalty(self) -> None:
        """
        V=1000, V_next=1100, w=0.5, σ=0.1, λ=1.0
        log_growth   = log(1100/1000)
        risk_penalty = 1.0 * 0.5² * 0.1² = 0.0025
        total        = log(1.1) - 0.0025
        """
        V: float = 1000.0
        V_next: float = 1100.0
        w_next: float = 0.5
        sigma: float = 0.1
        lam: float = 1.0

        log_growth: float = math.log(V_next / V)
        risk_penalty: float = lam * (w_next ** 2) * (sigma ** 2)
        total: float = log_growth - risk_penalty

        result: RewardBreakdown = compute_reward(
            V=V, V_next=V_next, w_next=w_next, sigma=sigma, lambda_risk=lam
        )
        expected = RewardBreakdown(
            total=total, log_growth=log_growth, risk_penalty=risk_penalty
        )
        _assert_eq(result, expected)

    def test_portfolio_loss_with_risk_penalty(self) -> None:
        """
        V=1000, V_next=900 → log_growth negative; risk adds further loss.
        """
        V: float = 1000.0
        V_next: float = 900.0
        w_next: float = 0.5
        sigma: float = 0.2
        lam: float = 0.5

        log_growth: float = math.log(900.0 / 1000.0)
        risk_penalty: float = 0.5 * (0.5 ** 2) * (0.2 ** 2)
        total: float = log_growth - risk_penalty

        result: RewardBreakdown = compute_reward(
            V=V, V_next=V_next, w_next=w_next, sigma=sigma, lambda_risk=lam
        )
        expected = RewardBreakdown(
            total=total, log_growth=log_growth, risk_penalty=risk_penalty
        )
        _assert_eq(result, expected)

    def test_flat_portfolio_only_risk_penalty(self) -> None:
        """V == V_next ⟹ log_growth=0, total = -risk_penalty."""
        w_next: float = 0.3
        sigma: float = 0.05
        lam: float = 2.0
        risk_penalty: float = lam * (w_next ** 2) * (sigma ** 2)

        result: RewardBreakdown = compute_reward(
            V=500.0, V_next=500.0, w_next=w_next, sigma=sigma, lambda_risk=lam
        )
        expected = RewardBreakdown(
            total=-risk_penalty, log_growth=0.0, risk_penalty=risk_penalty
        )
        _assert_eq(result, expected)

    def test_high_lambda_dominates_reward(self) -> None:
        """Large lambda makes total strongly negative even with positive return."""
        V: float = 1000.0
        V_next: float = 1010.0
        w_next: float = 1.0
        sigma: float = 0.5
        lam: float = 100.0

        log_growth: float = math.log(1010.0 / 1000.0)
        risk_penalty: float = lam * (w_next ** 2) * (sigma ** 2)
        total: float = log_growth - risk_penalty

        result: RewardBreakdown = compute_reward(
            V=V, V_next=V_next, w_next=w_next, sigma=sigma, lambda_risk=lam
        )
        expected = RewardBreakdown(
            total=total, log_growth=log_growth, risk_penalty=risk_penalty
        )
        _assert_eq(result, expected)
        assert result.total < 0.0


# ── Degenerate parameter cases ─────────────────────────────────────────────

class TestDegenerateCases:
    def test_zero_lambda_no_risk_penalty(self) -> None:
        """lambda_risk=0 ⟹ risk_penalty=0 and total=log_growth."""
        V_next: float = 1200.0
        log_growth: float = math.log(1200.0 / 1000.0)

        result: RewardBreakdown = compute_reward(
            V=1000.0, V_next=V_next, w_next=0.8, sigma=0.3, lambda_risk=0.0
        )
        expected = RewardBreakdown(
            total=log_growth, log_growth=log_growth, risk_penalty=0.0
        )
        _assert_eq(result, expected)

    def test_zero_w_next_no_risk_penalty(self) -> None:
        """w_next=0 ⟹ w_next²=0, so risk_penalty=0 regardless of σ and λ."""
        log_growth: float = math.log(1100.0 / 1000.0)

        result: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1100.0, w_next=0.0, sigma=0.5, lambda_risk=10.0
        )
        expected = RewardBreakdown(
            total=log_growth, log_growth=log_growth, risk_penalty=0.0
        )
        _assert_eq(result, expected)

    def test_zero_sigma_no_risk_penalty(self) -> None:
        """sigma=0 ⟹ sigma²=0 ⟹ risk_penalty=0."""
        log_growth: float = math.log(1100.0 / 1000.0)

        result: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1100.0, w_next=0.5, sigma=0.0, lambda_risk=5.0
        )
        expected = RewardBreakdown(
            total=log_growth, log_growth=log_growth, risk_penalty=0.0
        )
        _assert_eq(result, expected)

    def test_w_next_one_full_weight_risk(self) -> None:
        """w_next=1 ⟹ risk_penalty = lambda * sigma²."""
        V: float = 1000.0
        V_next: float = 1000.0
        sigma: float = 0.3
        lam: float = 2.0
        risk_penalty: float = lam * 1.0 * (sigma ** 2)

        result: RewardBreakdown = compute_reward(
            V=V, V_next=V_next, w_next=1.0, sigma=sigma, lambda_risk=lam
        )
        assert result.risk_penalty == pytest.approx(risk_penalty, rel=_REL, abs=_ABS)
        assert result.log_growth == pytest.approx(0.0, abs=_ABS)
        assert result.total == pytest.approx(-risk_penalty, rel=_REL, abs=_ABS)


# ── sigma_is_std flag ──────────────────────────────────────────────────────

class TestSigmaMode:
    """sigma_is_std=True squares sigma; False treats it as variance directly."""

    def test_sigma_is_std_true_squares_sigma(self) -> None:
        """sigma_term = sigma² when sigma_is_std=True."""
        V: float = 1000.0
        V_next: float = 1000.0
        w_next: float = 0.5
        sigma: float = 0.2
        lam: float = 3.0
        risk_penalty_expected: float = lam * (w_next ** 2) * (sigma ** 2)

        result: RewardBreakdown = compute_reward(
            V=V, V_next=V_next, w_next=w_next, sigma=sigma,
            lambda_risk=lam, sigma_is_std=True,
        )
        assert result.risk_penalty == pytest.approx(risk_penalty_expected, rel=_REL, abs=_ABS)

    def test_sigma_is_std_false_uses_sigma_directly(self) -> None:
        """sigma_term = sigma (variance) when sigma_is_std=False."""
        V: float = 1000.0
        V_next: float = 1000.0
        w_next: float = 0.5
        sigma: float = 0.04   # treated as variance directly
        lam: float = 2.0
        risk_penalty_expected: float = lam * (w_next ** 2) * sigma   # NOT sigma²

        result: RewardBreakdown = compute_reward(
            V=V, V_next=V_next, w_next=w_next, sigma=sigma,
            lambda_risk=lam, sigma_is_std=False,
        )
        expected = RewardBreakdown(
            total=0.0 - risk_penalty_expected,
            log_growth=0.0,
            risk_penalty=risk_penalty_expected,
        )
        _assert_eq(result, expected)

    def test_std_and_variance_modes_are_equivalent_when_sigma_squared_matches(self) -> None:
        """
        std mode with σ=0.2  ≡  variance mode with variance=0.04 (=0.2²).
        risk_penalty must be identical.
        """
        std: float = 0.2
        variance: float = std ** 2
        w_next: float = 0.7
        lam: float = 1.5

        result_std: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1100.0, w_next=w_next, sigma=std,
            lambda_risk=lam, sigma_is_std=True,
        )
        result_var: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1100.0, w_next=w_next, sigma=variance,
            lambda_risk=lam, sigma_is_std=False,
        )
        assert result_std.risk_penalty == pytest.approx(result_var.risk_penalty, rel=_REL)
        assert result_std.total == pytest.approx(result_var.total, rel=_REL)

    def test_sigma_std_and_variance_differ_when_sigma_not_unit(self) -> None:
        """sigma_is_std=True and False give different risk_penalty for same sigma value."""
        result_std: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1000.0, w_next=1.0, sigma=0.3,
            lambda_risk=1.0, sigma_is_std=True,
        )
        result_var: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1000.0, w_next=1.0, sigma=0.3,
            lambda_risk=1.0, sigma_is_std=False,
        )
        assert result_std.risk_penalty != pytest.approx(result_var.risk_penalty, rel=1e-3)


# ── clamp_values=True ──────────────────────────────────────────────────────

class TestClampValues:
    """clamp_values=True replaces bad inputs with eps / 0 instead of raising."""

    def test_negative_V_clamped_raises_no_error(self) -> None:
        """clamp_values=True: V=-100 is replaced with eps, no ValueError."""
        result: RewardBreakdown = compute_reward(
            V=-100.0, V_next=1000.0, w_next=0.5, sigma=0.1,
            lambda_risk=1.0, clamp_values=True,
        )
        log_growth_expected: float = math.log(1000.0 / _EPS)
        assert result.log_growth == pytest.approx(log_growth_expected, rel=_REL, abs=_ABS)

    def test_negative_V_next_clamped_to_eps(self) -> None:
        """clamp_values=True: V_next=-500 is replaced with eps."""
        result: RewardBreakdown = compute_reward(
            V=1000.0, V_next=-500.0, w_next=0.5, sigma=0.1,
            lambda_risk=1.0, clamp_values=True,
        )
        log_growth_expected: float = math.log(_EPS / 1000.0)
        assert result.log_growth == pytest.approx(log_growth_expected, rel=_REL, abs=_ABS)

    def test_negative_sigma_clamped_to_zero_removes_risk_penalty(self) -> None:
        """clamp_values=True: sigma<0 → sigma_safe=0 → risk_penalty=0."""
        V: float = 1000.0
        V_next: float = 1100.0
        log_growth: float = math.log(V_next / V)

        result: RewardBreakdown = compute_reward(
            V=V, V_next=V_next, w_next=0.5, sigma=-0.5,
            lambda_risk=2.0, clamp_values=True,
        )
        expected = RewardBreakdown(
            total=log_growth, log_growth=log_growth, risk_penalty=0.0
        )
        _assert_eq(result, expected)

    def test_zero_V_clamped_to_eps(self) -> None:
        """clamp_values=True: V=0 is replaced with eps (no division-by-zero)."""
        result: RewardBreakdown = compute_reward(
            V=0.0, V_next=1000.0, w_next=0.5, sigma=0.1,
            lambda_risk=1.0, clamp_values=True,
        )
        log_growth_expected: float = math.log(1000.0 / _EPS)
        assert result.log_growth == pytest.approx(log_growth_expected, rel=_REL, abs=_ABS)


# ── clamp_values=False validation ─────────────────────────────────────────

class TestNoClampValidation:
    """clamp_values=False raises ValueError for invalid V, V_next, sigma."""

    def test_zero_V_raises(self) -> None:
        with pytest.raises(ValueError, match="V must be > 0"):
            compute_reward(
                V=0.0, V_next=1000.0, w_next=0.5, sigma=0.1,
                lambda_risk=1.0, clamp_values=False,
            )

    def test_negative_V_raises(self) -> None:
        with pytest.raises(ValueError, match="V must be > 0"):
            compute_reward(
                V=-1.0, V_next=1000.0, w_next=0.5, sigma=0.1,
                lambda_risk=1.0, clamp_values=False,
            )

    def test_zero_V_next_raises(self) -> None:
        with pytest.raises(ValueError, match="V_next must be > 0"):
            compute_reward(
                V=1000.0, V_next=0.0, w_next=0.5, sigma=0.1,
                lambda_risk=1.0, clamp_values=False,
            )

    def test_negative_V_next_raises(self) -> None:
        with pytest.raises(ValueError, match="V_next must be > 0"):
            compute_reward(
                V=1000.0, V_next=-1.0, w_next=0.5, sigma=0.1,
                lambda_risk=1.0, clamp_values=False,
            )

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            compute_reward(
                V=1000.0, V_next=1100.0, w_next=0.5, sigma=-0.1,
                lambda_risk=1.0, clamp_values=False,
            )

    def test_zero_sigma_accepted(self) -> None:
        """sigma=0 is valid (zero volatility)."""
        result: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1100.0, w_next=0.5, sigma=0.0,
            lambda_risk=1.0, clamp_values=False,
        )
        assert result.risk_penalty == pytest.approx(0.0, abs=_ABS)


# ── Always-raise validation ────────────────────────────────────────────────

class TestAlwaysRaisesValidation:
    """These errors are raised regardless of clamp_values."""

    def test_negative_lambda_risk_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_risk must be >= 0"):
            compute_reward(
                V=1000.0, V_next=1100.0, w_next=0.5, sigma=0.1, lambda_risk=-1.0
            )

    def test_negative_lambda_risk_raises_with_clamp_false(self) -> None:
        with pytest.raises(ValueError, match="lambda_risk must be >= 0"):
            compute_reward(
                V=1000.0, V_next=1100.0, w_next=0.5, sigma=0.1,
                lambda_risk=-0.001, clamp_values=False,
            )

    def test_w_next_above_upper_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="w_next out of bounds"):
            compute_reward(
                V=1000.0, V_next=1100.0, w_next=1.5, sigma=0.1, lambda_risk=1.0
            )

    def test_w_next_below_lower_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="w_next out of bounds"):
            compute_reward(
                V=1000.0, V_next=1100.0, w_next=-0.1, sigma=0.1, lambda_risk=1.0
            )

    def test_boundary_w_next_zero_accepted(self) -> None:
        result: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1100.0, w_next=0.0, sigma=0.1, lambda_risk=1.0
        )
        assert result.risk_penalty == pytest.approx(0.0, abs=_ABS)

    def test_boundary_w_next_one_accepted(self) -> None:
        result: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1100.0, w_next=1.0, sigma=0.1, lambda_risk=1.0
        )
        assert result.risk_penalty == pytest.approx(1.0 * 1.0 * (0.1 ** 2), rel=_REL)


# ── Weight bounds bypass ───────────────────────────────────────────────────

class TestWeightBoundsRewardFn:
    def test_weight_bounds_none_bypasses_w_next_check(self) -> None:
        result: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1100.0, w_next=2.0, sigma=0.1,
            lambda_risk=1.0, weight_bounds=None,
        )
        # risk_penalty = 1.0 * 2.0² * 0.1² = 0.04
        assert result.risk_penalty == pytest.approx(1.0 * 4.0 * 0.01, rel=_REL, abs=_ABS)

    def test_custom_weight_bounds_accepted(self) -> None:
        result: RewardBreakdown = compute_reward(
            V=1000.0, V_next=1100.0, w_next=1.5, sigma=0.1,
            lambda_risk=1.0, weight_bounds=(0.0, 2.0),
        )
        assert isinstance(result.total, float)

    def test_w_next_violates_custom_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="w_next out of bounds"):
            compute_reward(
                V=1000.0, V_next=1100.0, w_next=3.0, sigma=0.1,
                lambda_risk=1.0, weight_bounds=(0.0, 2.0),
            )


# ── compute_reward_from_values wrapper ────────────────────────────────────

class TestComputeRewardFromValues:
    def test_returns_float(self) -> None:
        result = compute_reward_from_values(
            V=1000.0, V_next=1100.0, w_next=0.5, sigma=0.1, lambda_risk=1.0
        )
        assert isinstance(result, float)

    def test_value_equals_breakdown_total(self) -> None:
        V: float = 1000.0
        V_next: float = 1050.0
        w_next: float = 0.6
        sigma: float = 0.15
        lam: float = 0.8

        breakdown: RewardBreakdown = compute_reward(
            V=V, V_next=V_next, w_next=w_next, sigma=sigma, lambda_risk=lam
        )
        scalar: float = compute_reward_from_values(
            V=V, V_next=V_next, w_next=w_next, sigma=sigma, lambda_risk=lam
        )
        assert scalar == pytest.approx(breakdown.total, rel=1e-12, abs=1e-15)

    def test_kwargs_forwarded_sigma_is_std_false(self) -> None:
        """sigma_is_std=False is forwarded via **kwargs and changes the result."""
        sigma: float = 0.04
        lam: float = 2.0
        w_next: float = 0.5

        # sigma_is_std=False: risk_penalty = 2*0.25*0.04 = 0.02
        expected_total: float = -(lam * (w_next ** 2) * sigma)

        scalar: float = compute_reward_from_values(
            V=1000.0, V_next=1000.0, w_next=w_next, sigma=sigma,
            lambda_risk=lam, sigma_is_std=False,
        )
        assert scalar == pytest.approx(expected_total, rel=_REL, abs=_ABS)

    def test_kwargs_forwarded_clamp_values_false_raises(self) -> None:
        """clamp_values=False is forwarded; invalid V raises ValueError."""
        with pytest.raises(ValueError, match="V must be > 0"):
            compute_reward_from_values(
                V=0.0, V_next=1000.0, w_next=0.5, sigma=0.1,
                lambda_risk=1.0, clamp_values=False,
            )

    def test_consistent_with_compute_reward_sigma_is_std_false(self) -> None:
        """compute_reward_from_values ≡ compute_reward(...).total for non-default kwargs."""
        kwargs = dict(
            V=800.0, V_next=900.0, w_next=0.4, sigma=0.09,
            lambda_risk=1.5, sigma_is_std=False,
        )
        breakdown: RewardBreakdown = compute_reward(**kwargs)
        scalar: float = compute_reward_from_values(**kwargs)
        assert scalar == pytest.approx(breakdown.total, rel=1e-12, abs=1e-15)
