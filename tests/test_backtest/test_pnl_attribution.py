"""Tests for P&L attribution functions."""

from decimal import Decimal

import pytest

from volatility_arbitrage.backtest.delta_hedged.attribution import (
    calculate_delta_pnl,
    calculate_gamma_pnl,
    calculate_theta_pnl,
    calculate_vega_pnl,
)


@pytest.mark.unit
class TestDeltaPnL:
    """Tests for delta P&L calculation."""

    def test_delta_pnl_positive_for_long_delta_up_move(self) -> None:
        """Long delta profits from up move."""
        delta = Decimal("0.50")
        spot_change = Decimal("5.00")
        pnl = calculate_delta_pnl(delta, spot_change)
        assert pnl == Decimal("2.50")

    def test_delta_pnl_negative_for_long_delta_down_move(self) -> None:
        """Long delta loses from down move."""
        delta = Decimal("0.50")
        spot_change = Decimal("-5.00")
        pnl = calculate_delta_pnl(delta, spot_change)
        assert pnl == Decimal("-2.50")

    def test_delta_pnl_zero_when_hedged(self) -> None:
        """Delta P&L should be ~0 when portfolio delta is 0."""
        delta = Decimal("0")
        spot_change = Decimal("10.00")
        pnl = calculate_delta_pnl(delta, spot_change)
        assert pnl == Decimal("0")

    def test_delta_pnl_short_delta_profits_from_down_move(self) -> None:
        """Short delta profits from down move."""
        delta = Decimal("-0.50")
        spot_change = Decimal("-5.00")
        pnl = calculate_delta_pnl(delta, spot_change)
        assert pnl == Decimal("2.50")


@pytest.mark.unit
class TestGammaPnL:
    """Tests for gamma P&L calculation."""

    def test_gamma_pnl_positive_for_long_gamma_up_move(self) -> None:
        """Long gamma profits from up moves."""
        gamma = Decimal("0.05")
        spot = Decimal("100")
        spot_return = Decimal("0.05")  # 5% up
        pnl = calculate_gamma_pnl(gamma, spot, spot_return)
        assert pnl > 0

    def test_gamma_pnl_positive_for_long_gamma_down_move(self) -> None:
        """Long gamma profits from down moves (convexity)."""
        gamma = Decimal("0.05")
        spot = Decimal("100")
        spot_return = Decimal("-0.05")  # 5% down
        pnl = calculate_gamma_pnl(gamma, spot, spot_return)
        assert pnl > 0

    def test_gamma_pnl_zero_for_no_move(self) -> None:
        """No gamma P&L if spot doesn't move."""
        gamma = Decimal("0.05")
        spot = Decimal("100")
        spot_return = Decimal("0")
        pnl = calculate_gamma_pnl(gamma, spot, spot_return)
        assert pnl == Decimal("0")

    def test_gamma_pnl_formula_correct(self) -> None:
        """Verify gamma P&L formula: 0.5 * gamma * S^2 * r^2."""
        gamma = Decimal("0.02")
        spot = Decimal("100")
        spot_return = Decimal("0.10")  # 10% move
        expected = Decimal("0.5") * gamma * (spot**2) * (spot_return**2)
        pnl = calculate_gamma_pnl(gamma, spot, spot_return)
        assert pnl == expected

    def test_gamma_pnl_scales_quadratically(self) -> None:
        """Gamma P&L scales with square of return."""
        gamma = Decimal("0.05")
        spot = Decimal("100")
        pnl_5pct = calculate_gamma_pnl(gamma, spot, Decimal("0.05"))
        pnl_10pct = calculate_gamma_pnl(gamma, spot, Decimal("0.10"))
        # 10% move should give 4x the P&L of 5% move
        assert float(pnl_10pct / pnl_5pct) == pytest.approx(4.0)


@pytest.mark.unit
class TestVegaPnL:
    """Tests for vega P&L calculation."""

    def test_vega_pnl_positive_for_long_vega_iv_increase(self) -> None:
        """Long vega profits from IV increase."""
        vega = Decimal("10.0")  # $10 per 1% vol change
        iv_change = Decimal("0.02")  # IV up 2 percentage points
        pnl = calculate_vega_pnl(vega, iv_change)
        assert pnl == Decimal("20.0")  # 2% * $10

    def test_vega_pnl_negative_for_long_vega_iv_decrease(self) -> None:
        """Long vega loses from IV decrease."""
        vega = Decimal("10.0")
        iv_change = Decimal("-0.02")  # IV down 2 percentage points
        pnl = calculate_vega_pnl(vega, iv_change)
        assert pnl == Decimal("-20.0")

    def test_vega_pnl_zero_for_no_iv_change(self) -> None:
        """No vega P&L if IV doesn't change."""
        vega = Decimal("10.0")
        iv_change = Decimal("0")
        pnl = calculate_vega_pnl(vega, iv_change)
        assert pnl == Decimal("0")


@pytest.mark.unit
class TestThetaPnL:
    """Tests for theta P&L calculation."""

    def test_theta_pnl_negative_for_long_options(self) -> None:
        """Long options have negative theta (time decay)."""
        theta = Decimal("-0.05")  # -$0.05 per day
        dt_days = Decimal("1.0")
        pnl = calculate_theta_pnl(theta, dt_days)
        assert pnl == Decimal("-0.05")

    def test_theta_pnl_accumulates_over_time(self) -> None:
        """Theta decay accumulates over multiple days."""
        theta = Decimal("-0.10")  # -$0.10 per day
        dt_days = Decimal("5.0")  # 5 days
        pnl = calculate_theta_pnl(theta, dt_days)
        assert pnl == Decimal("-0.50")

    def test_theta_pnl_positive_for_short_options(self) -> None:
        """Short options have positive theta."""
        theta = Decimal("0.05")  # +$0.05 per day
        dt_days = Decimal("1.0")
        pnl = calculate_theta_pnl(theta, dt_days)
        assert pnl == Decimal("0.05")
