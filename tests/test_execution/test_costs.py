"""Tests for transaction cost models."""

import math
import pytest

from volatility_arbitrage.execution.costs import (
    SquareRootImpactModel,
    TransactionCostModel,
)


@pytest.mark.unit
class TestSquareRootImpactModel:
    """Tests for SquareRootImpactModel."""

    def test_inheritance(self) -> None:
        """Model inherits from TransactionCostModel."""
        model = SquareRootImpactModel()
        assert isinstance(model, TransactionCostModel)

    def test_zero_order_returns_zero(self) -> None:
        """Zero order size should return zero cost."""
        model = SquareRootImpactModel()
        cost = model.calculate_cost(
            order_size=0,
            price=100.0,
            volatility=0.20,
            daily_volume=1_000_000,
        )
        assert cost == 0.0

    def test_spread_cost_only_with_zero_volume(self) -> None:
        """When volume is zero, only spread cost applies."""
        model = SquareRootImpactModel(half_spread_bps=10.0)
        cost = model.calculate_cost(
            order_size=1000,
            price=50.0,
            volatility=0.20,
            daily_volume=0,
        )
        # Spread cost: 1000 * 50 * 10/10000 = 50
        expected = 1000 * 50.0 * (10.0 / 10_000)
        assert cost == pytest.approx(expected)

    def test_cost_increases_with_order_size(self) -> None:
        """Larger orders should have higher costs."""
        model = SquareRootImpactModel()

        cost_small = model.calculate_cost(
            order_size=1000,
            price=100.0,
            volatility=0.20,
            daily_volume=1_000_000,
        )

        cost_large = model.calculate_cost(
            order_size=10_000,
            price=100.0,
            volatility=0.20,
            daily_volume=1_000_000,
        )

        assert cost_large > cost_small

    def test_cost_scales_sublinearly_with_size(self) -> None:
        """Cost should scale with sqrt of order size (sublinear)."""
        model = SquareRootImpactModel(half_spread_bps=0.0)  # Isolate impact

        # Double the order size
        cost_base = model.calculate_cost(
            order_size=10_000,
            price=100.0,
            volatility=0.20,
            daily_volume=1_000_000,
        )

        cost_double = model.calculate_cost(
            order_size=20_000,
            price=100.0,
            volatility=0.20,
            daily_volume=1_000_000,
        )

        # If cost scaled linearly, ratio would be 2.0
        # With sqrt scaling, ratio should be sqrt(2) * 2 ≈ 2.83 for value
        # But we have order_value * sqrt(participation), so:
        # cost_double / cost_base = (2 * value) * sqrt(2 * participation) / (value * sqrt(participation))
        #                        = 2 * sqrt(2) ≈ 2.83
        ratio = cost_double / cost_base
        expected_ratio = 2.0 * math.sqrt(2.0)
        assert ratio == pytest.approx(expected_ratio, rel=0.01)

    def test_cost_increases_with_volatility(self) -> None:
        """Higher volatility should increase impact cost."""
        model = SquareRootImpactModel(half_spread_bps=0.0)  # Isolate impact

        cost_low_vol = model.calculate_cost(
            order_size=10_000,
            price=100.0,
            volatility=0.10,
            daily_volume=1_000_000,
        )

        cost_high_vol = model.calculate_cost(
            order_size=10_000,
            price=100.0,
            volatility=0.30,
            daily_volume=1_000_000,
        )

        assert cost_high_vol > cost_low_vol
        # Should scale linearly with vol
        assert cost_high_vol / cost_low_vol == pytest.approx(3.0, rel=0.01)

    def test_absolute_value_of_order_size(self) -> None:
        """Negative order size should use absolute value."""
        model = SquareRootImpactModel()

        cost_positive = model.calculate_cost(
            order_size=1000,
            price=100.0,
            volatility=0.20,
            daily_volume=1_000_000,
        )

        cost_negative = model.calculate_cost(
            order_size=-1000,
            price=100.0,
            volatility=0.20,
            daily_volume=1_000_000,
        )

        assert cost_positive == cost_negative

    def test_estimate_impact_bps(self) -> None:
        """Test impact estimation in basis points."""
        model = SquareRootImpactModel(impact_coeff=0.1)

        # 1% of ADV at 20% vol
        impact_bps = model.estimate_impact_bps(
            order_size=10_000,
            volatility=0.20,
            daily_volume=1_000_000,
        )

        # Expected: 0.1 * 0.20 * sqrt(0.01) * 10000 = 20 bps
        expected = 0.1 * 0.20 * math.sqrt(0.01) * 10_000
        assert impact_bps == pytest.approx(expected, rel=0.01)

    def test_estimate_impact_zero_volume(self) -> None:
        """Impact estimation with zero volume returns zero."""
        model = SquareRootImpactModel()
        impact = model.estimate_impact_bps(
            order_size=1000,
            volatility=0.20,
            daily_volume=0,
        )
        assert impact == 0.0

    def test_default_parameters_reasonable(self) -> None:
        """Default parameters should give reasonable costs."""
        model = SquareRootImpactModel()

        # Trade 1% of ADV at 20% vol, $100 stock
        cost = model.calculate_cost(
            order_size=10_000,
            price=100.0,
            volatility=0.20,
            daily_volume=1_000_000,
        )

        order_value = 10_000 * 100.0
        cost_bps = (cost / order_value) * 10_000

        # Should be in reasonable range (5-25 bps for this size)
        assert 5 < cost_bps < 30
