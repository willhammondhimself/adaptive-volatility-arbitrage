"""Tests for DeltaHedger class."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from volatility_arbitrage.backtest.delta_hedged.hedger import DeltaHedger
from volatility_arbitrage.backtest.delta_hedged.types import (
    HedgeConfig,
    RebalanceFrequency,
)
from volatility_arbitrage.execution.costs import SquareRootImpactModel
from volatility_arbitrage.models.black_scholes import OptionType


@pytest.mark.unit
class TestDeltaHedgerInitialization:
    """Tests for hedger initialization."""

    def test_initialize_creates_delta_neutral_position(self) -> None:
        """Initialize creates a delta-neutral portfolio."""
        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.DAILY)
        hedger = DeltaHedger(config)

        state = hedger.initialize(
            timestamp=datetime(2024, 1, 2, 9, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
            option_position=Decimal("10"),  # 10 contracts
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )

        # Portfolio delta should be 0 after initialization
        assert state.portfolio_delta == Decimal("0")
        # Hedge shares should be negative for long call
        assert state.hedge_shares < 0

    def test_initialize_sets_option_parameters(self) -> None:
        """Initialize stores option parameters."""
        config = HedgeConfig()
        hedger = DeltaHedger(config)

        hedger.initialize(
            timestamp=datetime(2024, 1, 2, 9, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
            option_position=Decimal("10"),
            strike=Decimal("460.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.PUT,
        )

        assert hedger._strike == Decimal("460.00")
        assert hedger._option_type == OptionType.PUT

    def test_initialize_calculates_greeks(self) -> None:
        """Initialize calculates portfolio Greeks."""
        config = HedgeConfig()
        hedger = DeltaHedger(config)

        state = hedger.initialize(
            timestamp=datetime(2024, 1, 2, 9, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
            option_position=Decimal("10"),
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )

        # Long call has positive gamma, vega; negative theta
        assert state.portfolio_gamma > 0
        assert state.portfolio_vega > 0
        assert state.portfolio_theta < 0


@pytest.mark.unit
class TestDeltaHedgerUpdate:
    """Tests for hedger update logic."""

    @pytest.fixture
    def initialized_hedger(self) -> DeltaHedger:
        """Create an initialized hedger."""
        config = HedgeConfig(
            rebalance_frequency=RebalanceFrequency.DAILY,
            delta_threshold=Decimal("0.05"),
        )
        hedger = DeltaHedger(config)
        hedger.initialize(
            timestamp=datetime(2024, 1, 2, 9, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
            option_position=Decimal("10"),
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )
        return hedger

    def test_update_without_rebalance_high_threshold(self) -> None:
        """With high threshold, small moves don't trigger threshold rebalance."""
        config = HedgeConfig(
            rebalance_frequency=RebalanceFrequency.DAILY,
            delta_threshold=Decimal("100.0"),  # Very high threshold
        )
        hedger = DeltaHedger(config)
        hedger.initialize(
            timestamp=datetime(2024, 1, 2, 9, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
            option_position=Decimal("10"),
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )

        # Small spot move, same day, high threshold
        state, attr, cost = hedger.update(
            timestamp=datetime(2024, 1, 2, 10, 30),
            spot=Decimal("450.50"),
            iv=Decimal("0.20"),
        )

        # Should not rebalance (same day, threshold too high)
        assert not attr.rebalanced
        assert cost == Decimal("0")

    def test_update_triggers_time_rebalance(
        self, initialized_hedger: DeltaHedger
    ) -> None:
        """Daily rebalance triggers after 24 hours."""
        # Next day
        state, attr, cost = initialized_hedger.update(
            timestamp=datetime(2024, 1, 3, 9, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
        )

        assert attr.rebalanced
        assert initialized_hedger.rebalance_count == 1

    def test_update_triggers_threshold_rebalance(
        self, initialized_hedger: DeltaHedger
    ) -> None:
        """Large spot move triggers rebalance."""
        # Large spot move that creates delta drift
        state, attr, cost = initialized_hedger.update(
            timestamp=datetime(2024, 1, 2, 10, 30),
            spot=Decimal("460.00"),  # +10 from 450
            iv=Decimal("0.20"),
        )

        # Delta drift should trigger rebalance
        assert attr.rebalanced or abs(state.portfolio_delta) <= Decimal("0.05")

    def test_update_calculates_pnl_attribution(
        self, initialized_hedger: DeltaHedger
    ) -> None:
        """Update calculates P&L attribution."""
        state, attr, cost = initialized_hedger.update(
            timestamp=datetime(2024, 1, 3, 9, 30),
            spot=Decimal("455.00"),
            iv=Decimal("0.22"),
        )

        # Check attribution is calculated
        assert attr.timestamp == datetime(2024, 1, 3, 9, 30)
        # Gamma P&L should be positive for spot move
        assert attr.gamma_pnl >= 0
        # Vega P&L should be positive for IV increase (long vega)
        assert attr.vega_pnl > 0
        # Theta should be negative (long options)
        assert attr.theta_pnl < 0

    def test_update_records_transaction_costs(
        self, initialized_hedger: DeltaHedger
    ) -> None:
        """Rebalancing incurs transaction costs."""
        # Force a rebalance
        state, attr, cost = initialized_hedger.update(
            timestamp=datetime(2024, 1, 3, 9, 30),
            spot=Decimal("455.00"),
            iv=Decimal("0.20"),
        )

        if attr.rebalanced:
            assert cost > 0
            assert attr.transaction_costs > 0


@pytest.mark.unit
class TestDeltaHedgerRebalanceFrequency:
    """Tests for different rebalance frequencies."""

    def test_continuous_rebalance(self) -> None:
        """Continuous mode rebalances every tick."""
        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.CONTINUOUS)
        hedger = DeltaHedger(config)

        hedger.initialize(
            timestamp=datetime(2024, 1, 2, 9, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
            option_position=Decimal("10"),
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )

        # Every update should trigger rebalance
        for i in range(3):
            state, attr, cost = hedger.update(
                timestamp=datetime(2024, 1, 2, 9, 31 + i),
                spot=Decimal("450.00") + Decimal(str(i)),
                iv=Decimal("0.20"),
            )
            assert attr.rebalanced

        assert hedger.rebalance_count == 3

    def test_hourly_rebalance(self) -> None:
        """Hourly mode rebalances after 1 hour (with high threshold to isolate time)."""
        config = HedgeConfig(
            rebalance_frequency=RebalanceFrequency.HOURLY,
            delta_threshold=Decimal("100.0"),  # High to isolate time-based rebalance
        )
        hedger = DeltaHedger(config)

        hedger.initialize(
            timestamp=datetime(2024, 1, 2, 9, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
            option_position=Decimal("10"),
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )

        # 30 minutes - no rebalance (time not elapsed, threshold high)
        state, attr, cost = hedger.update(
            timestamp=datetime(2024, 1, 2, 10, 0),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
        )
        assert not attr.rebalanced

        # 1 hour - should rebalance (time elapsed)
        state, attr, cost = hedger.update(
            timestamp=datetime(2024, 1, 2, 10, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
        )
        assert attr.rebalanced


@pytest.mark.unit
class TestDeltaHedgerReset:
    """Tests for hedger reset."""

    def test_reset_clears_state(self) -> None:
        """Reset clears all state."""
        config = HedgeConfig()
        hedger = DeltaHedger(config)

        hedger.initialize(
            timestamp=datetime(2024, 1, 2, 9, 30),
            spot=Decimal("450.00"),
            iv=Decimal("0.20"),
            option_position=Decimal("10"),
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )

        hedger.update(
            timestamp=datetime(2024, 1, 3, 9, 30),
            spot=Decimal("455.00"),
            iv=Decimal("0.21"),
        )

        hedger.reset()

        assert hedger.state is None
        assert len(hedger.attribution_history) == 0
        assert hedger.rebalance_count == 0
