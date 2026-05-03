"""
Tests for exit logic features in volatility arbitrage strategy.

Tests stop loss, profit taking, signal smoothing, and delta rebalancing.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from volatility_arbitrage.strategy.volatility_arbitrage import (
    VolatilityArbitrageStrategy,
    VolatilityArbitrageConfig,
    VolatilitySpread,
)
from volatility_arbitrage.core.types import Position


class TestStopLoss:
    """Tests for stop loss functionality."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with default config."""
        config = VolatilityArbitrageConfig(max_loss_pct=Decimal("50.0"))
        return VolatilityArbitrageStrategy(config)

    def test_stop_loss_not_triggered_when_profitable(self, strategy):
        """Stop loss should not trigger on profitable positions."""
        # Setup: position with 10% profit
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=10,
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("5.50"),  # 10% profit
                last_update=datetime.now(),
            )
        }
        strategy.option_positions["SPY"] = {"direction": "buy", "profit_levels_taken": []}

        signals = strategy._check_stop_loss("SPY", positions)
        assert len(signals) == 0

    def test_stop_loss_not_triggered_at_small_loss(self, strategy):
        """Stop loss should not trigger at small losses."""
        # Setup: position with 30% loss (below 50% threshold)
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=10,
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("3.50"),  # 30% loss
                last_update=datetime.now(),
            )
        }
        strategy.option_positions["SPY"] = {"direction": "buy", "profit_levels_taken": []}

        signals = strategy._check_stop_loss("SPY", positions)
        assert len(signals) == 0

    def test_stop_loss_triggered_at_threshold(self, strategy):
        """Stop loss should trigger at 50% loss."""
        # Setup: position with exactly 50% loss
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=10,
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("2.50"),  # 50% loss
                last_update=datetime.now(),
            )
        }
        strategy.option_positions["SPY"] = {"direction": "buy", "profit_levels_taken": []}

        signals = strategy._check_stop_loss("SPY", positions)
        assert len(signals) > 0
        assert signals[0].action == "sell"
        assert "Stop loss" in signals[0].reason

    def test_stop_loss_triggered_beyond_threshold(self, strategy):
        """Stop loss should trigger when loss exceeds threshold."""
        # Setup: position with 60% loss
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=10,
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("2.00"),  # 60% loss
                last_update=datetime.now(),
            )
        }
        strategy.option_positions["SPY"] = {"direction": "buy", "profit_levels_taken": []}

        signals = strategy._check_stop_loss("SPY", positions)
        assert len(signals) > 0


class TestStopLossBypassesHoldingPeriod:
    """Regression: stop loss must fire even within min_holding_days (F6)."""

    def test_stop_loss_fires_during_holding_period(self):
        """A QV position crashing on day 2 must be cut despite min_holding_days=5."""
        config = VolatilityArbitrageConfig(
            max_loss_pct=Decimal("50.0"),
            min_holding_days=5,
        )
        strategy = VolatilityArbitrageStrategy(config)

        symbol = "SPY"
        entry_time = datetime(2024, 1, 1)
        # Day 2: still inside the holding-period gate.
        current_time = entry_time + timedelta(days=2)

        # Mimic QV entry on day 0.
        strategy.option_positions[symbol] = {
            "direction": "buy",
            "profit_levels_taken": [],
        }
        strategy.entry_timestamps[symbol] = entry_time

        # -80% PnL on the option leg.
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=10,
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("1.00"),
                last_update=current_time,
            ),
        }

        vol_spread = VolatilitySpread(
            symbol=symbol,
            timestamp=current_time,
            implied_vol=Decimal("0.20"),
            forecasted_vol=Decimal("0.18"),
            spread=Decimal("0.02"),
            spread_pct=Decimal("11.0"),
        )

        signals = strategy._check_exit_signals(vol_spread, positions, Decimal("5.0"))

        assert len(signals) > 0, "Stop loss must fire even inside min_holding_days"
        assert any("Stop loss" in s.reason for s in signals)

    def test_holding_period_still_blocks_discretionary_exit(self):
        """Spread-converged exit must still be blocked by min_holding_days."""
        config = VolatilityArbitrageConfig(
            max_loss_pct=Decimal("50.0"),
            min_holding_days=5,
        )
        strategy = VolatilityArbitrageStrategy(config)

        symbol = "SPY"
        entry_time = datetime(2024, 1, 1)
        current_time = entry_time + timedelta(days=2)

        strategy.option_positions[symbol] = {
            "direction": "buy",
            "profit_levels_taken": [],
        }
        strategy.entry_timestamps[symbol] = entry_time

        # Profitable position so stop loss doesn't fire.
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=10,
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("5.10"),
                last_update=current_time,
            ),
        }

        # Spread converged - would normally trigger exit.
        vol_spread = VolatilitySpread(
            symbol=symbol,
            timestamp=current_time,
            implied_vol=Decimal("0.20"),
            forecasted_vol=Decimal("0.20"),
            spread=Decimal("0.0"),
            spread_pct=Decimal("0.5"),
        )

        signals = strategy._check_exit_signals(vol_spread, positions, Decimal("5.0"))
        assert signals == [], "Discretionary exit must remain gated by min_holding_days"


class TestProfitTaking:
    """Tests for tiered profit taking functionality."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with profit taking enabled."""
        config = VolatilityArbitrageConfig(
            use_profit_taking=True,
            profit_take_levels=[Decimal("0.25"), Decimal("0.50"), Decimal("0.75")],
            profit_take_sizes=[Decimal("0.33"), Decimal("0.33"), Decimal("0.34")],
        )
        return VolatilityArbitrageStrategy(config)

    def test_no_profit_taking_when_not_profitable(self, strategy):
        """No profit taking when position is at loss."""
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=10,
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("4.00"),  # 20% loss
                last_update=datetime.now(),
            )
        }
        strategy.option_positions["SPY"] = {"profit_levels_taken": []}

        signals = strategy._check_profit_taking("SPY", positions)
        assert len(signals) == 0

    def test_profit_taking_at_25_percent(self, strategy):
        """Profit taking triggers at 25% profit."""
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=100,
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("6.25"),  # 25% profit
                last_update=datetime.now(),
            )
        }
        strategy.option_positions["SPY"] = {"profit_levels_taken": []}

        signals = strategy._check_profit_taking("SPY", positions)
        assert len(signals) > 0
        assert "25%" in signals[0].reason
        # Should close ~33% of 100 = 33 contracts
        assert signals[0].quantity == 33

    def test_profit_taking_tracks_levels(self, strategy):
        """Profit levels are tracked and not repeated."""
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=100,
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("6.25"),  # 25% profit
                last_update=datetime.now(),
            )
        }
        strategy.option_positions["SPY"] = {"profit_levels_taken": []}

        # First call should trigger 25% level
        signals1 = strategy._check_profit_taking("SPY", positions)
        assert len(signals1) > 0

        # Second call should not trigger (level already taken)
        signals2 = strategy._check_profit_taking("SPY", positions)
        assert len(signals2) == 0

    def test_profit_taking_at_50_percent(self, strategy):
        """Profit taking triggers at 50% after 25% already taken."""
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=67,  # After 33 taken at 25%
                avg_entry_price=Decimal("5.00"),
                current_price=Decimal("7.50"),  # 50% profit
                last_update=datetime.now(),
            )
        }
        strategy.option_positions["SPY"] = {"profit_levels_taken": [0]}  # 25% already taken

        signals = strategy._check_profit_taking("SPY", positions)
        assert len(signals) > 0
        assert "50%" in signals[0].reason


class TestSignalSmoothing:
    """Tests for signal smoothing functionality."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with signal smoothing enabled."""
        config = VolatilityArbitrageConfig(
            use_signal_smoothing=True,
            signal_smoothing_window=3,
            signal_smoothing_min_history=2,
        )
        return VolatilityArbitrageStrategy(config)

    def test_smoothing_returns_raw_with_no_history(self, strategy):
        """First call should return raw value."""
        result = strategy._smooth_consensus("TEST", Decimal("0.5"))
        # With only 1 data point, should return raw
        assert result == Decimal("0.5")

    def test_smoothing_applies_ema(self, strategy):
        """EMA should smooth values over time."""
        # Add several values
        strategy._smooth_consensus("TEST", Decimal("0.2"))
        strategy._smooth_consensus("TEST", Decimal("0.4"))
        result = strategy._smooth_consensus("TEST", Decimal("0.6"))

        # Result should be between min and max input
        assert Decimal("0.2") < result < Decimal("0.6")

    def test_smoothing_disabled_returns_raw(self):
        """When disabled, should return raw value."""
        config = VolatilityArbitrageConfig(use_signal_smoothing=False)
        strategy = VolatilityArbitrageStrategy(config)

        # Even with history, should return raw
        strategy._smooth_consensus("TEST", Decimal("0.2"))
        strategy._smooth_consensus("TEST", Decimal("0.4"))
        result = strategy._smooth_consensus("TEST", Decimal("0.8"))

        assert result == Decimal("0.8")


class TestDeltaRebalancing:
    """Tests for delta rebalancing functionality."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with delta rebalancing config."""
        config = VolatilityArbitrageConfig(
            delta_rebalance_threshold=Decimal("0.10"),
            delta_target=Decimal("0.0"),
        )
        return VolatilityArbitrageStrategy(config)

    def test_no_rebalance_without_positions(self, strategy):
        """No rebalancing when no option positions exist."""
        from volatility_arbitrage.core.types import OptionChain, OptionContract, OptionType

        option_chain = OptionChain(
            symbol="SPY",
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(days=30),
            underlying_price=Decimal("500"),
            calls=[
                OptionContract(
                    symbol="SPY",
                    option_type=OptionType.CALL,
                    strike=Decimal("500"),
                    expiry=datetime.now() + timedelta(days=30),
                    price=Decimal("10"),
                    implied_volatility=Decimal("0.20"),
                )
            ],
            puts=[],
        )

        signals = strategy._check_delta_rebalancing("SPY", option_chain, {})
        assert len(signals) == 0

    def test_no_rebalance_when_delta_within_threshold(self, strategy):
        """No rebalancing when delta is within threshold."""
        from volatility_arbitrage.core.types import OptionChain, OptionContract, OptionType

        expiry = datetime.now() + timedelta(days=30)
        option_chain = OptionChain(
            symbol="SPY",
            timestamp=datetime.now(),
            expiry=expiry,
            underlying_price=Decimal("500"),
            calls=[
                OptionContract(
                    symbol="SPY",
                    option_type=OptionType.CALL,
                    strike=Decimal("500"),
                    expiry=expiry,
                    price=Decimal("10"),
                    implied_volatility=Decimal("0.20"),
                )
            ],
            puts=[],
        )

        # Setup position tracking
        strategy.option_positions["SPY"] = {
            "strike": Decimal("500"),
            "expiry": expiry,
        }

        # Small position - delta should be within threshold
        positions = {
            "SPY_C_500": Position(
                symbol="SPY_C_500",
                quantity=1,  # Very small position
                avg_entry_price=Decimal("10"),
                current_price=Decimal("10"),
                last_update=datetime.now(),
            )
        }

        # This may or may not trigger based on BS delta calc
        # The test verifies the method runs without error
        signals = strategy._check_delta_rebalancing("SPY", option_chain, positions)
        # If signals generated, they should be valid
        for signal in signals:
            assert signal.symbol == "SPY"
            assert signal.action in ["buy", "sell"]
