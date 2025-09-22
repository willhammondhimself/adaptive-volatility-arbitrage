"""
Tests for multi-asset backtest engine.

Focused on core functionality: position tracking, Greeks calculation, and P&L accuracy.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from volatility_arbitrage.backtest import (
    MultiAssetBacktestEngine,
    MultiAssetPosition,
    PortfolioGreeks,
)
from volatility_arbitrage.core.config import BacktestConfig
from volatility_arbitrage.core.types import OptionType
from volatility_arbitrage.strategy.base import BuyAndHoldStrategy


@pytest.mark.unit
class TestMultiAssetPosition:
    """Tests for MultiAssetPosition model."""

    def test_stock_position_creation(self):
        """Test creating a stock position."""
        pos = MultiAssetPosition(
            symbol="SPY",
            asset_type="stock",
            quantity=100,
            entry_price=Decimal("450"),
            current_price=Decimal("455"),
            last_update=datetime.now(),
        )

        assert pos.symbol == "SPY"
        assert pos.asset_type == "stock"
        assert not pos.is_option
        assert pos.quantity == 100

    def test_option_position_creation(self):
        """Test creating an option position."""
        pos = MultiAssetPosition(
            symbol="SPY_CALL_450_20240315",
            asset_type="option",
            quantity=10,
            entry_price=Decimal("10.50"),
            current_price=Decimal("12.00"),
            last_update=datetime.now(),
            option_type=OptionType.CALL,
            strike=Decimal("450"),
            expiry=datetime(2024, 3, 15),
            underlying_price=Decimal("455"),
            implied_volatility=Decimal("0.20"),
            risk_free_rate=Decimal("0.05"),
        )

        assert pos.is_option
        assert pos.option_type == OptionType.CALL
        assert pos.strike == Decimal("450")

    def test_stock_market_value(self):
        """Test market value calculation for stock."""
        pos = MultiAssetPosition(
            symbol="SPY",
            asset_type="stock",
            quantity=100,
            entry_price=Decimal("450"),
            current_price=Decimal("455"),
            last_update=datetime.now(),
        )

        # 100 shares * $455 = $45,500
        assert pos.market_value == Decimal("45500")

    def test_option_market_value(self):
        """Test market value calculation for option."""
        pos = MultiAssetPosition(
            symbol="SPY_CALL_450_20240315",
            asset_type="option",
            quantity=10,
            entry_price=Decimal("10.50"),
            current_price=Decimal("12.00"),
            last_update=datetime.now(),
            option_type=OptionType.CALL,
            strike=Decimal("450"),
            expiry=datetime(2024, 3, 15),
        )

        # 10 contracts * $12.00 * 100 multiplier = $12,000
        assert pos.market_value == Decimal("12000")

    def test_unrealized_pnl_long(self):
        """Test unrealized P&L for long position."""
        pos = MultiAssetPosition(
            symbol="SPY",
            asset_type="stock",
            quantity=100,
            entry_price=Decimal("450"),
            current_price=Decimal("455"),
            last_update=datetime.now(),
        )

        # ($455 - $450) * 100 = $500 profit
        assert pos.unrealized_pnl == Decimal("500")

    def test_unrealized_pnl_short(self):
        """Test unrealized P&L for short position."""
        pos = MultiAssetPosition(
            symbol="SPY",
            asset_type="stock",
            quantity=-100,
            entry_price=Decimal("450"),
            current_price=Decimal("455"),
            last_update=datetime.now(),
        )

        # ($450 - $455) * 100 = -$500 loss
        assert pos.unrealized_pnl == Decimal("-500")

    def test_greeks_calculation_for_option(self):
        """Test Greeks calculation for option position."""
        pos = MultiAssetPosition(
            symbol="SPY_CALL_450_20240315",
            asset_type="option",
            quantity=10,
            entry_price=Decimal("10.50"),
            current_price=Decimal("12.00"),
            last_update=datetime(2024, 1, 1),
            option_type=OptionType.CALL,
            strike=Decimal("450"),
            expiry=datetime(2024, 3, 15),
            underlying_price=Decimal("455"),
            implied_volatility=Decimal("0.20"),
            risk_free_rate=Decimal("0.05"),
        )

        greeks = pos.calculate_greeks()

        assert greeks is not None
        assert greeks.delta != 0  # ATM call should have non-zero delta
        assert greeks.gamma > 0  # Gamma always positive
        assert greeks.vega > 0  # Vega always positive

    def test_greeks_none_for_stock(self):
        """Test that Greeks are None for stock positions."""
        pos = MultiAssetPosition(
            symbol="SPY",
            asset_type="stock",
            quantity=100,
            entry_price=Decimal("450"),
            current_price=Decimal("455"),
            last_update=datetime.now(),
        )

        greeks = pos.calculate_greeks()
        assert greeks is None


@pytest.mark.unit
class TestPortfolioGreeks:
    """Tests for PortfolioGreeks aggregation."""

    def test_portfolio_greeks_creation(self):
        """Test creating PortfolioGreeks."""
        greeks = PortfolioGreeks(
            delta=Decimal("0.5"),
            gamma=Decimal("0.05"),
            vega=Decimal("100"),
            theta=Decimal("-5"),
            rho=Decimal("10"),
        )

        assert greeks.delta == Decimal("0.5")
        assert greeks.vega == Decimal("100")

    def test_portfolio_greeks_to_dict(self):
        """Test converting PortfolioGreeks to dict."""
        greeks = PortfolioGreeks(
            delta=Decimal("0.5"),
            gamma=Decimal("0.05"),
            vega=Decimal("100"),
            theta=Decimal("-5"),
            rho=Decimal("10"),
        )

        d = greeks.to_dict()
        assert "delta" in d
        assert "vega" in d
        assert isinstance(d["delta"], float)


@pytest.mark.integration
class TestMultiAssetEngine:
    """Integration tests for MultiAssetBacktestEngine."""

    def test_simple_backtest(self, sample_market_data):
        """Test basic backtest with simple strategy."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            commission_rate=Decimal("0.001"),
            slippage=Decimal("0.0005"),
        )

        strategy = BuyAndHoldStrategy("SPY", 100)
        engine = MultiAssetBacktestEngine(config, strategy)

        result = engine.run(sample_market_data)

        assert result.initial_capital == Decimal("100000")
        assert result.final_capital > 0
        assert len(result.trades) > 0
        assert len(result.equity_curve) > 0

    def test_portfolio_greeks_tracking(self):
        """Test that portfolio Greeks are tracked."""
        config = BacktestConfig(initial_capital=Decimal("100000"))
        strategy = BuyAndHoldStrategy("SPY", 100)
        engine = MultiAssetBacktestEngine(config, strategy)

        # Create synthetic data
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = pd.DataFrame({
            "timestamp": dates,
            "symbol": "SPY",
            "open": 450.0,
            "high": 455.0,
            "low": 445.0,
            "close": 450.0,
            "volume": 1000000,
        })

        result = engine.run(data)

        # Check Greeks were tracked
        assert len(engine.greeks_history) > 0
        assert "portfolio_delta" in result.equity_curve.columns

    def test_option_expiration_handling(self):
        """Test that expiring options are handled."""
        config = BacktestConfig(initial_capital=Decimal("100000"))
        strategy = BuyAndHoldStrategy("SPY", 100)
        engine = MultiAssetBacktestEngine(config, strategy)

        # Add an option position that expires soon
        now = datetime.now()
        expiry = now + timedelta(days=1)

        pos = MultiAssetPosition(
            symbol="SPY_CALL_450_TEST",
            asset_type="option",
            quantity=10,
            entry_price=Decimal("10"),
            current_price=Decimal("12"),
            last_update=now,
            option_type=OptionType.CALL,
            strike=Decimal("450"),
            expiry=expiry,
            underlying_price=Decimal("455"),
            implied_volatility=Decimal("0.20"),
            risk_free_rate=Decimal("0.05"),
        )

        engine.multi_positions["SPY_CALL_450_TEST"] = pos

        # Trigger expiration handling
        engine._handle_expirations(expiry - timedelta(hours=1))

        # Position should be closed
        assert "SPY_CALL_450_TEST" not in engine.multi_positions
        assert len(engine.trades) > 0  # Closing trade recorded

    def test_pnl_accuracy(self):
        """Test P&L calculation accuracy."""
        config = BacktestConfig(
            initial_capital=Decimal("100000"),
            commission_rate=Decimal("0"),  # No commissions for clean test
            slippage=Decimal("0"),  # No slippage
        )

        strategy = BuyAndHoldStrategy("SPY", 100)
        engine = MultiAssetBacktestEngine(config, strategy)

        # Create data with known price changes
        initial_price = 100.0
        final_price = 110.0
        dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="D")

        data = pd.DataFrame({
            "timestamp": dates,
            "symbol": "SPY",
            "open": initial_price,
            "high": initial_price + 1,
            "low": initial_price - 1,
            "close": [initial_price, 102, 105, 108, final_price],
            "volume": 1000000,
        })

        result = engine.run(data)

        # Expected: Buy 100 shares at ~$100, sell at ~$110
        # Profit should be approximately $1000 (100 shares * $10 gain)
        profit = result.final_capital - result.initial_capital

        # Allow for small execution differences
        assert abs(profit - Decimal("1000")) < Decimal("200")
