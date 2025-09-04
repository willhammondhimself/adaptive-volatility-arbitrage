"""
Unit tests for core types module.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from volatility_arbitrage.core.types import (
    TickData,
    OptionContract,
    OptionChain,
    Trade,
    Position,
    OptionType,
    TradeType,
)


@pytest.mark.unit
class TestTickData:
    """Tests for TickData model."""

    def test_tick_data_creation(self, sample_tick_data):
        """Test creating valid tick data."""
        assert sample_tick_data.symbol == "SPY"
        assert sample_tick_data.price == Decimal("450.50")
        assert sample_tick_data.volume == 1000000

    def test_tick_data_immutable(self, sample_tick_data):
        """Test that tick data is immutable."""
        with pytest.raises(ValidationError):
            sample_tick_data.price = Decimal("500")

    def test_symbol_uppercase(self):
        """Test that symbol is converted to uppercase."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1),
            symbol="spy",
            price=Decimal("450"),
            volume=1000,
        )
        assert tick.symbol == "SPY"

    def test_mid_price_calculation(self):
        """Test mid price calculation."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1),
            symbol="SPY",
            price=Decimal("450"),
            volume=1000,
            bid=Decimal("449.90"),
            ask=Decimal("450.10"),
        )
        assert tick.mid_price == Decimal("450.00")

    def test_spread_calculation(self):
        """Test bid-ask spread calculation."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1),
            symbol="SPY",
            price=Decimal("450"),
            volume=1000,
            bid=Decimal("449.90"),
            ask=Decimal("450.10"),
        )
        assert tick.spread == Decimal("0.20")

    def test_invalid_price(self):
        """Test that negative price raises error."""
        with pytest.raises(ValidationError):
            TickData(
                timestamp=datetime(2024, 1, 1),
                symbol="SPY",
                price=Decimal("-450"),
                volume=1000,
            )

    def test_ask_bid_validation(self):
        """Test that ask must be >= bid."""
        with pytest.raises(ValidationError):
            TickData(
                timestamp=datetime(2024, 1, 1),
                symbol="SPY",
                price=Decimal("450"),
                volume=1000,
                bid=Decimal("450.10"),
                ask=Decimal("449.90"),
            )


@pytest.mark.unit
class TestOptionContract:
    """Tests for OptionContract model."""

    def test_option_contract_creation(self, sample_option_contract):
        """Test creating valid option contract."""
        assert sample_option_contract.symbol == "SPY"
        assert sample_option_contract.option_type == OptionType.CALL
        assert sample_option_contract.strike == Decimal("450")

    def test_mid_price(self, sample_option_contract):
        """Test option mid price calculation."""
        assert sample_option_contract.mid_price == Decimal("10.50")

    def test_symbol_uppercase(self):
        """Test symbol conversion to uppercase."""
        option = OptionContract(
            symbol="spy",
            option_type=OptionType.CALL,
            strike=Decimal("450"),
            expiry=datetime(2024, 3, 15),
            price=Decimal("10"),
        )
        assert option.symbol == "SPY"


@pytest.mark.unit
class TestOptionChain:
    """Tests for OptionChain model."""

    def test_option_chain_creation(self, sample_option_chain):
        """Test creating valid option chain."""
        assert sample_option_chain.symbol == "SPY"
        assert len(sample_option_chain.calls) == 5
        assert len(sample_option_chain.puts) == 5

    def test_time_to_expiry(self, sample_option_chain):
        """Test time to expiry calculation."""
        tte = sample_option_chain.time_to_expiry
        assert tte > 0

    def test_get_atm_strike(self, sample_option_chain):
        """Test finding ATM strike."""
        atm = sample_option_chain.get_atm_strike()
        assert atm == Decimal("450")

    def test_expiry_validation(self):
        """Test that expiry must be in future."""
        with pytest.raises(ValidationError):
            OptionChain(
                symbol="SPY",
                timestamp=datetime(2024, 3, 15),
                expiry=datetime(2024, 1, 1),  # In the past
                underlying_price=Decimal("450"),
            )


@pytest.mark.unit
class TestTrade:
    """Tests for Trade model."""

    def test_trade_creation(self, sample_trade):
        """Test creating valid trade."""
        assert sample_trade.symbol == "SPY"
        assert sample_trade.trade_type == TradeType.BUY
        assert sample_trade.quantity == 100

    def test_total_cost_buy(self):
        """Test total cost calculation for buy."""
        trade = Trade(
            timestamp=datetime(2024, 1, 1),
            symbol="SPY",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("450"),
            commission=Decimal("10"),
        )
        # (100 * 450) + 10 = 45,010
        assert trade.total_cost == Decimal("45010")

    def test_total_cost_sell(self):
        """Test total cost calculation for sell."""
        trade = Trade(
            timestamp=datetime(2024, 1, 1),
            symbol="SPY",
            trade_type=TradeType.SELL,
            quantity=100,
            price=Decimal("450"),
            commission=Decimal("10"),
        )
        # (100 * 450) - 10 = 44,990
        assert trade.total_cost == Decimal("44990")

    def test_notional_value(self, sample_trade):
        """Test notional value calculation."""
        assert sample_trade.notional_value == Decimal("45050")


@pytest.mark.unit
class TestPosition:
    """Tests for Position model."""

    def test_position_creation(self, sample_position):
        """Test creating valid position."""
        assert sample_position.symbol == "SPY"
        assert sample_position.quantity == 100

    def test_market_value(self, sample_position):
        """Test market value calculation."""
        # 100 * 455 = 45,500
        assert sample_position.market_value == Decimal("45500")

    def test_unrealized_pnl_long(self, sample_position):
        """Test unrealized P&L for long position."""
        # (455 - 450) * 100 = 500
        assert sample_position.unrealized_pnl == Decimal("500")

    def test_unrealized_pnl_short(self):
        """Test unrealized P&L for short position."""
        pos = Position(
            symbol="SPY",
            quantity=-100,  # Short
            avg_entry_price=Decimal("450"),
            current_price=Decimal("455"),
            last_update=datetime(2024, 1, 1),
        )
        # (450 - 455) * 100 = -500
        assert pos.unrealized_pnl == Decimal("-500")

    def test_unrealized_pnl_pct(self, sample_position):
        """Test unrealized P&L percentage."""
        # (500 / 45000) * 100 = 1.11%
        pnl_pct = sample_position.unrealized_pnl_pct
        assert abs(pnl_pct - Decimal("1.11")) < Decimal("0.01")

    def test_update_price(self, sample_position):
        """Test updating position price."""
        new_time = datetime(2024, 1, 2)
        sample_position.update_price(Decimal("460"), new_time)

        assert sample_position.current_price == Decimal("460")
        assert sample_position.last_update == new_time
