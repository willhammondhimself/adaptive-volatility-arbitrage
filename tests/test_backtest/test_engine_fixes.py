"""
Regression tests for multi_asset_engine fixes.

Covers:
- F9: daily delta-hedge cost must scale with underlying notional, not share count.
- F13a: data-quality counters must only count executed trades (post cash/margin checks).
- F13b: PLACEHOLDER IV warning must not fire when use_real_options_data=False
        (i.e. user opted out of real-data lookup).
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from volatility_arbitrage.backtest import (
    MultiAssetBacktestEngine,
    MultiAssetPosition,
)
from volatility_arbitrage.core.config import BacktestConfig, VolatilityArbitrageConfig
from volatility_arbitrage.core.types import OptionType
from volatility_arbitrage.strategy.base import BuyAndHoldStrategy, Signal


def _make_engine(initial_cash="100000", use_real_options_data=False, daily_hedge_cost=None):
    kwargs = {"initial_capital": Decimal(initial_cash)}
    if daily_hedge_cost is not None:
        kwargs["daily_hedge_cost"] = Decimal(daily_hedge_cost)
    config = BacktestConfig(**kwargs)
    strategy = BuyAndHoldStrategy("SPY", 1)
    sc = VolatilityArbitrageConfig(use_real_options_data=use_real_options_data)
    engine = MultiAssetBacktestEngine(config, strategy, strategy_config=sc)
    return engine


def _add_long_call(engine, *, underlying_price, contracts=10, iv="0.20"):
    """Add a long ATM-ish call so calculate_greeks returns a non-trivial delta."""
    now = datetime(2024, 1, 2)
    expiry = now + timedelta(days=30)
    pos = MultiAssetPosition(
        symbol="SPY_CALL_400_20240201",
        asset_type="option",
        quantity=contracts,
        entry_price=Decimal("10"),
        current_price=Decimal("10"),
        last_update=now,
        option_type=OptionType.CALL,
        strike=Decimal("400"),
        expiry=expiry,
        underlying_price=Decimal(str(underlying_price)),
        implied_volatility=Decimal(iv),
        risk_free_rate=Decimal("0.05"),
    )
    engine.multi_positions[pos.symbol] = pos
    return pos


@pytest.mark.unit
class TestHedgeCostScalesWithUnderlying:
    """F9: hedge cost must use dollar-delta (delta * 100 * S), not share count."""

    def test_hedge_cost_higher_for_higher_underlying_price(self):
        """Same option, higher underlying price → higher hedge cost."""
        rate = "0.0002"

        # Engine A: underlying at $100
        eng_a = _make_engine(daily_hedge_cost=rate)
        pos_a = _add_long_call(eng_a, underlying_price=100)
        cash_before_a = eng_a.cash

        # Engine B: same setup, underlying at $1000
        eng_b = _make_engine(daily_hedge_cost=rate)
        pos_b = _add_long_call(eng_b, underlying_price=1000)
        cash_before_b = eng_b.cash

        # Build a one-row day_data so _process_day can run end-to-end. Use a
        # symbol the strategy won't trade on this bar (BuyAndHoldStrategy buys
        # SPY once it sees a SPY row; using "ZZZ" keeps the day quiet).
        ts = datetime(2024, 1, 3)
        day = pd.DataFrame({
            "timestamp": [ts],
            "symbol": ["ZZZ"],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1_000_000],
        })

        eng_a._process_day(ts, day)
        eng_b._process_day(ts, day)

        cost_a = cash_before_a - eng_a.cash
        cost_b = cash_before_b - eng_b.cash

        # Hedge cost should scale ~10x with underlying price (100 -> 1000)
        # Margin financing piece is identical (no shorts), so the ratio is purely
        # the hedge-cost component.
        assert cost_b > cost_a * Decimal("9"), (
            f"hedge cost did not scale with underlying: "
            f"S=100 cost={cost_a}, S=1000 cost={cost_b}"
        )

    def test_hedge_cost_matches_dollar_delta_formula(self):
        """Cost equals |delta| * 100 * S * rate, where delta is position-scaled."""
        rate = Decimal("0.001")  # large rate to dwarf any margin financing
        eng = _make_engine(daily_hedge_cost=str(rate))
        pos = _add_long_call(eng, underlying_price=400, contracts=10)

        # Compute expected hedge cost from the position's own greeks
        greeks = pos.calculate_greeks()
        assert greeks is not None
        expected_cost = abs(greeks.delta) * Decimal("100") * pos.underlying_price * rate

        cash_before = eng.cash
        ts = datetime(2024, 1, 3)
        day = pd.DataFrame({
            "timestamp": [ts], "symbol": ["ZZZ"],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1_000_000],
        })
        eng._process_day(ts, day)

        # No short options → no margin financing cost in this engine
        actual_cost = cash_before - eng.cash
        # Allow a tiny tolerance because greeks recompute identically; should match.
        assert abs(actual_cost - expected_cost) < Decimal("0.01"), (
            f"actual={actual_cost} expected={expected_cost}"
        )


@pytest.mark.unit
class TestDataQualityCountersOnlyCountExecutedTrades:
    """F13a: counters must not increment for trades rejected by cash/margin check."""

    def test_counters_not_incremented_when_cash_insufficient(self):
        # Tiny cash so any option buy will be rejected
        eng = _make_engine(initial_cash="1", use_real_options_data=False)

        ts = datetime(2024, 1, 3)
        # Underlying price row required by _execute_option_signal
        day = pd.DataFrame({
            "timestamp": [ts], "symbol": ["SPY"],
            "open": [400.0], "high": [400.0], "low": [400.0], "close": [400.0],
            "volume": [1_000_000],
        })

        # Buy 10 contracts of an ATM call ~30d out — premium > $1 cash
        expiry_str = (ts + timedelta(days=30)).strftime("%Y%m%d")
        signal = Signal(
            symbol=f"SPY_CALL_400_{expiry_str}",
            action="buy",
            quantity=10,
        )

        eng._execute_option_signal(signal, ts, day)

        # Trade should be rejected; counters should remain 0
        assert len(eng.trades) == 0
        assert eng.trades_with_real_data == 0
        assert eng.trades_with_placeholder_data == 0

    def test_counter_increments_when_trade_executes(self):
        eng = _make_engine(initial_cash="1000000", use_real_options_data=False)

        ts = datetime(2024, 1, 3)
        day = pd.DataFrame({
            "timestamp": [ts], "symbol": ["SPY"],
            "open": [400.0], "high": [400.0], "low": [400.0], "close": [400.0],
            "volume": [1_000_000],
        })
        expiry_str = (ts + timedelta(days=30)).strftime("%Y%m%d")
        signal = Signal(
            symbol=f"SPY_CALL_400_{expiry_str}",
            action="buy",
            quantity=1,
        )

        eng._execute_option_signal(signal, ts, day)

        assert len(eng.trades) == 1
        # use_real_options_data=False → placeholder path
        assert eng.trades_with_placeholder_data == 1
        assert eng.trades_with_real_data == 0


@pytest.mark.unit
class TestPlaceholderWarningGated:
    """F13b: no per-trade PLACEHOLDER IV warning when user opted out of real data."""

    def test_no_warning_when_use_real_options_data_false(self, caplog):
        import logging

        eng = _make_engine(initial_cash="1000000", use_real_options_data=False)

        ts = datetime(2024, 1, 3)
        day = pd.DataFrame({
            "timestamp": [ts], "symbol": ["SPY"],
            "open": [400.0], "high": [400.0], "low": [400.0], "close": [400.0],
            "volume": [1_000_000],
        })
        expiry_str = (ts + timedelta(days=30)).strftime("%Y%m%d")
        signal = Signal(
            symbol=f"SPY_CALL_400_{expiry_str}",
            action="buy",
            quantity=1,
        )

        with caplog.at_level(logging.WARNING):
            eng._execute_option_signal(signal, ts, day)

        offending = [
            r for r in caplog.records
            if "PLACEHOLDER IV" in r.getMessage() and "No real data available" in r.getMessage()
        ]
        assert offending == [], (
            "PLACEHOLDER IV warning fired even though use_real_options_data=False"
        )
