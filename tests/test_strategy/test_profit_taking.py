"""
Regression tests for tiered profit-take sizing.

Bug F4: `_check_profit_taking` sized each tier off the *current* live position
quantity instead of the *original* entry quantity. With default tiers
(0.33 / 0.33 / 0.34 summing to 1.0, intended to fully close), the actual
cumulative exit was only ~70% — ~30% of the position lingered forever.

These tests pin the fix: tier sizing must base off `entry_quantity` so the
three default tiers cumulatively close the entire entry size.
"""

from datetime import datetime
from decimal import Decimal

from volatility_arbitrage.core.config import VolatilityArbitrageConfig
from volatility_arbitrage.core.types import Position
from volatility_arbitrage.strategy.volatility_arbitrage import (
    VolatilityArbitrageStrategy,
)


def _make_strategy() -> VolatilityArbitrageStrategy:
    cfg = VolatilityArbitrageConfig(
        use_profit_taking=True,
        profit_take_levels=[Decimal("0.25"), Decimal("0.50"), Decimal("0.75")],
        profit_take_sizes=[Decimal("0.33"), Decimal("0.33"), Decimal("0.34")],
    )
    return VolatilityArbitrageStrategy(cfg)


def _make_long_position(symbol: str, quantity: int, pnl_pct: Decimal) -> Position:
    """Long position whose unrealized PnL is `pnl_pct` of cost basis."""
    entry = Decimal("10")
    current = entry * (Decimal("1") + pnl_pct)
    return Position(
        symbol=symbol,
        quantity=quantity,
        avg_entry_price=entry,
        current_price=current,
        last_update=datetime(2024, 1, 1),
    )


def test_tiered_profit_take_fully_closes_position() -> None:
    """All three default tiers together must close the entire entry quantity."""
    strat = _make_strategy()
    underlying = "SPY"
    leg_symbol = "SPY_PUT_500_20240115"
    entry_qty = 100

    strat.option_positions[underlying] = {
        "entry_timestamp": datetime(2024, 1, 1),
        "direction": "sell",
        "strike": Decimal("500"),
        "expiry": datetime(2024, 1, 15),
        "profit_levels_taken": [],
        "entry_quantity": entry_qty,
    }

    remaining = entry_qty
    cumulative_close = 0

    # Walk all three tiers; PnL crosses 25%, 50%, 75% in turn.
    for pnl_pct in (Decimal("0.25"), Decimal("0.50"), Decimal("0.75")):
        # Position is short (quantity < 0); use negative quantity to mirror
        # what the strategy actually books for short-vol entries.
        pos = _make_long_position(leg_symbol, -remaining, pnl_pct)
        # Short position: profit when current < entry. Flip price so PnL pct matches.
        pos.current_price = pos.avg_entry_price * (Decimal("1") - pnl_pct)

        signals = strat._check_profit_taking(underlying, {leg_symbol: pos})
        assert len(signals) == 1, f"Expected one signal at PnL={pnl_pct}, got {signals}"
        sig = signals[0]
        assert sig.symbol == leg_symbol
        # Short position closes via "buy"
        assert sig.action == "buy"
        cumulative_close += sig.quantity
        remaining -= sig.quantity

    # Three default tiers sum to 1.0 → must close the full entry quantity
    # (allow ±1 for integer rounding across tiers).
    assert abs(cumulative_close - entry_qty) <= 1, (
        f"Cumulative close {cumulative_close} does not match entry {entry_qty}; "
        "tiers are compounding on the remaining quantity instead of the entry quantity."
    )
    assert remaining <= 1


def test_profit_take_clamps_to_live_quantity() -> None:
    """If entry_quantity * tier_size exceeds live quantity, clamp to live."""
    strat = _make_strategy()
    underlying = "SPY"
    leg_symbol = "SPY_PUT_500_20240115"

    # Entry quantity says 100, but live quantity is only 5 (e.g. partial fills).
    strat.option_positions[underlying] = {
        "entry_timestamp": datetime(2024, 1, 1),
        "direction": "sell",
        "strike": Decimal("500"),
        "expiry": datetime(2024, 1, 15),
        "profit_levels_taken": [],
        "entry_quantity": 100,
    }
    pos = _make_long_position(leg_symbol, -5, Decimal("0.25"))
    pos.current_price = pos.avg_entry_price * Decimal("0.75")

    signals = strat._check_profit_taking(underlying, {leg_symbol: pos})
    assert len(signals) == 1
    # int(100 * 0.33) = 33, clamped to 5.
    assert signals[0].quantity == 5


def test_profit_take_legacy_falls_back_to_live_quantity() -> None:
    """Positions stored before F1 lack `entry_quantity` and must still work."""
    strat = _make_strategy()
    underlying = "SPY"
    leg_symbol = "SPY_PUT_500_20240115"

    # No entry_quantity key — simulate a pre-F1 position dict.
    strat.option_positions[underlying] = {
        "entry_timestamp": datetime(2024, 1, 1),
        "direction": "sell",
        "strike": Decimal("500"),
        "expiry": datetime(2024, 1, 15),
        "profit_levels_taken": [],
    }
    pos = _make_long_position(leg_symbol, -100, Decimal("0.25"))
    pos.current_price = pos.avg_entry_price * Decimal("0.75")

    signals = strat._check_profit_taking(underlying, {leg_symbol: pos})
    assert len(signals) == 1
    # Falls back to old behavior: int(100 * 0.33) = 33.
    assert signals[0].quantity == 33
