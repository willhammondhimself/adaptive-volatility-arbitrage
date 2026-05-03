"""
Regression tests for F5: stop-loss / profit-take must not aggregate the
bare-symbol underlying delta-hedge stock position with option legs.

Pre-F5, `_check_stop_loss` and `_check_profit_taking` matched related positions
with `if symbol in pos_symbol`. For symbol="SPY", the bare-`SPY` underlying
hedge stock position satisfies `"SPY" in "SPY"` and got folded into the
option-strategy P&L. Two consequences:

1. Stock cost basis has no 100x contract multiplier; mixed-unit `pnl_pct`
   is meaningless and miscalibrates thresholds.
2. The inner profit-take loop emitted partial-close Signals against the
   bare-`SPY` hedge, partially dismantling delta neutrality while option
   legs remained open.

The fix uses `pos_symbol.startswith(symbol + "_")` to match only option legs
(format `SYMBOL_TYPE_STRIKE_EXPIRY`).
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
        max_loss_pct=Decimal("50"),
    )
    return VolatilityArbitrageStrategy(cfg)


def _short_option(symbol: str, qty: int, pnl_frac: Decimal) -> Position:
    """Short option leg whose unrealized PnL is pnl_frac of cost basis."""
    entry = Decimal("10")
    # Short profits when current price drops; flip price by pnl_frac.
    current = entry * (Decimal("1") - pnl_frac)
    return Position(
        symbol=symbol,
        quantity=-abs(qty),
        avg_entry_price=entry,
        current_price=current,
        last_update=datetime(2024, 1, 1),
    )


def _underlying_stock(symbol: str, qty: int) -> Position:
    """Bare-symbol long stock used as structural delta hedge."""
    return Position(
        symbol=symbol,
        quantity=qty,
        avg_entry_price=Decimal("440"),
        current_price=Decimal("440"),
        last_update=datetime(2024, 1, 1),
    )


def _seed_option_position(strat: VolatilityArbitrageStrategy, underlying: str, entry_qty: int) -> None:
    strat.option_positions[underlying] = {
        "entry_timestamp": datetime(2024, 1, 1),
        "direction": "sell",
        "strike": Decimal("440"),
        "expiry": datetime(2024, 6, 15),
        "profit_levels_taken": [],
        "entry_quantity": entry_qty,
    }


def test_profit_take_does_not_target_underlying_hedge() -> None:
    """Profit-take signals must only target option legs, never bare SPY stock."""
    strat = _make_strategy()
    underlying = "SPY"
    call_leg = "SPY_CALL_440_20240615"
    put_leg = "SPY_PUT_440_20240615"

    entry_qty = 10
    _seed_option_position(strat, underlying, entry_qty)

    positions = {
        underlying: _underlying_stock(underlying, 100),  # bare-SPY hedge
        call_leg: _short_option(call_leg, entry_qty, Decimal("0.25")),
        put_leg: _short_option(put_leg, entry_qty, Decimal("0.25")),
    }

    signals = strat._check_profit_taking(underlying, positions)

    assert signals, "Expected profit-take signals at +25% PnL"
    targeted = {s.symbol for s in signals}
    assert underlying not in targeted, (
        f"Profit-take emitted a signal against bare-{underlying} underlying "
        f"hedge stock; would partially dismantle delta neutrality. Signals: {signals}"
    )
    assert targeted.issubset({call_leg, put_leg}), (
        f"Profit-take targeted unexpected symbols: {targeted}"
    )


def test_stop_loss_pnl_excludes_underlying_hedge() -> None:
    """Stop-loss aggregation must ignore the bare-symbol stock leg.

    Construct a scenario where option legs are at -60% (past the 50% stop)
    but a flat stock hedge with large cost basis would dilute the combined
    pnl_pct below threshold. Pre-F5 the stop would not fire; post-F5 it does.
    """
    strat = _make_strategy()
    underlying = "SPY"
    call_leg = "SPY_CALL_440_20240615"

    _seed_option_position(strat, underlying, 1)

    # Option leg at -60% (past stop threshold of -50%).
    call_pos = Position(
        symbol=call_leg,
        quantity=-1,
        avg_entry_price=Decimal("10"),
        current_price=Decimal("16"),  # short loses when price rises
        last_update=datetime(2024, 1, 1),
    )

    # Big flat stock hedge: cost basis dominant, zero PnL.
    stock_pos = _underlying_stock(underlying, 10_000)

    positions = {underlying: stock_pos, call_leg: call_pos}
    signals = strat._check_stop_loss(underlying, positions)

    assert signals, (
        "Stop-loss did not fire even though the option leg is past threshold; "
        "the bare-symbol stock cost basis is being mixed in and diluting pnl_pct."
    )
    # Close signals should also avoid the bare hedge (defense-in-depth via
    # _generate_close_signals; F5 itself does not change that path, but the
    # caller path here is what matters).
    targeted = {s.symbol for s in signals}
    assert call_leg in targeted
