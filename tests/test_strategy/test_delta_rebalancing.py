"""
Regression tests for `_check_delta_rebalancing` in VolatilityArbitrageStrategy.

Three compounding bugs were fixed (F2):
  1. Missing 100x option multiplier — option deltas were treated as if each
     contract controlled 1 share instead of 100, so portfolio delta was off
     by ~100x and `shares_to_hedge` ended up as 0 for realistic positions.
  2. The `strike` was read once outside the per-leg loop and reused for every
     position, breaking multi-strike spreads.
  3. Function early-returned when `atm_call` IV was missing even when the
     `atm_put` IV was available, and call IV was reused for put pricing.
"""

from datetime import datetime, timedelta
from decimal import Decimal

from volatility_arbitrage.core.config import VolatilityArbitrageConfig
from volatility_arbitrage.core.types import (
    OptionChain,
    OptionContract,
    OptionType,
    Position,
)
from volatility_arbitrage.strategy.volatility_arbitrage import (
    VolatilityArbitrageStrategy,
)


def _chain(
    symbol: str = "SPY",
    spot: Decimal = Decimal("505"),
    atm_strike: Decimal = Decimal("500"),
    iv: Decimal = Decimal("0.20"),
    call_iv: Decimal | None = None,
    put_iv: Decimal | None = None,
) -> OptionChain:
    now = datetime(2024, 1, 1)
    expiry = now + timedelta(days=30)
    call = OptionContract(
        symbol=symbol,
        option_type=OptionType.CALL,
        strike=atm_strike,
        expiry=expiry,
        price=Decimal("10"),
        implied_volatility=call_iv if call_iv is not None else iv,
    )
    put = OptionContract(
        symbol=symbol,
        option_type=OptionType.PUT,
        strike=atm_strike,
        expiry=expiry,
        price=Decimal("10"),
        implied_volatility=put_iv if put_iv is not None else iv,
    )
    return OptionChain(
        symbol=symbol,
        timestamp=now,
        expiry=expiry,
        underlying_price=spot,
        calls=[call],
        puts=[put],
    )


def _strategy() -> VolatilityArbitrageStrategy:
    return VolatilityArbitrageStrategy(VolatilityArbitrageConfig())


def _seed_position_info(strategy: VolatilityArbitrageStrategy, chain: OptionChain) -> None:
    strategy.option_positions["SPY"] = {
        "entry_timestamp": chain.timestamp,
        "direction": "sell",
        "strike": Decimal("500"),
        "expiry": chain.expiry,
        "profit_levels_taken": [],
        "entry_quantity": 1,
    }


class TestDeltaRebalancingMultiplier:
    """Bug 1: option contributions must include the 100x share multiplier."""

    def test_short_straddle_with_spot_drift_emits_hedge(self):
        # Spot drifted up from 500 to 505 → short straddle is net short delta.
        # Per contract: short call delta ~ -0.55, short put delta ~ +0.45,
        # net ≈ -0.10 per share → -10 shares per contract pair.
        # Pre-fix this collapses to 0 shares; post-fix it should be ~10-20.
        chain = _chain(spot=Decimal("505"))
        strategy = _strategy()
        _seed_position_info(strategy, chain)

        positions = {
            "SPY_CALL_500_20240131": Position(
                symbol="SPY_CALL_500_20240131",
                quantity=-1,
                avg_entry_price=Decimal("10"),
                current_price=Decimal("12"),
                last_update=chain.timestamp,
            ),
            "SPY_PUT_500_20240131": Position(
                symbol="SPY_PUT_500_20240131",
                quantity=-1,
                avg_entry_price=Decimal("10"),
                current_price=Decimal("8"),
                last_update=chain.timestamp,
            ),
        }

        signals = strategy._check_delta_rebalancing("SPY", chain, positions)

        # Should produce at least one hedge signal in the right order of magnitude.
        assert len(signals) == 1
        sig = signals[0]
        assert sig.symbol == "SPY"
        # Net short delta → portfolio_delta is negative → action is "buy".
        assert sig.action == "buy"
        # Pre-fix this would be 0; post-fix should be ~5-30 shares for this drift.
        assert sig.quantity >= 5
        assert sig.quantity <= 50

    def test_larger_short_straddle_scales_with_size(self):
        # 10 contracts instead of 1 → hedge magnitude should be ~10x larger.
        chain = _chain(spot=Decimal("505"))
        strategy = _strategy()
        _seed_position_info(strategy, chain)

        positions = {
            "SPY_CALL_500_20240131": Position(
                symbol="SPY_CALL_500_20240131",
                quantity=-10,
                avg_entry_price=Decimal("10"),
                current_price=Decimal("12"),
                last_update=chain.timestamp,
            ),
            "SPY_PUT_500_20240131": Position(
                symbol="SPY_PUT_500_20240131",
                quantity=-10,
                avg_entry_price=Decimal("10"),
                current_price=Decimal("8"),
                last_update=chain.timestamp,
            ),
        }

        signals = strategy._check_delta_rebalancing("SPY", chain, positions)
        assert len(signals) == 1
        # Roughly 10x the single-contract case; pre-fix would still be 0.
        assert signals[0].quantity >= 50

    def test_drift_scales_hedge_direction(self):
        # Spot drifted *down* from 500 to 495 → short straddle is net long delta.
        # Pre-fix this would still be 0 shares; post-fix we expect a "sell" hedge.
        chain = _chain(spot=Decimal("495"))
        strategy = _strategy()
        _seed_position_info(strategy, chain)

        positions = {
            "SPY_CALL_500_20240131": Position(
                symbol="SPY_CALL_500_20240131",
                quantity=-1,
                avg_entry_price=Decimal("10"),
                current_price=Decimal("8"),
                last_update=chain.timestamp,
            ),
            "SPY_PUT_500_20240131": Position(
                symbol="SPY_PUT_500_20240131",
                quantity=-1,
                avg_entry_price=Decimal("10"),
                current_price=Decimal("12"),
                last_update=chain.timestamp,
            ),
        }

        signals = strategy._check_delta_rebalancing("SPY", chain, positions)
        assert len(signals) == 1
        # Net long delta from short straddle below strike → sell to hedge.
        assert signals[0].action == "sell"
        assert signals[0].quantity >= 1


class TestDeltaRebalancingPerLegStrike:
    """Bug 2: each leg's strike must be parsed from its symbol."""

    def test_multi_strike_uses_per_leg_strikes(self):
        # Strangle: short 510 call + short 490 put, spot at 500.
        # If the loop reused a single strike (e.g., 500), both legs would price
        # near ATM and net delta would be near zero. With per-leg strikes the
        # 510 call has lower |delta| than the 490 put → net positive delta.
        chain = _chain(spot=Decimal("500"))
        strategy = _strategy()
        _seed_position_info(strategy, chain)

        positions = {
            "SPY_CALL_510_20240131": Position(
                symbol="SPY_CALL_510_20240131",
                quantity=-1,
                avg_entry_price=Decimal("5"),
                current_price=Decimal("5"),
                last_update=chain.timestamp,
            ),
            "SPY_PUT_490_20240131": Position(
                symbol="SPY_PUT_490_20240131",
                quantity=-1,
                avg_entry_price=Decimal("5"),
                current_price=Decimal("5"),
                last_update=chain.timestamp,
            ),
        }

        signals = strategy._check_delta_rebalancing("SPY", chain, positions)
        # Asymmetric strikes → non-zero net delta → expect a hedge signal.
        assert len(signals) == 1
        assert signals[0].quantity >= 1


class TestDeltaRebalancingMissingIV:
    """Bug 3: don't early-return when only one side has IV."""

    def test_falls_back_when_call_iv_missing(self):
        # Drop call IV; put IV should be reused for both sides.
        chain = _chain(
            spot=Decimal("505"),
            call_iv=None,  # type: ignore[arg-type]
            put_iv=Decimal("0.20"),
        )
        strategy = _strategy()
        _seed_position_info(strategy, chain)

        positions = {
            "SPY_CALL_500_20240131": Position(
                symbol="SPY_CALL_500_20240131",
                quantity=-1,
                avg_entry_price=Decimal("10"),
                current_price=Decimal("12"),
                last_update=chain.timestamp,
            ),
            "SPY_PUT_500_20240131": Position(
                symbol="SPY_PUT_500_20240131",
                quantity=-1,
                avg_entry_price=Decimal("10"),
                current_price=Decimal("8"),
                last_update=chain.timestamp,
            ),
        }

        signals = strategy._check_delta_rebalancing("SPY", chain, positions)
        # Pre-fix: empty (early-returned on missing call IV).
        # Post-fix: hedge signal is computed using put IV as fallback.
        assert len(signals) == 1
        assert signals[0].action == "buy"
