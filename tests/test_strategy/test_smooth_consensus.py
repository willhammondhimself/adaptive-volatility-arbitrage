"""
Regression tests for `_smooth_consensus` (F10).

Pre-fix bug: the EMA was recomputed each call from a finite deque
(`maxlen = signal_smoothing_window * 2`). Once the buffer rolled over,
the impulse value at index 0 was dropped and the EMA was implicitly
re-seeded from a recent value, breaking the geometric-decay property
of a true running EMA.

Post-fix: a single `self.ema_state[symbol]` is updated incrementally,
preserving the full geometric tail.
"""

from decimal import Decimal

import pytest

from volatility_arbitrage.core.config import VolatilityArbitrageConfig
from volatility_arbitrage.strategy.volatility_arbitrage import (
    VolatilityArbitrageStrategy,
)


def _make_strategy(window: int = 3, min_history: int = 1) -> VolatilityArbitrageStrategy:
    cfg = VolatilityArbitrageConfig(
        use_signal_smoothing=True,
        signal_smoothing_window=window,
        signal_smoothing_min_history=min_history,
    )
    return VolatilityArbitrageStrategy(cfg)


def test_impulse_geometric_decay_survives_buffer_rollover():
    # window=3 → alpha = 2/(3+1) = 0.5
    # Feed an impulse [1.0, 0, 0, 0, 0, 0, 0]; with min_history=1 every call
    # returns the running EMA. After the impulse, EMA should decay geometrically:
    # ema_n = 0.5^n. By call 7 (six zeros after impulse), ema = 0.5^6 = 0.015625.
    strat = _make_strategy(window=3, min_history=1)
    sym = "TEST"

    sequence = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = [float(strat._smooth_consensus(sym, Decimal(str(v)))) for v in sequence]

    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.25)
    assert out[3] == pytest.approx(0.125)
    assert out[4] == pytest.approx(0.0625)
    assert out[5] == pytest.approx(0.03125)
    assert out[6] == pytest.approx(0.015625)


def test_warmup_returns_raw_until_min_history():
    strat = _make_strategy(window=3, min_history=3)
    sym = "WARM"

    # First two calls are warm-up: must return the raw input verbatim.
    assert strat._smooth_consensus(sym, Decimal("0.8")) == Decimal("0.8")
    assert strat._smooth_consensus(sym, Decimal("0.4")) == Decimal("0.4")

    # Third call hits min_history; returns smoothed EMA.
    # alpha=0.5, state after calls: 0.8 → 0.6 → 0.5
    out3 = float(strat._smooth_consensus(sym, Decimal("0.4")))
    assert out3 == pytest.approx(0.5)


def test_disabled_smoothing_passthrough():
    cfg = VolatilityArbitrageConfig(use_signal_smoothing=False)
    strat = VolatilityArbitrageStrategy(cfg)
    raw = Decimal("0.42")
    assert strat._smooth_consensus("X", raw) is raw


def test_per_symbol_state_isolated():
    strat = _make_strategy(window=3, min_history=1)
    strat._smooth_consensus("A", Decimal("1.0"))
    strat._smooth_consensus("A", Decimal("0.0"))
    # B starts fresh — first observation should pass through as the seed.
    out_b = float(strat._smooth_consensus("B", Decimal("0.7")))
    assert out_b == pytest.approx(0.7)
