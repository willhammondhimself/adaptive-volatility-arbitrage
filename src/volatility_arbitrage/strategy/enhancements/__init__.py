"""
Strategy Enhancement Modules for Volatility Arbitrage.

Each enhancement is toggleable via config flags and can be tested
independently via the ablation study framework.

Enhancements:
1. Regime Transition Signals - Detect vol regime CHANGES
2. Term Structure Leverage - Scale position by term structure
3. Vol-of-Vol (VVIX) Integration - Use VVIX as sizing modifier
4. Intraday Vol Patterns - Overnight vs intraday decomposition
5. Dynamic Signal Weighting - ML-based adaptive weights
6. Asymmetric Profit Taking - Different targets for long/short vol
7. Alternative RV Methods - Parkinson, Garman-Klass, Yang-Zhang
"""

# Enhancement 1: Regime Transition Signals
from volatility_arbitrage.strategy.enhancements.regime_transition import (
    RegimeTransitionConfig,
    RegimeTransitionSignal,
    integrate_regime_signal,
)

# Enhancement 2: Term Structure Leverage
from volatility_arbitrage.strategy.enhancements.term_structure_leverage import (
    TermStructureLeverageConfig,
    TermStructureLeverageCalculator,
    TermStructureRegime,
)

# Enhancement 3: Vol-of-Vol (VVIX) Integration
from volatility_arbitrage.strategy.enhancements.vov_signal import (
    VoVConfig,
    VoVSignalGenerator,
)

# Enhancement 4: Intraday Vol Patterns
from volatility_arbitrage.strategy.enhancements.intraday_volatility import (
    IntradayVolConfig,
    IntradayVolCalculator,
    IntradayVolDecomposition,
)

# Enhancement 5: Dynamic Signal Weighting
from volatility_arbitrage.strategy.enhancements.dynamic_weighting import (
    DynamicWeightingConfig,
    DynamicSignalWeighter,
    SignalOutcome,
)

# Enhancement 6: Asymmetric Profit Taking
from volatility_arbitrage.strategy.enhancements.asymmetric_profit import (
    AsymmetricProfitConfig,
    AsymmetricProfitManager,
)

# Enhancement 7: Alternative RV Methods
from volatility_arbitrage.strategy.enhancements.alt_rv_methods import (
    calculate_rv,
    parkinson_rv,
    garman_klass_rv,
    yang_zhang_rv,
    close_to_close_rv,
    ensemble_rv,
)

__all__ = [
    # Regime Transition
    "RegimeTransitionConfig",
    "RegimeTransitionSignal",
    "integrate_regime_signal",
    # Term Structure Leverage
    "TermStructureLeverageConfig",
    "TermStructureLeverageCalculator",
    "TermStructureRegime",
    # Vol-of-Vol
    "VoVConfig",
    "VoVSignalGenerator",
    # Intraday Volatility
    "IntradayVolConfig",
    "IntradayVolCalculator",
    "IntradayVolDecomposition",
    # Dynamic Weighting
    "DynamicWeightingConfig",
    "DynamicSignalWeighter",
    "SignalOutcome",
    # Asymmetric Profit
    "AsymmetricProfitConfig",
    "AsymmetricProfitManager",
    # Alternative RV Methods
    "calculate_rv",
    "parkinson_rv",
    "garman_klass_rv",
    "yang_zhang_rv",
    "close_to_close_rv",
    "ensemble_rv",
]
