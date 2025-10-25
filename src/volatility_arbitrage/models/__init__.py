"""Pricing models and volatility forecasting."""

from volatility_arbitrage.models.black_scholes import (
    BlackScholesModel,
    Greeks,
    calculate_implied_volatility,
)
from volatility_arbitrage.models.volatility import (
    VolatilityForecaster,
    HistoricalVolatility,
    EWMAVolatility,
    GARCHVolatility,
)
# Optional imports - don't break if not installed
try:
    from volatility_arbitrage.models.heston import (
        HestonModel,
        HestonParameters,
        HestonCalibrator,
        compare_to_black_scholes,
    )
except ImportError:
    pass

try:
    from volatility_arbitrage.models.regime import (
        RegimeDetector,
        RegimeStatistics,
        GaussianMixtureRegimeDetector,
        HiddenMarkovRegimeDetector,
        regime_conditional_metrics,
    )
except ImportError:
    pass

__all__ = [
    "BlackScholesModel",
    "Greeks",
    "calculate_implied_volatility",
    "VolatilityForecaster",
    "HistoricalVolatility",
    "EWMAVolatility",
    "GARCHVolatility",
    "HestonModel",
    "HestonParameters",
    "HestonCalibrator",
    "compare_to_black_scholes",
    "RegimeDetector",
    "RegimeStatistics",
    "GaussianMixtureRegimeDetector",
    "HiddenMarkovRegimeDetector",
    "regime_conditional_metrics",
]
