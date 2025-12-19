"""
Intraday Volatility Patterns Enhancement.

Decomposes realized volatility into overnight vs intraday components:

- Overnight RV: Volatility from close-to-open (news, international markets)
- Intraday RV: Volatility from open-to-close (US market hours)

Signal logic:
- Overnight dominant (ratio > 1.5): News-driven market, reduce position size
- Intraday dominant (ratio < 0.67): Orderly momentum, slight size boost
- Balanced: No adjustment
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque
import numpy as np
import pandas as pd


@dataclass
class IntradayVolConfig:
    """Configuration for intraday volatility decomposition."""

    # Enable flag
    enabled: bool = True

    # RV calculation window
    window: int = 20

    # Regime thresholds (overnight/intraday ratio)
    overnight_dominant_threshold: float = 1.5   # Ratio > 1.5 = overnight dominant
    intraday_dominant_threshold: float = 0.67   # Ratio < 0.67 = intraday dominant

    # Position size scalars
    overnight_dominant_scalar: float = 0.75     # Reduce size when overnight dominant
    intraday_dominant_scalar: float = 1.1       # Boost when intraday dominant
    balanced_scalar: float = 1.0                # No change when balanced


@dataclass
class IntradayVolDecomposition:
    """Result of intraday volatility decomposition."""
    overnight_rv: float
    intraday_rv: float
    total_rv: float
    ratio: float
    regime: str


class IntradayVolCalculator:
    """
    Calculate and decompose volatility into overnight vs intraday components.

    Overnight return = log(open[t] / close[t-1])
    Intraday return = log(close[t] / open[t])

    Usage:
        calc = IntradayVolCalculator(config)

        # Update with daily OHLC
        decomp = calc.update(open=100.5, close=101.0, prev_close=100.0)

        # Get position size scalar
        scalar = calc.get_position_scalar()
    """

    def __init__(self, config: IntradayVolConfig = None):
        self.config = config or IntradayVolConfig()
        self.overnight_returns: deque = deque(maxlen=252)
        self.intraday_returns: deque = deque(maxlen=252)
        self.prev_close: Optional[float] = None
        self.decomposition: Optional[IntradayVolDecomposition] = None

    def update(
        self,
        open_price: float,
        close_price: float,
        prev_close: Optional[float] = None
    ) -> Optional[IntradayVolDecomposition]:
        """
        Update with new OHLC data and recalculate decomposition.

        Args:
            open_price: Today's open
            close_price: Today's close
            prev_close: Previous day's close (optional, uses stored value if None)

        Returns:
            IntradayVolDecomposition or None if insufficient history
        """
        if not self.config.enabled:
            return None

        # Use provided prev_close or stored value
        if prev_close is None:
            prev_close = self.prev_close

        if prev_close is not None and prev_close > 0:
            # Calculate returns
            overnight_ret = np.log(open_price / prev_close)
            intraday_ret = np.log(close_price / open_price)

            self.overnight_returns.append(overnight_ret)
            self.intraday_returns.append(intraday_ret)

        # Store for next iteration
        self.prev_close = close_price

        # Need enough history
        if len(self.overnight_returns) < self.config.window:
            return None

        # Calculate component RVs
        overnight_arr = np.array(list(self.overnight_returns))[-self.config.window:]
        intraday_arr = np.array(list(self.intraday_returns))[-self.config.window:]

        overnight_rv = overnight_arr.std() * np.sqrt(252)
        intraday_rv = intraday_arr.std() * np.sqrt(252)

        # Avoid division by zero
        if intraday_rv < 1e-6:
            ratio = 1.0
        else:
            ratio = overnight_rv / intraday_rv

        # Classify regime
        if ratio > self.config.overnight_dominant_threshold:
            regime = "OVERNIGHT_DOMINANT"
        elif ratio < self.config.intraday_dominant_threshold:
            regime = "INTRADAY_DOMINANT"
        else:
            regime = "BALANCED"

        # Total RV (assuming independence)
        total_rv = np.sqrt(overnight_rv**2 + intraday_rv**2)

        self.decomposition = IntradayVolDecomposition(
            overnight_rv=overnight_rv,
            intraday_rv=intraday_rv,
            total_rv=total_rv,
            ratio=ratio,
            regime=regime,
        )

        return self.decomposition

    def get_position_scalar(self) -> float:
        """
        Get position size scalar based on overnight/intraday regime.

        Returns:
            Scalar to multiply base position size
        """
        if not self.config.enabled or self.decomposition is None:
            return 1.0

        if self.decomposition.regime == "OVERNIGHT_DOMINANT":
            return self.config.overnight_dominant_scalar
        elif self.decomposition.regime == "INTRADAY_DOMINANT":
            return self.config.intraday_dominant_scalar
        else:
            return self.config.balanced_scalar

    def get_statistics(self) -> dict:
        """Get decomposition statistics for logging."""
        if self.decomposition is None:
            return {
                "overnight_rv": None,
                "intraday_rv": None,
                "ratio": None,
                "regime": "UNKNOWN",
                "history_len": len(self.overnight_returns),
            }

        return {
            "overnight_rv": self.decomposition.overnight_rv,
            "intraday_rv": self.decomposition.intraday_rv,
            "total_rv": self.decomposition.total_rv,
            "ratio": self.decomposition.ratio,
            "regime": self.decomposition.regime,
            "history_len": len(self.overnight_returns),
        }

    def reset(self) -> None:
        """Reset state for new backtest run."""
        self.overnight_returns.clear()
        self.intraday_returns.clear()
        self.prev_close = None
        self.decomposition = None
