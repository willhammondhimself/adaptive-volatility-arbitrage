"""
Vol-of-Vol (VVIX) Signal Enhancement.

Uses volatility-of-volatility as a position sizing modifier:

- High VVIX: Vol is itself volatile → favor trading vol (reduce short vol size)
- Low VVIX: Vol is stable → favor selling vol (increase short vol size)

VVIX is estimated from the standard deviation of daily IV changes,
annualized to match VIX scaling.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque
import numpy as np
import pandas as pd


@dataclass
class VoVConfig:
    """Configuration for vol-of-vol signal."""

    # Enable flag
    enabled: bool = True

    # VVIX thresholds (annualized)
    high_threshold: float = 0.50    # High VVIX regime
    low_threshold: float = 0.25     # Low VVIX regime

    # Position size scalars
    high_long_scalar: float = 1.3   # Boost long vol when VVIX high
    high_short_scalar: float = 0.7  # Reduce short vol when VVIX high
    low_short_scalar: float = 1.2   # Boost short vol when VVIX low
    low_long_scalar: float = 0.8    # Reduce long vol when VVIX low

    # Estimation parameters
    lookback_window: int = 20       # Days for VVIX estimation


class VoVSignalGenerator:
    """
    Generate position size scalars based on vol-of-vol (VVIX proxy).

    VVIX is estimated as:
        VVIX = std(daily_IV_changes) * sqrt(252)

    This captures how volatile the implied volatility itself is.

    Usage:
        generator = VoVSignalGenerator(config)

        # Update with new IV each day
        generator.update_iv(current_atm_iv)

        # Get position size scalar
        scalar = generator.get_position_scalar("SHORT_VOL")
    """

    def __init__(self, config: VoVConfig = None):
        self.config = config or VoVConfig()
        self.iv_history: deque = deque(maxlen=252)
        self.vov_estimate: Optional[float] = None

    def update_iv(self, atm_iv: float) -> Optional[float]:
        """
        Update with new ATM IV and recalculate VVIX estimate.

        Args:
            atm_iv: Current ATM implied volatility (annualized, e.g., 0.20 = 20%)

        Returns:
            Updated VVIX estimate or None if insufficient history
        """
        if not self.config.enabled:
            return None

        self.iv_history.append(atm_iv)

        if len(self.iv_history) < self.config.lookback_window + 1:
            return None

        # Calculate daily IV changes
        iv_series = pd.Series(list(self.iv_history))
        iv_changes = iv_series.diff().dropna()

        # VVIX = annualized std of IV changes
        self.vov_estimate = iv_changes.iloc[-self.config.lookback_window:].std() * np.sqrt(252)

        return self.vov_estimate

    def get_regime(self) -> str:
        """Classify current VVIX regime."""
        if self.vov_estimate is None:
            return "UNKNOWN"

        if self.vov_estimate > self.config.high_threshold:
            return "HIGH"
        elif self.vov_estimate < self.config.low_threshold:
            return "LOW"
        else:
            return "NORMAL"

    def get_position_scalar(self, position_type: str) -> float:
        """
        Get position size scalar based on current VVIX regime.

        Args:
            position_type: "LONG_VOL" or "SHORT_VOL"

        Returns:
            Scalar to multiply base position size (1.0 = no change)
        """
        if not self.config.enabled or self.vov_estimate is None:
            return 1.0

        regime = self.get_regime()

        if regime == "HIGH":
            # High VVIX: Vol is volatile
            if position_type == "LONG_VOL":
                return self.config.high_long_scalar
            else:
                return self.config.high_short_scalar

        elif regime == "LOW":
            # Low VVIX: Vol is stable
            if position_type == "SHORT_VOL":
                return self.config.low_short_scalar
            else:
                return self.config.low_long_scalar

        return 1.0  # NORMAL regime

    def get_statistics(self) -> dict:
        """Get VVIX statistics for logging."""
        if not self.iv_history:
            return {
                "vov_estimate": None,
                "regime": "UNKNOWN",
                "iv_history_len": 0,
            }

        iv_arr = np.array(self.iv_history)
        return {
            "vov_estimate": self.vov_estimate,
            "regime": self.get_regime(),
            "iv_history_len": len(self.iv_history),
            "iv_current": iv_arr[-1] if len(iv_arr) > 0 else None,
            "iv_mean": float(iv_arr.mean()),
            "iv_std": float(iv_arr.std()),
        }

    def reset(self) -> None:
        """Reset state for new backtest run."""
        self.iv_history.clear()
        self.vov_estimate = None
