"""
Term Structure Leverage Enhancement.

Scales position size based on term structure slope (contango/backwardation):

- Steep contango (>8%): Very favorable for short vol → up to 2.0x leverage
- Moderate contango (5-8%): Favorable → 1.25x
- Flat (-2% to +2%): Baseline → 1.0x
- Backwardation (-8% to -2%): Cautious → 0.75x
- Steep backwardation (<-8%): Minimal → 0.5x
- Extreme backwardation (<-15%): VETO trade entirely

Term structure slope = (IV_60d - IV_30d) / IV_30d
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class TermStructureRegime(Enum):
    """Term structure regime classification."""
    STEEP_CONTANGO = "STEEP_CONTANGO"
    MODERATE_CONTANGO = "MODERATE_CONTANGO"
    MILD_CONTANGO = "MILD_CONTANGO"
    FLAT = "FLAT"
    MILD_BACKWARDATION = "MILD_BACKWARDATION"
    MODERATE_BACKWARDATION = "MODERATE_BACKWARDATION"
    STEEP_BACKWARDATION = "STEEP_BACKWARDATION"
    EXTREME_BACKWARDATION = "EXTREME_BACKWARDATION"


@dataclass
class TermStructureLeverageConfig:
    """Configuration for term structure leverage."""

    # Enable flag
    enabled: bool = True

    # Contango thresholds (positive = contango)
    steep_contango_threshold: float = 0.08      # > 8%
    moderate_contango_threshold: float = 0.05   # 5-8%
    mild_contango_threshold: float = 0.02       # 2-5%

    # Backwardation thresholds (negative = backwardation)
    mild_backwardation_threshold: float = -0.02     # -2% to 0%
    moderate_backwardation_threshold: float = -0.05 # -5% to -2%
    steep_backwardation_threshold: float = -0.08    # -8% to -5%
    extreme_backwardation_threshold: float = -0.15  # < -15% = VETO

    # Leverage multipliers
    steep_contango_leverage: float = 2.0
    moderate_contango_leverage: float = 1.5
    mild_contango_leverage: float = 1.25
    flat_leverage: float = 1.0
    mild_backwardation_leverage: float = 1.0
    moderate_backwardation_leverage: float = 0.75
    steep_backwardation_leverage: float = 0.5

    # Maximum leverage cap
    max_leverage: float = 2.0
    min_leverage: float = 0.25

    # Veto on extreme backwardation
    veto_on_extreme_backwardation: bool = True


class TermStructureLeverageCalculator:
    """
    Calculate position leverage based on term structure slope.

    Contango is favorable for short vol (term premium), while
    backwardation signals market stress.

    Usage:
        calc = TermStructureLeverageCalculator(config)

        # Calculate leverage for a trade
        leverage, regime, veto = calc.get_leverage(iv_30d=0.18, iv_60d=0.20)

        if veto:
            # Block the trade
            pass
        else:
            position_size *= leverage
    """

    def __init__(self, config: TermStructureLeverageConfig = None):
        self.config = config or TermStructureLeverageConfig()

    def calculate_slope(self, iv_30d: float, iv_60d: float) -> Optional[float]:
        """
        Calculate term structure slope.

        Args:
            iv_30d: 30-day implied volatility
            iv_60d: 60-day implied volatility

        Returns:
            Slope as percentage (e.g., 0.05 = 5% contango)
        """
        if iv_30d <= 0:
            return None

        return (iv_60d - iv_30d) / iv_30d

    def classify_regime(self, slope: float) -> TermStructureRegime:
        """Classify term structure regime based on slope."""
        if slope >= self.config.steep_contango_threshold:
            return TermStructureRegime.STEEP_CONTANGO
        elif slope >= self.config.moderate_contango_threshold:
            return TermStructureRegime.MODERATE_CONTANGO
        elif slope >= self.config.mild_contango_threshold:
            return TermStructureRegime.MILD_CONTANGO
        elif slope >= self.config.mild_backwardation_threshold:
            return TermStructureRegime.FLAT
        elif slope >= self.config.moderate_backwardation_threshold:
            return TermStructureRegime.MILD_BACKWARDATION
        elif slope >= self.config.steep_backwardation_threshold:
            return TermStructureRegime.MODERATE_BACKWARDATION
        elif slope >= self.config.extreme_backwardation_threshold:
            return TermStructureRegime.STEEP_BACKWARDATION
        else:
            return TermStructureRegime.EXTREME_BACKWARDATION

    def get_leverage(
        self,
        iv_30d: float,
        iv_60d: float,
        signal_direction: str = "SHORT_VOL"
    ) -> Tuple[float, TermStructureRegime, bool, Optional[str]]:
        """
        Get leverage multiplier based on term structure.

        Args:
            iv_30d: 30-day implied volatility
            iv_60d: 60-day implied volatility
            signal_direction: "SHORT_VOL" or "LONG_VOL"

        Returns:
            (leverage, regime, is_vetoed, veto_reason)
        """
        if not self.config.enabled:
            return (1.0, TermStructureRegime.FLAT, False, None)

        slope = self.calculate_slope(iv_30d, iv_60d)
        if slope is None:
            return (1.0, TermStructureRegime.FLAT, False, None)

        regime = self.classify_regime(slope)

        # Check for veto
        if (self.config.veto_on_extreme_backwardation and
            regime == TermStructureRegime.EXTREME_BACKWARDATION and
            signal_direction == "SHORT_VOL"):
            return (
                0.0,
                regime,
                True,
                f"EXTREME_BACKWARDATION: slope={slope*100:.1f}%"
            )

        # Get base leverage
        leverage_map = {
            TermStructureRegime.STEEP_CONTANGO: self.config.steep_contango_leverage,
            TermStructureRegime.MODERATE_CONTANGO: self.config.moderate_contango_leverage,
            TermStructureRegime.MILD_CONTANGO: self.config.mild_contango_leverage,
            TermStructureRegime.FLAT: self.config.flat_leverage,
            TermStructureRegime.MILD_BACKWARDATION: self.config.mild_backwardation_leverage,
            TermStructureRegime.MODERATE_BACKWARDATION: self.config.moderate_backwardation_leverage,
            TermStructureRegime.STEEP_BACKWARDATION: self.config.steep_backwardation_leverage,
            TermStructureRegime.EXTREME_BACKWARDATION: self.config.steep_backwardation_leverage,
        }

        leverage = leverage_map.get(regime, 1.0)

        # For long vol, invert the logic (backwardation is favorable)
        if signal_direction == "LONG_VOL":
            # Backwardation = favorable for long vol
            if regime in [TermStructureRegime.STEEP_BACKWARDATION,
                         TermStructureRegime.MODERATE_BACKWARDATION]:
                leverage = max(leverage, 1.25)
            # Contango = unfavorable for long vol
            elif regime in [TermStructureRegime.STEEP_CONTANGO,
                           TermStructureRegime.MODERATE_CONTANGO]:
                leverage = min(leverage, 0.75)

        # Apply bounds
        leverage = max(self.config.min_leverage, min(leverage, self.config.max_leverage))

        return (leverage, regime, False, None)

    def get_statistics(self, iv_30d: float, iv_60d: float) -> dict:
        """Get term structure statistics for logging."""
        slope = self.calculate_slope(iv_30d, iv_60d)
        if slope is None:
            return {"slope": None, "regime": None}

        regime = self.classify_regime(slope)
        return {
            "slope": slope,
            "slope_pct": slope * 100,
            "regime": regime.value,
            "iv_30d": iv_30d,
            "iv_60d": iv_60d,
        }
