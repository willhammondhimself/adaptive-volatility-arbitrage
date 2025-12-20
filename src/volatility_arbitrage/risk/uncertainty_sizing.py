"""
Uncertainty-Adjusted Position Sizing.

Extends Kelly criterion with a confidence scalar based on model uncertainty.
Higher epistemic uncertainty → smaller position size.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class UncertaintySizingConfig:
    """Configuration for uncertainty-adjusted sizing."""

    kelly_fraction: float = 0.25
    """Fractional Kelly multiplier (0.25 = quarter Kelly)."""

    uncertainty_penalty: float = 2.0
    """
    Penalty coefficient for uncertainty.
    Higher values = more aggressive size reduction when uncertain.
    """

    min_position_pct: float = 0.01
    """Minimum position size as fraction of capital."""

    max_position_pct: float = 0.15
    """Maximum position size as fraction of capital."""


def size_position_with_uncertainty(
    signal_strength: float,
    uncertainty: float,
    capital: float,
    kelly_fraction: float = 0.25,
    uncertainty_penalty: float = 2.0,
    max_position_pct: float = 0.15,
    min_position_pct: float = 0.01,
) -> float:
    """
    Calculate position size with uncertainty adjustment.

    Computes a base Kelly-style size, then scales it down based on
    model uncertainty. High uncertainty → smaller position.

    Formula:
        base_size = kelly_fraction * capital * abs(signal_strength)
        confidence_scalar = 1.0 / (1.0 + uncertainty_penalty * uncertainty)
        position = base_size * confidence_scalar

    Args:
        signal_strength: Signal magnitude (e.g., z-score, 0-1 normalized)
        uncertainty: Epistemic uncertainty from model (e.g., std of MC samples)
        capital: Total capital available
        kelly_fraction: Fractional Kelly multiplier
        uncertainty_penalty: Coefficient for uncertainty discount
        max_position_pct: Maximum position as fraction of capital
        min_position_pct: Minimum position as fraction of capital

    Returns:
        Position size in currency units
    """
    if capital <= 0:
        return 0.0

    # Base size from Kelly-style calculation
    base_size = kelly_fraction * capital * abs(signal_strength)

    # Confidence scalar: discount for uncertainty
    # When uncertainty=0, scalar=1.0 (full size)
    # When uncertainty is high, scalar approaches 0
    confidence_scalar = 1.0 / (1.0 + uncertainty_penalty * uncertainty)

    # Apply uncertainty discount
    adjusted_size = base_size * confidence_scalar

    # Apply bounds
    max_size = capital * max_position_pct
    min_size = capital * min_position_pct

    return max(min_size, min(adjusted_size, max_size))


class UncertaintySizer:
    """
    Stateful uncertainty-adjusted position sizer.

    Wraps the sizing function with configuration for repeated use.
    """

    def __init__(self, config: Optional[UncertaintySizingConfig] = None) -> None:
        self.config = config or UncertaintySizingConfig()

    def calculate_size(
        self,
        signal_strength: float,
        uncertainty: float,
        capital: float,
    ) -> float:
        """
        Calculate position size with uncertainty adjustment.

        Args:
            signal_strength: Signal magnitude
            uncertainty: Model epistemic uncertainty
            capital: Available capital

        Returns:
            Position size in currency units
        """
        return size_position_with_uncertainty(
            signal_strength=signal_strength,
            uncertainty=uncertainty,
            capital=capital,
            kelly_fraction=self.config.kelly_fraction,
            uncertainty_penalty=self.config.uncertainty_penalty,
            max_position_pct=self.config.max_position_pct,
            min_position_pct=self.config.min_position_pct,
        )

    def calculate_size_pct(
        self,
        signal_strength: float,
        uncertainty: float,
    ) -> float:
        """
        Calculate position size as percentage of capital.

        Args:
            signal_strength: Signal magnitude
            uncertainty: Model epistemic uncertainty

        Returns:
            Position size as fraction of capital (0-1)
        """
        # Use capital=1.0 to get percentage directly
        return size_position_with_uncertainty(
            signal_strength=signal_strength,
            uncertainty=uncertainty,
            capital=1.0,
            kelly_fraction=self.config.kelly_fraction,
            uncertainty_penalty=self.config.uncertainty_penalty,
            max_position_pct=self.config.max_position_pct,
            min_position_pct=self.config.min_position_pct,
        )
