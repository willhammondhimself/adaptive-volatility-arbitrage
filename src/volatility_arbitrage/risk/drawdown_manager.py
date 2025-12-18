"""
Drawdown Recovery Manager for Volatility Arbitrage Strategy.

Implements dynamic position scaling based on current drawdown level.
When equity falls below peak, position sizes are reduced to preserve capital
and enable faster recovery.

Key Features:
- Tiered drawdown thresholds with configurable scalars
- Smooth transitions between tiers (optional linear interpolation)
- Complete trading halt at extreme drawdowns
- Recovery detection for position size restoration
"""

from dataclasses import dataclass, field
from typing import Optional
from decimal import Decimal


@dataclass
class DrawdownConfig:
    """Configuration for drawdown-based position scaling."""

    # Tiered thresholds: list of (drawdown_pct, position_scalar) tuples
    # Drawdown is expressed as positive percentage (e.g., 0.05 = 5% drawdown)
    thresholds: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.02, 1.00),   # <2% DD: full size
        (0.04, 0.75),   # 2-4% DD: 75% size
        (0.06, 0.50),   # 4-6% DD: 50% size
        (0.08, 0.25),   # 6-8% DD: 25% size
        (0.10, 0.10),   # 8-10% DD: minimal trading
    ])

    # Halt all trading if drawdown exceeds this level
    halt_threshold: float = 0.12  # 12% drawdown = stop trading

    # Use linear interpolation between thresholds (smoother transitions)
    use_interpolation: bool = False

    # Minimum position size (floor even in worst drawdowns)
    min_position_scalar: float = 0.05

    # Recovery threshold: restore normal sizing when DD falls below this
    recovery_threshold: float = 0.01  # Must recover to <1% DD for full size


class DrawdownManager:
    """
    Dynamic position scaling based on drawdown level.

    Reduces position sizes during drawdowns to:
    1. Preserve remaining capital
    2. Reduce risk of catastrophic losses
    3. Enable faster recovery with smaller positions

    Usage:
        manager = DrawdownManager()

        # Each day, update with current equity
        scalar = manager.get_position_scalar(current_equity)

        # Apply to position sizing
        position_size = base_position_size * scalar
    """

    def __init__(self, config: Optional[DrawdownConfig] = None):
        self.config = config or DrawdownConfig()
        self.peak_equity: Optional[float] = None
        self.current_drawdown: float = 0.0
        self.is_halted: bool = False

        # Track state for logging/debugging
        self.last_scalar: float = 1.0
        self.days_in_drawdown: int = 0
        self.max_drawdown_seen: float = 0.0

        # Sort thresholds to ensure proper lookup
        self._thresholds = sorted(self.config.thresholds, key=lambda x: x[0])

    def update(self, equity: float) -> float:
        """
        Update drawdown state with new equity value.

        Args:
            equity: Current portfolio equity value

        Returns:
            Position size scalar (0.0 to 1.0)
        """
        # Initialize peak on first call
        if self.peak_equity is None:
            self.peak_equity = equity

        # Update peak if we have new high
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.days_in_drawdown = 0
        else:
            self.days_in_drawdown += 1

        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0

        # Track maximum drawdown
        self.max_drawdown_seen = max(self.max_drawdown_seen, self.current_drawdown)

        # Check halt condition
        if self.current_drawdown >= self.config.halt_threshold:
            self.is_halted = True
            self.last_scalar = 0.0
            return 0.0

        # Resume trading if recovered
        if self.is_halted and self.current_drawdown <= self.config.recovery_threshold:
            self.is_halted = False

        if self.is_halted:
            self.last_scalar = 0.0
            return 0.0

        # Get position scalar based on drawdown level
        scalar = self._calculate_scalar()
        self.last_scalar = scalar
        return scalar

    def get_position_scalar(self, equity: float) -> float:
        """
        Convenience method - alias for update().

        Args:
            equity: Current portfolio equity value

        Returns:
            Position size scalar (0.0 to 1.0)
        """
        return self.update(equity)

    def _calculate_scalar(self) -> float:
        """Calculate position scalar based on current drawdown."""
        dd = self.current_drawdown

        # No drawdown - full size
        if dd <= 0:
            return 1.0

        # Below first threshold - full size
        if dd < self._thresholds[0][0]:
            return 1.0

        # Beyond last threshold - minimum size
        if dd >= self._thresholds[-1][0]:
            return max(self.config.min_position_scalar, self._thresholds[-1][1])

        # Find applicable threshold tier
        if self.config.use_interpolation:
            return self._interpolate_scalar(dd)
        else:
            return self._step_scalar(dd)

    def _step_scalar(self, dd: float) -> float:
        """Step function - use scalar from highest breached threshold."""
        scalar = 1.0
        for threshold, tier_scalar in self._thresholds:
            if dd >= threshold:
                scalar = tier_scalar
            else:
                break
        return max(self.config.min_position_scalar, scalar)

    def _interpolate_scalar(self, dd: float) -> float:
        """Linear interpolation between thresholds for smoother transitions."""
        # Find the two thresholds we're between
        lower_thresh, lower_scalar = 0.0, 1.0
        upper_thresh, upper_scalar = self._thresholds[-1]

        for i, (thresh, scalar) in enumerate(self._thresholds):
            if dd < thresh:
                upper_thresh, upper_scalar = thresh, scalar
                if i > 0:
                    lower_thresh, lower_scalar = self._thresholds[i - 1]
                break
            lower_thresh, lower_scalar = thresh, scalar

        # Linear interpolation
        if upper_thresh == lower_thresh:
            return lower_scalar

        t = (dd - lower_thresh) / (upper_thresh - lower_thresh)
        interpolated = lower_scalar + t * (upper_scalar - lower_scalar)

        return max(self.config.min_position_scalar, interpolated)

    def reset(self) -> None:
        """Reset manager state (e.g., for new backtest run)."""
        self.peak_equity = None
        self.current_drawdown = 0.0
        self.is_halted = False
        self.last_scalar = 1.0
        self.days_in_drawdown = 0
        self.max_drawdown_seen = 0.0

    def get_status(self) -> dict:
        """Get current drawdown manager status for logging/debugging."""
        return {
            "peak_equity": self.peak_equity,
            "current_drawdown_pct": self.current_drawdown * 100,
            "max_drawdown_pct": self.max_drawdown_seen * 100,
            "position_scalar": self.last_scalar,
            "is_halted": self.is_halted,
            "days_in_drawdown": self.days_in_drawdown,
        }

    def should_allow_new_trade(self) -> bool:
        """Check if new trades are allowed based on current drawdown state."""
        return not self.is_halted and self.last_scalar > 0

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as percentage."""
        return self.current_drawdown * 100
