"""
Asymmetric Profit Taking Enhancement.

Different profit/loss targets for long vs short vol positions based on
the asymmetric nature of volatility returns:

- Short vol: Frequent small profits, occasional large losses (negative skew)
  → Take profits quickly, allow wider stop loss

- Long vol: Infrequent large profits, frequent small losses (positive skew)
  → Let winners run (fat tails), cut losers quickly
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AsymmetricProfitConfig:
    """Configuration for asymmetric profit taking."""

    # Short vol targets (mean-reversion, smaller targets)
    short_vol_profit_target: float = 0.08   # +8% profit target
    short_vol_stop_loss: float = -0.15      # -15% stop loss

    # Long vol targets (fat tail capture)
    long_vol_profit_target: float = 0.20    # +20% profit target
    long_vol_stop_loss: float = -0.08       # -8% stop loss

    # Enable flag
    enabled: bool = True


class AsymmetricProfitManager:
    """
    Manages asymmetric profit targets for long/short vol positions.

    Usage:
        manager = AsymmetricProfitManager(config)

        # Get targets for a position
        target, stop = manager.get_targets(position_type="SHORT_VOL")

        # Check exit conditions
        should_exit, reason = manager.check_exit(
            position_type="SHORT_VOL",
            position_return=0.09
        )
    """

    def __init__(self, config: AsymmetricProfitConfig = None):
        self.config = config or AsymmetricProfitConfig()

    def get_targets(self, position_type: str) -> Tuple[float, float]:
        """
        Get profit target and stop loss for position type.

        Args:
            position_type: "LONG_VOL" or "SHORT_VOL"

        Returns:
            (profit_target, stop_loss) tuple
        """
        if not self.config.enabled:
            # Symmetric fallback
            return (0.10, -0.10)

        if position_type == "SHORT_VOL":
            return (
                self.config.short_vol_profit_target,
                self.config.short_vol_stop_loss
            )
        elif position_type == "LONG_VOL":
            return (
                self.config.long_vol_profit_target,
                self.config.long_vol_stop_loss
            )
        else:
            raise ValueError(f"Unknown position type: {position_type}")

    def check_exit(
        self,
        position_type: str,
        position_return: float
    ) -> Tuple[bool, str]:
        """
        Check if position should exit based on P&L.

        Args:
            position_type: "LONG_VOL" or "SHORT_VOL"
            position_return: Current return on position (e.g., 0.05 = 5%)

        Returns:
            (should_exit, exit_reason) tuple
        """
        profit_target, stop_loss = self.get_targets(position_type)

        if position_return >= profit_target:
            return (True, "PROFIT_TARGET")
        elif position_return <= stop_loss:
            return (True, "STOP_LOSS")

        return (False, "")

    def get_config_dict(self) -> dict:
        """Get configuration as dictionary for logging."""
        return {
            "enabled": self.config.enabled,
            "short_vol_profit_target": self.config.short_vol_profit_target,
            "short_vol_stop_loss": self.config.short_vol_stop_loss,
            "long_vol_profit_target": self.config.long_vol_profit_target,
            "long_vol_stop_loss": self.config.long_vol_stop_loss,
        }
