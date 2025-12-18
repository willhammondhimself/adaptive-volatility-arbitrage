"""
Kelly Criterion Position Sizing for Volatility Arbitrage Strategy.

Implements optimal position sizing based on the Kelly criterion with
safety adjustments for real-world trading:

1. Fractional Kelly (default 0.25x) to reduce variance
2. Rolling statistics from recent trades
3. Drawdown adjustment for capital preservation
4. Min/max position size bounds

The Kelly criterion gives the optimal bet size that maximizes
long-term geometric growth rate:

    f* = (p * b - q) / b

Where:
    f* = optimal fraction of capital
    p = probability of winning
    q = probability of losing (1 - p)
    b = win/loss ratio (average win / average loss)
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import numpy as np


@dataclass
class KellyConfig:
    """Configuration for Kelly position sizing."""

    # Fractional Kelly multiplier (1.0 = full Kelly, 0.25 = quarter Kelly)
    kelly_fraction: float = 0.25

    # Rolling window for win rate / win-loss ratio calculation
    lookback_trades: int = 50

    # Minimum trades needed before using Kelly (use default until then)
    min_trades_for_kelly: int = 20

    # Default position size before enough history
    default_position_pct: float = 0.05  # 5%

    # Position size bounds
    min_position_pct: float = 0.01  # Never less than 1%
    max_position_pct: float = 0.15  # Never more than 15%

    # Drawdown adjustment: reduce Kelly output during drawdowns
    drawdown_adjustment_enabled: bool = True
    drawdown_full_reduction: float = 0.10  # 10% DD = no Kelly adjustment
    drawdown_scalar_at_full: float = 0.5  # At 10% DD, use 50% of Kelly

    # Volatility adjustment: reduce size in high-vol regimes
    vol_adjustment_enabled: bool = True
    high_vol_percentile: float = 0.80  # Vol > 80th pctile
    high_vol_scalar: float = 0.75  # Reduce to 75% in high vol


@dataclass
class TradeRecord:
    """Record of a completed trade for Kelly calculation."""
    pnl: float
    entry_size: float  # Position size at entry
    is_win: bool = field(init=False)
    return_pct: float = field(init=False)

    def __post_init__(self):
        self.is_win = self.pnl > 0
        self.return_pct = self.pnl / self.entry_size if self.entry_size > 0 else 0


class KellyPositionSizer:
    """
    Kelly criterion-based position sizing with safety adjustments.

    Calculates optimal position size based on historical win rate
    and win/loss ratio, with fractional Kelly for reduced variance.

    Usage:
        sizer = KellyPositionSizer()

        # After each trade completes, record it
        sizer.add_trade(pnl=500, entry_size=10000)

        # Get position size for next trade
        kelly_pct = sizer.calculate_kelly()
        position_size = capital * kelly_pct

        # With drawdown and vol adjustments
        adjusted_pct = sizer.get_position_size(
            current_drawdown=0.05,
            vol_percentile=0.70
        )
    """

    def __init__(self, config: Optional[KellyConfig] = None):
        self.config = config or KellyConfig()
        self.trade_history: deque[TradeRecord] = deque(maxlen=self.config.lookback_trades)

        # Statistics cache
        self._cached_kelly: Optional[float] = None
        self._cache_dirty: bool = True

    def add_trade(self, pnl: float, entry_size: float) -> None:
        """
        Record a completed trade for Kelly calculation.

        Args:
            pnl: Profit/loss from the trade (absolute $)
            entry_size: Position size at entry (absolute $)
        """
        record = TradeRecord(pnl=pnl, entry_size=entry_size)
        self.trade_history.append(record)
        self._cache_dirty = True

    def calculate_kelly(self) -> float:
        """
        Calculate Kelly optimal position size.

        Returns:
            Optimal position as fraction of capital (e.g., 0.05 = 5%)
        """
        if self._cached_kelly is not None and not self._cache_dirty:
            return self._cached_kelly

        # Not enough history - use default
        if len(self.trade_history) < self.config.min_trades_for_kelly:
            return self.config.default_position_pct

        trades = list(self.trade_history)
        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]

        # Edge case: all wins or all losses
        if not wins or not losses:
            self._cached_kelly = self.config.default_position_pct
            self._cache_dirty = False
            return self._cached_kelly

        # Win probability
        p = len(wins) / len(trades)
        q = 1 - p

        # Win/loss ratio (average win / average loss magnitude)
        avg_win = np.mean([t.return_pct for t in wins])
        avg_loss = abs(np.mean([t.return_pct for t in losses]))

        if avg_loss <= 0:
            self._cached_kelly = self.config.default_position_pct
            self._cache_dirty = False
            return self._cached_kelly

        b = avg_win / avg_loss

        # Kelly formula: f* = (p * b - q) / b
        kelly_full = (p * b - q) / b

        # Apply fractional Kelly
        kelly_fractional = kelly_full * self.config.kelly_fraction

        # Clamp to bounds
        kelly_bounded = max(
            self.config.min_position_pct,
            min(kelly_fractional, self.config.max_position_pct)
        )

        # Handle negative Kelly (negative edge) - use minimum
        if kelly_bounded < 0:
            kelly_bounded = self.config.min_position_pct

        self._cached_kelly = kelly_bounded
        self._cache_dirty = False
        return kelly_bounded

    def get_position_size(
        self,
        current_drawdown: float = 0.0,
        vol_percentile: float = 0.5,
    ) -> float:
        """
        Get adjusted position size with drawdown and volatility adjustments.

        Args:
            current_drawdown: Current portfolio drawdown (0.05 = 5% DD)
            vol_percentile: Current vol regime percentile (0-1)

        Returns:
            Adjusted position size as fraction of capital
        """
        base_kelly = self.calculate_kelly()

        # Drawdown adjustment
        if self.config.drawdown_adjustment_enabled and current_drawdown > 0:
            dd_scalar = self._calculate_drawdown_scalar(current_drawdown)
            base_kelly *= dd_scalar

        # Volatility regime adjustment
        if self.config.vol_adjustment_enabled:
            if vol_percentile > self.config.high_vol_percentile:
                base_kelly *= self.config.high_vol_scalar

        # Final bounds check
        return max(
            self.config.min_position_pct,
            min(base_kelly, self.config.max_position_pct)
        )

    def _calculate_drawdown_scalar(self, drawdown: float) -> float:
        """Calculate position scalar based on drawdown level."""
        if drawdown <= 0:
            return 1.0

        if drawdown >= self.config.drawdown_full_reduction:
            return self.config.drawdown_scalar_at_full

        # Linear interpolation
        t = drawdown / self.config.drawdown_full_reduction
        return 1.0 - t * (1.0 - self.config.drawdown_scalar_at_full)

    def get_statistics(self) -> dict:
        """Get Kelly calculation statistics."""
        if len(self.trade_history) == 0:
            return {
                "trades_recorded": 0,
                "win_rate": 0,
                "avg_win_pct": 0,
                "avg_loss_pct": 0,
                "kelly_full": 0,
                "kelly_fractional": self.config.default_position_pct,
            }

        trades = list(self.trade_history)
        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t.return_pct for t in wins]) * 100 if wins else 0
        avg_loss = np.mean([t.return_pct for t in losses]) * 100 if losses else 0

        # Calculate full Kelly for reference
        kelly_full = 0
        if wins and losses:
            p = win_rate
            q = 1 - p
            b = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            if b > 0:
                kelly_full = (p * b - q) / b

        return {
            "trades_recorded": len(trades),
            "win_rate": win_rate * 100,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "win_loss_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            "kelly_full": kelly_full * 100,
            "kelly_fractional": self.calculate_kelly() * 100,
        }

    def reset(self) -> None:
        """Reset sizer state (e.g., for new backtest run)."""
        self.trade_history.clear()
        self._cached_kelly = None
        self._cache_dirty = True


class ConsensusScaledSizer:
    """
    Position sizing that scales with signal consensus strength.

    Combines Kelly sizing with consensus-based scaling:
    - Higher consensus = larger position (up to Kelly maximum)
    - Lower consensus = smaller position
    """

    def __init__(
        self,
        kelly_sizer: Optional[KellyPositionSizer] = None,
        min_consensus: float = 0.10,
        max_consensus: float = 0.50,
        scaling_method: str = "quadratic",  # linear, quadratic, cubic
    ):
        self.kelly_sizer = kelly_sizer or KellyPositionSizer()
        self.min_consensus = min_consensus
        self.max_consensus = max_consensus
        self.scaling_method = scaling_method

    def get_position_size(
        self,
        consensus: float,
        current_drawdown: float = 0.0,
        vol_percentile: float = 0.5,
    ) -> float:
        """
        Get position size scaled by consensus strength.

        Args:
            consensus: Absolute consensus score (0-1)
            current_drawdown: Current drawdown (0-1)
            vol_percentile: Current vol percentile (0-1)

        Returns:
            Position size as fraction of capital
        """
        # Get Kelly-based maximum
        kelly_max = self.kelly_sizer.get_position_size(
            current_drawdown=current_drawdown,
            vol_percentile=vol_percentile
        )

        # Normalize consensus to 0-1 range
        consensus_abs = abs(consensus)
        consensus_normalized = (consensus_abs - self.min_consensus) / \
                               (self.max_consensus - self.min_consensus)
        consensus_normalized = max(0, min(1, consensus_normalized))

        # Apply scaling method
        if self.scaling_method == "linear":
            scale_factor = consensus_normalized
        elif self.scaling_method == "quadratic":
            scale_factor = consensus_normalized ** 2
        elif self.scaling_method == "cubic":
            scale_factor = consensus_normalized ** 3
        else:
            scale_factor = consensus_normalized

        return kelly_max * scale_factor
