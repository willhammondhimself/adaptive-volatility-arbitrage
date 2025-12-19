"""
Regime Transition Signal Enhancement.

Detects vol regime TRANSITIONS (not just levels) to generate
momentum/mean-reversion signals:

- LOW → NORMAL: Long vol signal (momentum into vol expansion)
- HIGH → NORMAL: Short vol signal (mean-reversion after spike)

This captures the *velocity* of vol changes, not just the *level*.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque
import numpy as np


@dataclass
class RegimeTransitionConfig:
    """Configuration for regime transition signals."""

    # Enable flag
    enabled: bool = True

    # Regime boundaries (percentile)
    low_regime_boundary: float = 0.30    # Below 30th percentile = low vol
    high_regime_boundary: float = 0.70   # Above 70th percentile = high vol

    # Transition threshold (percentile points)
    transition_threshold: float = 0.10   # 10 percentile point change

    # Signal weight in consensus
    signal_weight: float = 0.15          # Weight when adding to 6-signal consensus


class RegimeTransitionSignal:
    """
    Generate signals based on regime transitions.

    Tracks RV percentile over time and detects significant transitions
    between low/normal/high vol regimes.

    Usage:
        signal = RegimeTransitionSignal(config)

        # Update with new data each day
        signal_value = signal.update(rv_percentile=0.45)

        # signal_value is:
        #   +1: LOW→NORMAL transition (long vol)
        #   -1: HIGH→NORMAL transition (short vol)
        #    0: No transition or within regime
    """

    def __init__(self, config: RegimeTransitionConfig = None):
        self.config = config or RegimeTransitionConfig()
        self.prev_percentile: Optional[float] = None
        self.prev_regime: Optional[str] = None
        self.percentile_history: deque = deque(maxlen=252)

    def _classify_regime(self, percentile: float) -> str:
        """Classify current regime based on percentile."""
        if percentile < self.config.low_regime_boundary:
            return "LOW"
        elif percentile > self.config.high_regime_boundary:
            return "HIGH"
        else:
            return "NORMAL"

    def update(self, rv_percentile: float) -> int:
        """
        Update with new RV percentile and return transition signal.

        Args:
            rv_percentile: Current RV percentile (0.0 to 1.0)

        Returns:
            Signal: +1 (long vol), -1 (short vol), 0 (neutral)
        """
        if not self.config.enabled:
            return 0

        self.percentile_history.append(rv_percentile)
        current_regime = self._classify_regime(rv_percentile)

        signal = 0

        if self.prev_percentile is not None and self.prev_regime is not None:
            # LOW → NORMAL: Vol is expanding, momentum long vol
            # Signal fires when crossing from LOW regime into NORMAL
            if self.prev_regime == "LOW" and current_regime == "NORMAL":
                signal = -1  # Long vol opportunity (vol expanding)

            # HIGH → NORMAL: Vol is contracting, mean-reversion short vol
            # Signal fires when crossing from HIGH regime into NORMAL
            elif self.prev_regime == "HIGH" and current_regime == "NORMAL":
                signal = 1  # Short vol opportunity (vol contracting)

        # Update state
        self.prev_percentile = rv_percentile
        self.prev_regime = current_regime

        return signal

    def get_transition_info(self) -> dict:
        """Get current transition state for logging."""
        return {
            "prev_percentile": self.prev_percentile,
            "prev_regime": self.prev_regime,
            "low_boundary": self.config.low_regime_boundary,
            "high_boundary": self.config.high_regime_boundary,
            "threshold": self.config.transition_threshold,
        }

    def reset(self) -> None:
        """Reset state for new backtest run."""
        self.prev_percentile = None
        self.prev_regime = None
        self.percentile_history.clear()


def integrate_regime_signal(
    base_signals: dict,
    base_weights: dict,
    regime_signal: int,
    regime_weight: float = 0.15
) -> Tuple[dict, dict]:
    """
    Integrate regime transition signal into existing 6-signal consensus.

    Adjusts existing weights proportionally to accommodate the new signal.

    Args:
        base_signals: Dict of signal names to values (-1, 0, +1)
        base_weights: Dict of signal names to weights (sum to 1.0)
        regime_signal: Regime transition signal value
        regime_weight: Weight for regime signal

    Returns:
        (updated_signals, updated_weights) with 7th signal added
    """
    if regime_signal == 0:
        # No transition, return original
        return base_signals, base_weights

    # Scale down existing weights to make room for regime signal
    scale_factor = 1.0 - regime_weight
    updated_weights = {k: v * scale_factor for k, v in base_weights.items()}

    # Add regime signal
    updated_signals = dict(base_signals)
    updated_signals["regime_transition"] = regime_signal
    updated_weights["regime_transition"] = regime_weight

    return updated_signals, updated_weights
