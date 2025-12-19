"""
Dynamic Signal Weighting (ML Ensemble) Enhancement.

Adapts signal weights based on recent accuracy using exponential moving average:

- Track each signal's accuracy (did it predict correctly?)
- Use EMA to smooth accuracy estimates (decay = 0.95)
- Map accuracy to weight: 50% accuracy = neutral, 75% = boosted, 25% = reduced
- Apply floor (5%) and ceiling (40%) constraints
- Normalize to sum = 1.0

This is the simplest adaptive approach with lowest overfitting risk.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np


@dataclass
class DynamicWeightingConfig:
    """Configuration for dynamic signal weighting."""

    # Enable flag
    enabled: bool = True

    # EMA parameters
    ema_decay: float = 0.95                 # Higher = more history weight
    min_samples_for_adaptation: int = 20    # Trades before adapting

    # Weight bounds
    min_weight: float = 0.05                # Prevent signal starvation
    max_weight: float = 0.40                # Prevent dominance

    # Base weights (used until min_samples reached)
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        "pc_ratio": 0.20,
        "iv_skew": 0.20,
        "iv_premium": 0.15,
        "term_structure": 0.15,
        "volume_spike": 0.15,
        "near_term_sentiment": 0.15,
    })


@dataclass
class SignalOutcome:
    """Record of a signal's outcome."""
    signal_name: str
    signal_value: int       # -1, 0, +1
    actual_return: float    # Actual position return
    was_correct: bool       # Did signal predict direction correctly?


class DynamicSignalWeighter:
    """
    Dynamically adjust signal weights based on recent accuracy.

    Uses exponential moving average of signal accuracy to adapt weights.
    More accurate signals get higher weights, less accurate get lower.

    Usage:
        weighter = DynamicSignalWeighter(config)

        # After a trade closes, record outcomes
        weighter.record_outcome(
            signals={"pc_ratio": 1, "iv_skew": -1, ...},
            actual_return=0.05,  # 5% profit
            position_direction=-1  # Was short vol
        )

        # Get current weights
        weights = weighter.get_weights()
    """

    def __init__(self, config: DynamicWeightingConfig = None):
        self.config = config or DynamicWeightingConfig()

        # EMA accuracy trackers per signal
        self.ema_accuracy: Dict[str, float] = {
            name: 0.5 for name in self.config.base_weights.keys()
        }

        # Trade counter
        self.trade_count: int = 0

        # Outcome history for analysis
        self.outcome_history: deque = deque(maxlen=500)

    def record_outcome(
        self,
        signals: Dict[str, int],
        actual_return: float,
        position_direction: int
    ) -> None:
        """
        Record outcome of a completed trade.

        Args:
            signals: Dict of signal names to values at entry
            actual_return: Realized return on position
            position_direction: +1 for long vol, -1 for short vol
        """
        if not self.config.enabled:
            return

        # Determine if trade was profitable
        trade_profitable = actual_return > 0

        # For each signal, determine if it was "correct"
        for signal_name, signal_value in signals.items():
            if signal_name not in self.ema_accuracy:
                continue

            if signal_value == 0:
                # Signal was neutral, skip
                continue

            # Signal was correct if:
            # - Signal positive (+1) AND position profitable
            # - Signal negative (-1) AND position unprofitable
            # Adjusted for position direction
            signal_agrees_with_position = (signal_value * position_direction) > 0
            was_correct = signal_agrees_with_position == trade_profitable

            # Update EMA accuracy
            old_acc = self.ema_accuracy[signal_name]
            new_acc = (
                self.config.ema_decay * old_acc +
                (1 - self.config.ema_decay) * (1.0 if was_correct else 0.0)
            )
            self.ema_accuracy[signal_name] = new_acc

            # Record for history
            self.outcome_history.append(SignalOutcome(
                signal_name=signal_name,
                signal_value=signal_value,
                actual_return=actual_return,
                was_correct=was_correct,
            ))

        self.trade_count += 1

    def _accuracy_to_weight(self, accuracy: float) -> float:
        """
        Map accuracy to raw weight.

        Linear mapping:
        - 50% accuracy → 1.0x weight (neutral)
        - 75% accuracy → 1.5x weight (boosted)
        - 25% accuracy → 0.5x weight (reduced)
        """
        # accuracy of 0.5 = multiplier of 1.0
        # accuracy of 0.75 = multiplier of 1.5
        # accuracy of 0.25 = multiplier of 0.5
        multiplier = 0.5 + accuracy  # Range: 0.5 to 1.5

        return multiplier

    def get_weights(self) -> Dict[str, float]:
        """
        Get current adaptive weights.

        Returns base weights if not enough samples, otherwise returns
        adapted weights normalized to sum to 1.0.
        """
        if not self.config.enabled:
            return dict(self.config.base_weights)

        # Use base weights until enough samples
        if self.trade_count < self.config.min_samples_for_adaptation:
            return dict(self.config.base_weights)

        # Calculate raw weights from accuracy
        raw_weights = {}
        for name, base_weight in self.config.base_weights.items():
            accuracy = self.ema_accuracy.get(name, 0.5)
            multiplier = self._accuracy_to_weight(accuracy)
            raw_weights[name] = base_weight * multiplier

        # Apply bounds
        bounded_weights = {}
        for name, weight in raw_weights.items():
            bounded = max(self.config.min_weight, min(weight, self.config.max_weight))
            bounded_weights[name] = bounded

        # Normalize to sum to 1.0
        total = sum(bounded_weights.values())
        if total > 0:
            normalized_weights = {k: v / total for k, v in bounded_weights.items()}
        else:
            normalized_weights = dict(self.config.base_weights)

        return normalized_weights

    def get_accuracy_stats(self) -> Dict[str, float]:
        """Get current accuracy estimates for all signals."""
        return dict(self.ema_accuracy)

    def get_statistics(self) -> dict:
        """Get comprehensive statistics for logging."""
        weights = self.get_weights()
        return {
            "trade_count": self.trade_count,
            "is_adapting": self.trade_count >= self.config.min_samples_for_adaptation,
            "ema_accuracy": dict(self.ema_accuracy),
            "current_weights": weights,
            "base_weights": dict(self.config.base_weights),
            "outcome_history_len": len(self.outcome_history),
        }

    def reset(self) -> None:
        """Reset state for new backtest run."""
        self.ema_accuracy = {
            name: 0.5 for name in self.config.base_weights.keys()
        }
        self.trade_count = 0
        self.outcome_history.clear()
