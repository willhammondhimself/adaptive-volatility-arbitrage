"""
Signal Veto Manager for Volatility Arbitrage Strategy.

Implements protective veto rules to prevent entries in dangerous market conditions.
Veto logic acts as a safety layer on top of the signal consensus model.

Key Veto Rules:
1. VIX Extreme: Block short vol when VIX > 40 (unlimited downside risk)
2. VIX Spike: Block short vol after VIX +30% in 5 days (vol clustering)
3. Extreme Backwardation: Block short vol when term structure inverted >15%
4. Signal Disagreement: No entry when 3+ signals conflict
5. Drawdown Pause: No new trades during extreme drawdowns
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import deque
from enum import Enum


class VetoReason(Enum):
    """Enumeration of veto reasons for tracking and analysis."""
    VIX_EXTREME = "VIX_EXTREME"
    VIX_SPIKE = "VIX_SPIKE"
    BACKWARDATION = "BACKWARDATION"
    SIGNAL_DISAGREEMENT = "SIGNAL_DISAGREEMENT"
    DRAWDOWN_HALT = "DRAWDOWN_HALT"
    REGIME_CRISIS = "REGIME_CRISIS"
    IV_PREMIUM_EXTREME = "IV_PREMIUM_EXTREME"


@dataclass
class VetoConfig:
    """Configuration for veto rules."""

    # VIX Extreme Rule
    vix_extreme_threshold: float = 40.0  # Block short vol when VIX > 40
    vix_extreme_enabled: bool = True

    # VIX Spike Rule
    vix_spike_threshold: float = 0.30  # 30% increase triggers veto
    vix_spike_lookback: int = 5  # Over 5 days
    vix_spike_cooldown: int = 5  # Stay out for 5 days after spike
    vix_spike_enabled: bool = True

    # Backwardation Rule (term structure inversion)
    backwardation_threshold: float = -0.15  # 15% backwardation triggers veto
    backwardation_enabled: bool = True

    # Signal Disagreement Rule
    signal_disagreement_threshold: int = 3  # Min conflicting signals
    signal_disagreement_enabled: bool = True

    # Regime Crisis Rule
    regime_crisis_percentile: float = 0.95  # Vol > 95th percentile
    regime_crisis_enabled: bool = True

    # IV Premium Extreme Rule (don't short when IV is TOO high - could go higher)
    iv_premium_extreme_threshold: float = 0.30  # IV 30%+ above RV is extreme
    iv_premium_extreme_enabled: bool = False  # Disabled by default


@dataclass
class VetoResult:
    """Result of veto check."""
    is_vetoed: bool
    reason: Optional[VetoReason] = None
    details: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_vetoed


class VetoManager:
    """
    Manages signal veto logic for volatility arbitrage entries.

    Prevents entries in dangerous market conditions by checking
    multiple veto rules before allowing trade signals.

    Usage:
        manager = VetoManager()

        # Update with daily VIX data
        manager.update_vix(current_vix)

        # Before entry, check veto
        result = manager.check_veto(
            signal_type="SHORT_VOL",
            current_vix=vix,
            term_structure=term_structure,
            signals=signals_dict,
            rv_percentile=rv_percentile,
            iv_premium=iv_premium
        )

        if result.is_vetoed:
            print(f"Trade blocked: {result.reason.value} - {result.details}")
    """

    def __init__(self, config: Optional[VetoConfig] = None):
        self.config = config or VetoConfig()

        # VIX history for spike detection
        self.vix_history: deque = deque(maxlen=30)

        # Spike cooldown tracking
        self.spike_cooldown_remaining: int = 0

        # Statistics for analysis
        self.veto_counts: dict[VetoReason, int] = {r: 0 for r in VetoReason}
        self.total_checks: int = 0
        self.total_vetoes: int = 0

    def update_vix(self, vix_value: float) -> None:
        """
        Update VIX history for spike detection.

        Args:
            vix_value: Current VIX level (or IV as proxy, scaled to VIX-like values)
        """
        self.vix_history.append(vix_value)

        # Decrement cooldown
        if self.spike_cooldown_remaining > 0:
            self.spike_cooldown_remaining -= 1

    def check_veto(
        self,
        signal_type: str,
        current_vix: float,
        term_structure: float,
        signals: dict,
        rv_percentile: float = 0.5,
        iv_premium: float = 0.0,
        is_drawdown_halted: bool = False,
    ) -> VetoResult:
        """
        Check all veto rules for a proposed entry.

        Args:
            signal_type: "SHORT_VOL" or "LONG_VOL"
            current_vix: Current VIX level (or ATM IV * 100 as proxy)
            term_structure: VIX term structure (positive = contango, negative = backwardation)
            signals: Dictionary of signal values (-1, 0, +1)
            rv_percentile: Current realized volatility percentile (0-1)
            iv_premium: IV premium over RV (e.g., 0.10 = 10% premium)
            is_drawdown_halted: Whether drawdown manager has halted trading

        Returns:
            VetoResult with is_vetoed, reason, and details
        """
        self.total_checks += 1

        # Rule 0: Drawdown Halt (highest priority)
        if is_drawdown_halted:
            return self._record_veto(
                VetoReason.DRAWDOWN_HALT,
                "Trading halted due to excessive drawdown"
            )

        # Rules for SHORT VOL entries
        if signal_type == "SHORT_VOL":
            # Rule 1: VIX Extreme
            if self.config.vix_extreme_enabled:
                if current_vix > self.config.vix_extreme_threshold:
                    return self._record_veto(
                        VetoReason.VIX_EXTREME,
                        f"VIX {current_vix:.1f} > {self.config.vix_extreme_threshold}"
                    )

            # Rule 2: VIX Spike
            if self.config.vix_spike_enabled:
                if self.spike_cooldown_remaining > 0:
                    return self._record_veto(
                        VetoReason.VIX_SPIKE,
                        f"In spike cooldown ({self.spike_cooldown_remaining} days remaining)"
                    )

                if self._check_vix_spike(current_vix):
                    self.spike_cooldown_remaining = self.config.vix_spike_cooldown
                    return self._record_veto(
                        VetoReason.VIX_SPIKE,
                        f"VIX spiked >{self.config.vix_spike_threshold*100:.0f}% in {self.config.vix_spike_lookback} days"
                    )

            # Rule 3: Extreme Backwardation
            if self.config.backwardation_enabled:
                if term_structure < self.config.backwardation_threshold:
                    return self._record_veto(
                        VetoReason.BACKWARDATION,
                        f"Term structure {term_structure*100:.1f}% (backwardation)"
                    )

            # Rule 4: Regime Crisis
            if self.config.regime_crisis_enabled:
                if rv_percentile > self.config.regime_crisis_percentile:
                    return self._record_veto(
                        VetoReason.REGIME_CRISIS,
                        f"RV at {rv_percentile*100:.0f}th percentile (crisis regime)"
                    )

            # Rule 5: IV Premium Extreme (optional)
            if self.config.iv_premium_extreme_enabled:
                if iv_premium > self.config.iv_premium_extreme_threshold:
                    return self._record_veto(
                        VetoReason.IV_PREMIUM_EXTREME,
                        f"IV premium {iv_premium*100:.1f}% too extreme for short"
                    )

        # Rules for ALL entries (both long and short)
        # Rule 6: Signal Disagreement
        if self.config.signal_disagreement_enabled:
            bullish_count = sum(1 for v in signals.values() if v > 0)
            bearish_count = sum(1 for v in signals.values() if v < 0)
            disagreement = min(bullish_count, bearish_count)

            if disagreement >= self.config.signal_disagreement_threshold:
                return self._record_veto(
                    VetoReason.SIGNAL_DISAGREEMENT,
                    f"{bullish_count} bullish vs {bearish_count} bearish signals"
                )

        # No veto triggered
        return VetoResult(is_vetoed=False)

    def _check_vix_spike(self, current_vix: float) -> bool:
        """Check if VIX has spiked above threshold over lookback period."""
        if len(self.vix_history) < self.config.vix_spike_lookback:
            return False

        # Get VIX value from lookback days ago
        lookback_vix = self.vix_history[-self.config.vix_spike_lookback]

        if lookback_vix <= 0:
            return False

        # Calculate percentage change
        vix_change = (current_vix - lookback_vix) / lookback_vix

        return vix_change > self.config.vix_spike_threshold

    def _record_veto(self, reason: VetoReason, details: str) -> VetoResult:
        """Record veto for statistics and return result."""
        self.veto_counts[reason] += 1
        self.total_vetoes += 1
        return VetoResult(is_vetoed=True, reason=reason, details=details)

    def reset(self) -> None:
        """Reset manager state (e.g., for new backtest run)."""
        self.vix_history.clear()
        self.spike_cooldown_remaining = 0
        self.veto_counts = {r: 0 for r in VetoReason}
        self.total_checks = 0
        self.total_vetoes = 0

    def get_statistics(self) -> dict:
        """Get veto statistics for analysis."""
        return {
            "total_checks": self.total_checks,
            "total_vetoes": self.total_vetoes,
            "veto_rate": self.total_vetoes / self.total_checks if self.total_checks > 0 else 0,
            "veto_breakdown": {r.value: c for r, c in self.veto_counts.items() if c > 0},
        }


def create_default_veto_manager() -> VetoManager:
    """Create veto manager with default conservative settings."""
    return VetoManager(VetoConfig())


def create_aggressive_veto_manager() -> VetoManager:
    """Create veto manager with more aggressive (relaxed) settings."""
    config = VetoConfig(
        vix_extreme_threshold=50.0,  # Higher threshold
        vix_spike_threshold=0.40,     # Allow larger spikes
        backwardation_threshold=-0.20,  # Allow more backwardation
        signal_disagreement_threshold=4,  # Allow more disagreement
    )
    return VetoManager(config)
