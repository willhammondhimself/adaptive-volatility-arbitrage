"""
Types for delta-hedged backtesting.

Dataclasses for hedge state, P&L attribution, and configuration.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from volatility_arbitrage.execution.costs import TransactionCostModel


class RebalanceFrequency(Enum):
    """Frequency at which to rebalance the delta hedge."""

    CONTINUOUS = "continuous"  # Every tick
    HOURLY = "hourly"
    FOUR_HOUR = "4h"
    DAILY = "daily"


@dataclass(frozen=True)
class HedgeState:
    """
    Snapshot of portfolio state at a point in time.

    Tracks option position, hedge shares, and all Greeks.
    """

    timestamp: datetime
    spot: Decimal
    iv: Decimal
    option_position: Decimal
    option_price: Decimal
    hedge_shares: Decimal
    portfolio_delta: Decimal
    portfolio_gamma: Decimal
    portfolio_vega: Decimal
    portfolio_theta: Decimal

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "spot": float(self.spot),
            "iv": float(self.iv),
            "option_position": float(self.option_position),
            "option_price": float(self.option_price),
            "hedge_shares": float(self.hedge_shares),
            "portfolio_delta": float(self.portfolio_delta),
            "portfolio_gamma": float(self.portfolio_gamma),
            "portfolio_vega": float(self.portfolio_vega),
            "portfolio_theta": float(self.portfolio_theta),
        }


@dataclass(frozen=True)
class PnLAttribution:
    """
    P&L decomposition using Taylor expansion.

    dV = Δ·dS + ½Γ·(dS)² + ν·dσ + θ·dt + residual

    For a delta-hedged portfolio (long option, short Δ shares):
    - delta_pnl should be ~0 (hedged away)
    - gamma_pnl captures convexity gains from realized volatility
    - vega_pnl captures gains/losses from IV changes
    - theta_pnl is time decay
    - transaction_costs is the cost of rebalancing
    """

    timestamp: datetime
    total_pnl: Decimal
    delta_pnl: Decimal
    gamma_pnl: Decimal
    vega_pnl: Decimal
    theta_pnl: Decimal
    transaction_costs: Decimal
    residual: Decimal
    rebalanced: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_pnl": float(self.total_pnl),
            "delta_pnl": float(self.delta_pnl),
            "gamma_pnl": float(self.gamma_pnl),
            "vega_pnl": float(self.vega_pnl),
            "theta_pnl": float(self.theta_pnl),
            "transaction_costs": float(self.transaction_costs),
            "residual": float(self.residual),
            "rebalanced": self.rebalanced,
        }


@dataclass
class HedgeConfig:
    """
    Configuration for delta hedging.

    Controls rebalancing frequency and threshold.
    """

    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    delta_threshold: Decimal = Decimal("0.10")
    """Rebalance when |portfolio_delta| exceeds this threshold."""

    cost_model: Optional[TransactionCostModel] = None
    """Transaction cost model for calculating rebalancing costs."""

    daily_volume: float = 50_000_000.0
    """Average daily volume for cost model (SPY ~50M shares/day)."""

    option_multiplier: int = 100
    """Contract multiplier for options (standard is 100)."""
