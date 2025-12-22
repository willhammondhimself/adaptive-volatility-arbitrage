"""
Delta-hedged backtesting module.

Provides P&L attribution to demonstrate that strategy alpha comes from
volatility (Vega/Gamma) rather than directional exposure (Delta).
"""

from .types import HedgeConfig, HedgeState, PnLAttribution, RebalanceFrequency
from .attribution import (
    calculate_delta_pnl,
    calculate_gamma_pnl,
    calculate_vega_pnl,
    calculate_theta_pnl,
)
from .hedger import DeltaHedger
from .engine import DeltaHedgedBacktest

__all__ = [
    "HedgeConfig",
    "HedgeState",
    "PnLAttribution",
    "RebalanceFrequency",
    "calculate_delta_pnl",
    "calculate_gamma_pnl",
    "calculate_vega_pnl",
    "calculate_theta_pnl",
    "DeltaHedger",
    "DeltaHedgedBacktest",
]
