"""
Statistical testing and validation framework for volatility arbitrage strategies.

This module provides tools for validating that strategy performance is statistically
significant and not due to data mining or random chance.

Main components:
- mcpt: Monte Carlo Permutation Testing framework
"""

from volatility_arbitrage.testing.mcpt import (
    bar_permute,
    run_insample_mcpt,
    run_walkforward_mcpt,
    MCPTConfig,
    MCPTResult,
    WalkForwardMCPTResult,
)

__all__ = [
    "bar_permute",
    "run_insample_mcpt",
    "run_walkforward_mcpt",
    "MCPTConfig",
    "MCPTResult",
    "WalkForwardMCPTResult",
]
