"""
Monte Carlo Permutation Testing (MCPT) for strategy validation.

This module implements permutation testing to validate that trading strategy
performance is statistically significant and not due to data mining or random chance.

Based on methodology from Timothy Masters' "Permutation and Randomization Tests
for Trading System Development" and the neurotrader888/mcpt repository.

Key components:
- bar_permute(): Permute time series data while preserving marginal distributions
- run_insample_mcpt(): In-sample permutation test with p-value computation
- run_walkforward_mcpt(): Walk-forward permutation test with configurable scope
"""

from volatility_arbitrage.testing.mcpt.config import MCPTConfig
from volatility_arbitrage.testing.mcpt.permutation import bar_permute
from volatility_arbitrage.testing.mcpt.insample_test import run_insample_mcpt, MCPTResult
from volatility_arbitrage.testing.mcpt.walkforward_test import (
    run_walkforward_mcpt,
    run_dual_walkforward_mcpt,
    WalkForwardMCPTResult,
    WalkForwardFold,
)

__all__ = [
    "MCPTConfig",
    "bar_permute",
    "run_insample_mcpt",
    "MCPTResult",
    "run_walkforward_mcpt",
    "run_dual_walkforward_mcpt",
    "WalkForwardMCPTResult",
    "WalkForwardFold",
]
