"""Monte Carlo simulation service for backtest trade analysis."""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for volatility_arbitrage imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from volatility_arbitrage.analysis.monte_carlo import block_bootstrap_resample

from backend.schemas.backtest import (
    MonteCarloRequest,
    MonteCarloResponse,
    MonteCarloMetrics,
    RiskAssessment,
)


class MonteCarloService:
    """Service for running Monte Carlo simulations on trade returns."""

    MAX_DISTRIBUTION_POINTS = 500  # Downsample for API response

    def run(
        self,
        request: MonteCarloRequest,
        observed_return: float | None = None,
        observed_sharpe: float | None = None,
        observed_drawdown: float | None = None,
    ) -> MonteCarloResponse:
        """
        Run block bootstrap Monte Carlo simulation.

        Args:
            request: MC configuration with trade returns
            observed_return: Actual backtest total return (for histogram marker)
            observed_sharpe: Actual backtest Sharpe ratio
            observed_drawdown: Actual backtest max drawdown

        Returns:
            MonteCarloResponse with distributions and confidence intervals
        """
        start_time = time.time()

        trade_returns = np.array(request.trade_returns)

        result = block_bootstrap_resample(
            trade_returns=trade_returns,
            n_simulations=request.n_simulations,
            block_size=request.block_size,
            initial_capital=request.initial_capital,
            random_seed=request.random_seed,
            winsorize_pct=request.winsorize_pct,
        )

        computation_time_ms = (time.time() - start_time) * 1000

        return MonteCarloResponse(
            n_simulations=result.n_simulations,
            n_trades=result.n_trades,
            total_return=MonteCarloMetrics(
                mean=float(result.return_mean),
                std=float(result.return_std),
                median=float(result.return_median),
                ci_lower=float(result.return_ci_lower),
                ci_upper=float(result.return_ci_upper),
            ),
            sharpe_ratio=MonteCarloMetrics(
                mean=float(result.sharpe_mean),
                std=float(result.sharpe_std),
                median=float(result.sharpe_median),
                ci_lower=float(result.sharpe_ci_lower),
                ci_upper=float(result.sharpe_ci_upper),
            ),
            max_drawdown=MonteCarloMetrics(
                mean=float(result.dd_mean),
                std=float(result.dd_std),
                median=float(result.dd_median),
                ci_lower=float(result.dd_ci_lower),
                ci_upper=float(result.dd_ci_upper),
            ),
            win_rate=MonteCarloMetrics(
                mean=float(result.win_rate_mean),
                std=float(result.win_rate_std),
                median=float(result.win_rate_mean),
                ci_lower=float(max(0, result.win_rate_mean - 2 * result.win_rate_std)),
                ci_upper=float(min(100, result.win_rate_mean + 2 * result.win_rate_std)),
            ),
            risk_assessment=self._calculate_risk_assessment(result),
            return_distribution=self._downsample(result.return_distribution),
            sharpe_distribution=self._downsample(result.sharpe_distribution),
            drawdown_distribution=self._downsample(result.dd_distribution),
            observed_return=observed_return,
            observed_sharpe=observed_sharpe,
            observed_drawdown=observed_drawdown,
            computation_time_ms=computation_time_ms,
        )

    def _calculate_risk_assessment(self, result) -> RiskAssessment:
        """Calculate probability-based risk metrics."""
        n = result.n_simulations
        return RiskAssessment(
            prob_loss=float(np.sum(result.return_distribution < 0) / n * 100),
            prob_low_sharpe=float(np.sum(result.sharpe_distribution < 0.5) / n * 100),
            prob_severe_drawdown=float(np.sum(result.dd_distribution < -20) / n * 100),
        )

    def _downsample(self, arr: np.ndarray) -> list[float]:
        """Downsample large arrays for API response."""
        if len(arr) <= self.MAX_DISTRIBUTION_POINTS:
            return [float(x) for x in arr]
        # Sort and take evenly spaced samples for histogram
        sorted_arr = np.sort(arr)
        indices = np.linspace(0, len(sorted_arr) - 1, self.MAX_DISTRIBUTION_POINTS, dtype=int)
        return [float(sorted_arr[i]) for i in indices]
