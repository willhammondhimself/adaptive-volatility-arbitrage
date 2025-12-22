"""
Backtest execution API endpoints.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.backtest import (
    BacktestRequest,
    BacktestResponse,
    DeltaHedgedRequest,
    DeltaHedgedResponse,
    MonteCarloRequest,
    MonteCarloResponse,
)
from backend.services.backtest_service import BacktestService
from backend.services.delta_hedged_service import DeltaHedgedService
from backend.services.monte_carlo_service import MonteCarloService

router = APIRouter(prefix="/api/v1/backtest", tags=["backtest"])

# Initialize services (singleton)
backtest_service = BacktestService()
delta_hedged_service = DeltaHedgedService()
monte_carlo_service = MonteCarloService()


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Execute a volatility arbitrage backtest.

    Runs the QV (Quantitative Volatility) strategy with configurable
    Phase 2 features:
    - Bayesian LSTM volatility forecasting
    - Square-root market impact model
    - Uncertainty-adjusted position sizing
    - Directional leverage

    **Input:**
    - data_dir: Path to JSON options data
    - max_days: Limit backtest duration (None for full history)
    - initial_capital: Starting capital
    - Strategy parameters (entry/exit thresholds, position sizing)
    - Phase 2 toggles (use_bayesian_lstm, use_impact_model, etc.)

    **Output:**
    - Performance metrics (return, Sharpe, max drawdown, trades)
    - Equity curve with drawdown tracking
    - Phase 2 feature activation status

    **Performance:**
    - Full dataset (~5 years): 5-15 minutes
    - max_days=100: ~30 seconds
    - max_days=30: ~10 seconds

    **Note:** This is a long-running operation. Consider using
    `max_days` parameter for quick testing.
    """
    try:
        return backtest_service.run(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data not found: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error running backtest: {str(e)}"
        )


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def run_monte_carlo(request: MonteCarloRequest) -> MonteCarloResponse:
    """
    Run Monte Carlo simulation on trade returns.

    Uses block bootstrap resampling to preserve serial correlation
    (winning/losing streaks) in trade sequences. Returns 95% confidence
    intervals for total return, Sharpe ratio, and max drawdown.

    **Workflow:**
    1. Run backtest via POST /run to get trade_returns
    2. Call this endpoint with those returns
    3. Display distribution histograms with confidence intervals

    **Input:**
    - trade_returns: Array of trade returns as decimals (e.g., 0.05 = 5%)
    - n_simulations: Number of bootstrap samples (default 10,000)
    - block_size: Trades per block for block bootstrap (default 3)

    **Output:**
    - Confidence intervals for total return, Sharpe, max drawdown
    - Risk probabilities: P(loss), P(Sharpe < 0.5), P(DD > 20%)
    - Downsampled distributions for histogram visualization

    **Performance:** ~1-2 seconds for 10,000 simulations
    """
    try:
        return monte_carlo_service.run(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Monte Carlo simulation error: {str(e)}"
        )


@router.post("/delta-hedged", response_model=DeltaHedgedResponse)
async def run_delta_hedged(request: DeltaHedgedRequest) -> DeltaHedgedResponse:
    """
    Run delta-hedged backtest with P&L attribution.

    Simulates a delta-neutral options portfolio and decomposes P&L into:
    - **Delta P&L**: Should be ~0 if hedging is effective
    - **Gamma P&L**: Convexity gains from price moves (½Γ·ΔS²)
    - **Vega P&L**: Gains from IV changes (ν·Δσ)
    - **Theta P&L**: Time decay
    - **Transaction costs**: From rebalancing the hedge

    The key insight: Vega+Gamma P&L is the alpha source. A properly hedged
    portfolio shows flat Delta P&L while Vega+Gamma drifts positive.

    **Input:**
    - days: Simulation length (5-365)
    - initial_spot/iv: Starting conditions
    - strike/expiry_days: Option parameters
    - rebalance_frequency: continuous|hourly|four_hour|daily
    - delta_threshold: Rebalance when |delta| exceeds this

    **Output:**
    - metrics: Sharpe ratios, total P&L, hedge effectiveness (R²)
    - attribution: Time series of cumulative P&L by component

    **Performance:** ~100-500ms depending on simulation length
    """
    try:
        return delta_hedged_service.run(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Delta-hedged backtest error: {str(e)}"
        )
