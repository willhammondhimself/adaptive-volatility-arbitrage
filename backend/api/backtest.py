"""
Backtest execution API endpoints.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.backtest import BacktestRequest, BacktestResponse
from backend.services.backtest_service import BacktestService

router = APIRouter(prefix="/api/v1/backtest", tags=["backtest"])

# Initialize service (singleton)
backtest_service = BacktestService()


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
