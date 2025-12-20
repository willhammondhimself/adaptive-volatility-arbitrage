"""
Pydantic schemas for backtest API endpoints.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    """Request to run a backtest."""

    # Data source
    data_dir: str = Field(
        default="src/volatility_arbitrage/data/SPY_Options_2019_24",
        description="Directory containing JSON options data",
    )
    max_days: Optional[int] = Field(
        default=None, ge=1, description="Maximum days to run (None for all)"
    )

    # Capital
    initial_capital: float = Field(
        default=100000.0, gt=0, description="Starting capital"
    )

    # Strategy parameters
    entry_threshold_pct: float = Field(
        default=5.0, ge=0, description="Entry threshold as percentage"
    )
    exit_threshold_pct: float = Field(
        default=2.0, ge=0, description="Exit threshold as percentage"
    )
    position_size_pct: float = Field(
        default=15.0, ge=1, le=50, description="Position size as percentage of capital"
    )
    max_positions: int = Field(default=5, ge=1, le=20, description="Maximum positions")

    # Demo mode for instant response
    demo_mode: bool = Field(
        default=False, description="Return mock data for UI testing"
    )

    # Phase 2 toggles
    use_bayesian_lstm: bool = Field(
        default=False, description="Use Bayesian LSTM for vol forecasting"
    )
    use_impact_model: bool = Field(
        default=False, description="Use square-root impact model for costs"
    )
    use_uncertainty_sizing: bool = Field(
        default=False, description="Use uncertainty-adjusted position sizing"
    )
    use_leverage: bool = Field(default=False, description="Enable leverage")


class BacktestMetrics(BaseModel):
    """Performance metrics from backtest."""

    total_return: float = Field(..., description="Total return as decimal")
    sharpe_ratio: float = Field(..., description="Annualized Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown as decimal")
    total_trades: int = Field(..., ge=0, description="Total number of trades")
    win_rate: Optional[float] = Field(
        default=None, ge=0, le=1, description="Win rate if trades > 0"
    )
    annual_return: Optional[float] = Field(
        default=None, description="Annualized return"
    )
    volatility: Optional[float] = Field(
        default=None, description="Annualized volatility"
    )


class EquityPoint(BaseModel):
    """Single point on equity curve."""

    date: str = Field(..., description="Date in ISO format")
    equity: float = Field(..., description="Portfolio equity")
    drawdown: float = Field(..., description="Current drawdown as decimal")
    buy_hold_equity: Optional[float] = Field(default=None, description="Buy-and-hold equity")


class Phase2Status(BaseModel):
    """Status of Phase 2 features."""

    bayesian_lstm_active: bool
    impact_model_active: bool
    uncertainty_sizer_active: bool
    leverage_active: bool


class BacktestResponse(BaseModel):
    """Response containing backtest results."""

    metrics: BacktestMetrics
    equity_curve: List[EquityPoint]
    phase2_status: Phase2Status
    computation_time_ms: float = Field(..., description="Time to run backtest in ms")
    data_range: Dict[str, str] = Field(
        ..., description="Start and end dates of backtest"
    )
    trade_returns: Optional[List[float]] = Field(
        default=None,
        description="Trade returns for Monte Carlo resampling (decimals)",
    )


# Monte Carlo schemas

class MonteCarloRequest(BaseModel):
    """Request to run Monte Carlo simulation on trade returns."""

    trade_returns: List[float] = Field(
        ..., min_length=5, description="Trade returns as decimals"
    )
    n_simulations: int = Field(
        default=10000, ge=100, le=100000, description="Number of bootstrap samples"
    )
    block_size: int = Field(
        default=3, ge=1, le=10, description="Block size for block bootstrap"
    )
    initial_capital: float = Field(
        default=100000.0, gt=0, description="Initial capital for equity calculations"
    )
    random_seed: Optional[int] = Field(
        default=None, description="Optional seed for reproducibility"
    )
    winsorize_pct: Optional[float] = Field(
        default=None, ge=90, le=99, description="Winsorize outliers at percentile"
    )


class MonteCarloMetrics(BaseModel):
    """Statistics for a single metric from MC simulation."""

    mean: float
    std: float
    median: float
    ci_lower: float = Field(..., description="2.5th percentile (95% CI lower)")
    ci_upper: float = Field(..., description="97.5th percentile (95% CI upper)")


class RiskAssessment(BaseModel):
    """Probability assessments from MC distribution."""

    prob_loss: float = Field(..., ge=0, le=100, description="P(total return < 0)")
    prob_low_sharpe: float = Field(..., ge=0, le=100, description="P(Sharpe < 0.5)")
    prob_severe_drawdown: float = Field(
        ..., ge=0, le=100, description="P(DD > 20%)"
    )


class MonteCarloResponse(BaseModel):
    """Response from Monte Carlo simulation."""

    n_simulations: int
    n_trades: int

    total_return: MonteCarloMetrics
    sharpe_ratio: MonteCarloMetrics
    max_drawdown: MonteCarloMetrics
    win_rate: MonteCarloMetrics

    risk_assessment: RiskAssessment

    # Histogram data for frontend charts (downsampled)
    return_distribution: List[float]
    sharpe_distribution: List[float]
    drawdown_distribution: List[float]

    # Observed values from actual backtest (for histogram markers)
    observed_return: Optional[float] = Field(
        default=None, description="Actual backtest total return %"
    )
    observed_sharpe: Optional[float] = Field(
        default=None, description="Actual backtest Sharpe ratio"
    )
    observed_drawdown: Optional[float] = Field(
        default=None, description="Actual backtest max drawdown %"
    )

    computation_time_ms: float
