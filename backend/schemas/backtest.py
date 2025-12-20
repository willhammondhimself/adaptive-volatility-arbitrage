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
