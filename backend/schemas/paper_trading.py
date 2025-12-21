"""
Pydantic schemas for paper trading API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    """Request to start paper trading."""

    initial_capital: float = Field(default=100000.0, ge=1000, le=10_000_000)
    uncertainty_threshold: float = Field(default=0.02, ge=0.001, le=0.5)
    position_pct: float = Field(default=0.10, ge=0.01, le=0.50)


class StartResponse(BaseModel):
    """Response after starting paper trading."""

    session_id: int
    message: str


class StopResponse(BaseModel):
    """Response after stopping paper trading."""

    session_id: int
    stats: "StatsResponse"


class StatusResponse(BaseModel):
    """Current paper trading status."""

    is_running: bool
    session_id: Optional[int]
    capital: float
    position: int
    avg_cost: float
    cumulative_pnl: float
    last_update: Optional[str]
    tick_count: int


class TradeRecord(BaseModel):
    """Single trade record."""

    id: int
    timestamp: str
    symbol: str
    side: str
    quantity: int
    price: float
    forecast_vol: float
    uncertainty: float
    pnl: Optional[float]
    cumulative_pnl: Optional[float]


class TradesResponse(BaseModel):
    """List of trades."""

    session_id: Optional[int]
    trades: list[TradeRecord]


class StatsResponse(BaseModel):
    """Trading statistics."""

    total_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_estimate: float
    skipped_ticks: int


# For forward reference
StopResponse.model_rebuild()
