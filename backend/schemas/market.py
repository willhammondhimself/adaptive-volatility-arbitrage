"""
Pydantic schemas for live market data API endpoints.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class QuoteResponse(BaseModel):
    """Stock quote response."""

    symbol: str = Field(..., description="Ticker symbol")
    price: float = Field(..., description="Current price")
    bid: Optional[float] = Field(default=None, description="Bid price")
    ask: Optional[float] = Field(default=None, description="Ask price")
    change: Optional[float] = Field(default=None, description="Price change")
    change_percent: Optional[float] = Field(default=None, description="Percent change")
    volume: Optional[int] = Field(default=None, description="Trading volume")
    timestamp: datetime = Field(..., description="Quote timestamp")
    is_stale: bool = Field(default=False, description="True if data is from stale cache")


class OptionContractResponse(BaseModel):
    """Single option contract."""

    strike: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


class OptionChainResponse(BaseModel):
    """Option chain response."""

    symbol: str = Field(..., description="Underlying symbol")
    expiry: str = Field(..., description="Expiration date (YYYY-MM-DD)")
    underlying_price: float = Field(..., description="Current underlying price")
    risk_free_rate: Optional[float] = Field(default=None, description="Risk-free rate")
    calls: List[OptionContractResponse] = Field(default_factory=list)
    puts: List[OptionContractResponse] = Field(default_factory=list)
    available_expiries: List[str] = Field(
        default_factory=list, description="Available expiration dates"
    )
    timestamp: datetime = Field(..., description="Data timestamp")
    is_stale: bool = Field(default=False, description="True if data is from stale cache")


class VixResponse(BaseModel):
    """VIX quote response."""

    level: float = Field(..., description="VIX level")
    change: Optional[float] = Field(default=None, description="Change from previous close")
    change_percent: Optional[float] = Field(default=None, description="Percent change")
    timestamp: datetime = Field(..., description="Quote timestamp")
    is_stale: bool = Field(default=False, description="True if data is from stale cache")


class MarketStatusResponse(BaseModel):
    """Market hours status."""

    is_open: bool = Field(..., description="True if market is open")
    market_phase: str = Field(
        ..., description="Market phase: pre, regular, after, closed"
    )
    current_time: datetime = Field(..., description="Current time (ET)")
    next_open: Optional[datetime] = Field(default=None, description="Next market open")
    next_close: Optional[datetime] = Field(default=None, description="Next market close")
