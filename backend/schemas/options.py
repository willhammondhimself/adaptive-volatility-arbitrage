"""Options pricing schemas for unified surface endpoint and BS playground."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SurfaceMode(str, Enum):
    """Surface computation mode."""

    HESTON = "heston"
    BLACK_SCHOLES = "black_scholes"
    MARKET_IV = "market_iv"


class SurfaceValueType(str, Enum):
    """Surface value type."""

    PRICE = "price"
    IV = "iv"


# --- Unified Surface ---


class UnifiedSurfaceRequest(BaseModel):
    """Request for unified surface computation."""

    mode: SurfaceMode
    value_type: SurfaceValueType = SurfaceValueType.PRICE

    # Model params (Heston)
    heston_params: Optional[dict] = None  # v0, theta, kappa, sigma_v, rho

    # Model params (Black-Scholes)
    bs_sigma: Optional[float] = Field(None, gt=0, le=2.0)

    # Common params
    spot: Optional[float] = Field(None, gt=0)
    r: float = Field(default=0.05)
    strike_range: Optional[List[float]] = None  # [min, max]
    maturity_range: Optional[List[float]] = None  # [min, max] in years
    num_strikes: int = Field(default=40, gt=0, le=100)
    num_maturities: int = Field(default=20, gt=0, le=50)

    # Market IV params
    symbol: Optional[str] = None
    expiry_count: int = Field(default=5, gt=0, le=10)


class UnifiedSurfaceResponse(BaseModel):
    """Response from unified surface computation."""

    mode: str
    strikes: List[float]
    maturities: List[float]
    values: List[List[float]]  # [maturity_idx][strike_idx]
    value_type: str  # "price" or "iv"
    computation_time_ms: float
    cache_hit: bool = False
    symbol: Optional[str] = None
    underlying_price: Optional[float] = None


# --- Black-Scholes Playground ---


class BSPriceRequest(BaseModel):
    """Request for BS single option pricing."""

    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, description="Time to expiry in years")
    r: float = Field(default=0.05, description="Risk-free rate")
    sigma: float = Field(..., gt=0, le=3.0, description="Volatility")
    option_type: str = Field(default="call", pattern="^(call|put)$")


class BSPriceResponse(BaseModel):
    """Response with option price and Greeks."""

    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class BSPnLHeatmapRequest(BaseModel):
    """Request for P&L heatmap computation."""

    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    r: float = Field(default=0.05)
    sigma: float = Field(..., gt=0, le=3.0)
    option_type: str = Field(default="call", pattern="^(call|put)$")
    entry_price: float = Field(..., description="Entry price to compute P&L from")
    spot_range: List[float] = Field(..., min_length=2, max_length=2)
    vol_range: List[float] = Field(..., min_length=2, max_length=2)
    num_spots: int = Field(default=50, gt=0, le=100)
    num_vols: int = Field(default=30, gt=0, le=100)


class BSPnLHeatmapResponse(BaseModel):
    """Response with P&L grid."""

    spots: List[float]
    vols: List[float]
    pnl: List[List[float]]  # [vol_idx][spot_idx]


# --- IV Surface ---


class IVSurfaceRequest(BaseModel):
    """Request for live IV surface."""

    symbol: str
    expiry_count: int = Field(default=5, gt=0, le=10)


class IVSurfaceResponse(BaseModel):
    """Response with IV surface data."""

    symbol: str
    underlying_price: float
    strikes: List[float]
    maturities: List[float]  # in years
    expiry_dates: List[str]  # ISO format dates
    ivs: List[List[float]]  # [maturity_idx][strike_idx]
    computation_time_ms: float


# --- Snapshots ---


class SnapshotMetadata(BaseModel):
    """Snapshot list item."""

    id: int
    symbol: str
    captured_at: str
    underlying_price: float
    vix_level: Optional[float] = None


class SnapshotDetail(BaseModel):
    """Full snapshot data."""

    id: int
    symbol: str
    captured_at: str
    underlying_price: float
    vix_level: Optional[float] = None
    strikes: List[float]
    maturities: List[float]
    values: List[List[Optional[float]]]
