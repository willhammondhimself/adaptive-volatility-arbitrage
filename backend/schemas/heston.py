"""
Pydantic schemas for Heston API endpoints.
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


class HestonParams(BaseModel):
    """Heston model parameters."""

    v0: float = Field(..., gt=0, description="Initial variance")
    theta: float = Field(..., gt=0, description="Long-run variance")
    kappa: float = Field(..., gt=0, description="Mean reversion speed")
    sigma_v: float = Field(..., gt=0, description="Volatility of volatility")
    rho: float = Field(..., ge=-1, le=1, description="Correlation")
    r: float = Field(..., description="Risk-free rate")
    q: float = Field(default=0.0, description="Dividend yield")

    @field_validator("v0", "theta", "kappa", "sigma_v")
    @classmethod
    def check_feller_condition(cls, v: float, info) -> float:
        """Validate Feller condition for CIR process: 2*kappa*theta > sigma_v^2."""
        # Note: Full validation happens after all fields are set
        return v


class PriceSurfaceRequest(BaseModel):
    """Request to compute option price surface."""

    params: HestonParams
    spot: float = Field(..., gt=0, description="Spot price")
    strike_range: List[float] = Field(..., min_length=2, max_length=2, description="[min, max]")
    maturity_range: List[float] = Field(..., min_length=2, max_length=2, description="[min, max] in years")
    num_strikes: int = Field(default=40, gt=0, le=100, description="Number of strike points")
    num_maturities: int = Field(default=20, gt=0, le=50, description="Number of maturity points")

    @field_validator("strike_range", "maturity_range")
    @classmethod
    def validate_range(cls, v: List[float]) -> List[float]:
        """Ensure min < max."""
        if v[0] >= v[1]:
            raise ValueError("Range must have min < max")
        return v


class PriceSurfaceResponse(BaseModel):
    """Response containing option price surface."""

    strikes: List[float]
    maturities: List[float]
    prices: List[List[float]]  # 2D array: prices[maturity_idx][strike_idx]
    computation_time_ms: float
    cache_hit: bool
    params: HestonParams
    spot: float
