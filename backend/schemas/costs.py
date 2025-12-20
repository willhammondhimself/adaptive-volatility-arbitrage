"""
Pydantic schemas for transaction cost API endpoints.
"""

from pydantic import BaseModel, Field


class CostEstimateRequest(BaseModel):
    """Request for transaction cost estimate."""

    order_size: float = Field(..., gt=0, description="Number of contracts or shares")
    price: float = Field(..., gt=0, description="Current price per unit")
    daily_volume: float = Field(
        ..., gt=0, description="Average daily volume for normalization"
    )
    half_spread_bps: float = Field(
        default=5.0, ge=0, description="Half bid-ask spread in basis points"
    )
    impact_coefficient: float = Field(
        default=0.1, ge=0, description="Market impact coefficient (Almgren-Chriss)"
    )


class CostEstimateResponse(BaseModel):
    """Response containing transaction cost breakdown."""

    total_cost: float = Field(..., description="Total transaction cost in dollars")
    spread_cost: float = Field(..., description="Cost from bid-ask spread")
    impact_cost: float = Field(..., description="Cost from market impact")
    impact_bps: float = Field(..., description="Market impact in basis points")
    effective_price: float = Field(
        ..., description="Effective execution price (price + cost/size)"
    )
    cost_as_pct: float = Field(
        ..., description="Total cost as percentage of notional"
    )
