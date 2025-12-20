"""
Transaction cost estimation API endpoints.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.costs import CostEstimateRequest, CostEstimateResponse
from backend.services.cost_service import CostService

router = APIRouter(prefix="/api/v1/costs", tags=["costs"])

# Initialize service (singleton)
cost_service = CostService()


@router.post("/estimate", response_model=CostEstimateResponse)
async def estimate_costs(request: CostEstimateRequest) -> CostEstimateResponse:
    """
    Estimate transaction costs for an order.

    Uses Square Root Impact Model (Almgren-Chriss) to calculate:
    - Spread cost: Fixed cost from bid-ask spread
    - Market impact: Temporary price impact from order execution
    - Effective price: Total cost-adjusted execution price

    **Input:**
    - order_size: Number of contracts/shares
    - price: Current price per unit
    - daily_volume: Average daily volume for impact calculation
    - half_spread_bps: Half bid-ask spread in basis points (default: 5)
    - impact_coefficient: Market impact coefficient (default: 0.1)

    **Performance:**
    - Typical: <1ms
    """
    try:
        return cost_service.estimate(request)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error estimating costs: {str(e)}"
        )
