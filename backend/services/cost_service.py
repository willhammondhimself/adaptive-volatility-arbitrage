"""
Transaction cost estimation service using Square Root Impact Model.
"""

import sys
import time
from pathlib import Path

# Add src path to import volatility_arbitrage modules
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from volatility_arbitrage.execution.costs import SquareRootImpactModel

from backend.schemas.costs import CostEstimateRequest, CostEstimateResponse


class CostService:
    """Service for transaction cost estimation."""

    def __init__(
        self,
        default_half_spread_bps: float = 5.0,
        default_impact_coeff: float = 0.1,
    ):
        self.default_half_spread_bps = default_half_spread_bps
        self.default_impact_coeff = default_impact_coeff

    def estimate(self, request: CostEstimateRequest) -> CostEstimateResponse:
        """
        Estimate transaction costs for an order.

        Args:
            request: Cost estimate request with order details

        Returns:
            Cost breakdown with spread, impact, and effective price
        """
        # Create model with request parameters
        model = SquareRootImpactModel(
            half_spread_bps=request.half_spread_bps,
            impact_coeff=request.impact_coefficient,
        )

        # Calculate order value
        order_value = request.order_size * request.price

        # Calculate spread cost
        spread_cost = order_value * (request.half_spread_bps / 10_000)

        # Calculate impact in bps
        # Assume moderate volatility if not provided (20%)
        volatility = 0.20
        impact_bps = model.estimate_impact_bps(
            order_size=request.order_size,
            volatility=volatility,
            daily_volume=request.daily_volume,
        )

        # Calculate impact cost
        impact_cost = order_value * (impact_bps / 10_000)

        # Total cost
        total_cost = spread_cost + impact_cost

        # Effective price
        effective_price = request.price + (total_cost / request.order_size)

        # Cost as percentage
        cost_as_pct = (total_cost / order_value) * 100 if order_value > 0 else 0.0

        return CostEstimateResponse(
            total_cost=total_cost,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            impact_bps=impact_bps,
            effective_price=effective_price,
            cost_as_pct=cost_as_pct,
        )
