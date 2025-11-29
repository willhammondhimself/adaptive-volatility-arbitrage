"""
Heston API endpoints.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.heston import PriceSurfaceRequest, PriceSurfaceResponse
from backend.services.heston_service import HestonService

router = APIRouter(prefix="/api/v1/heston", tags=["heston"])

# Initialize service (singleton)
heston_service = HestonService(cache_size=1000)


@router.post("/price-surface", response_model=PriceSurfaceResponse)
async def compute_price_surface(request: PriceSurfaceRequest) -> PriceSurfaceResponse:
    """
    Compute option price surface across strikes and maturities using Heston FFT.

    The surface is computed for call options across a grid of strike prices
    and times to maturity. Results are cached for improved performance.

    **Performance:**
    - Cache hit: <5ms
    - Cache miss: 150-300ms (depending on grid size)
    - Expected cache hit rate: ~80%
    """
    try:
        return heston_service.compute_price_surface(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing price surface: {str(e)}")


@router.delete("/cache")
async def clear_cache() -> dict:
    """Clear the price surface cache."""
    heston_service.clear_cache()
    return {"message": "Cache cleared successfully"}


@router.get("/cache/stats")
async def get_cache_stats() -> dict:
    """Get cache statistics."""
    return heston_service.get_cache_stats()
