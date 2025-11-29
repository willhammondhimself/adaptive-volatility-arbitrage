"""
Heston option pricing service with caching.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add research path to import HestonFFT
research_path = Path(__file__).parent.parent.parent / "research" / "lib"
sys.path.insert(0, str(research_path))

from heston_fft import HestonFFT  # type: ignore

from backend.schemas.heston import HestonParams, PriceSurfaceRequest, PriceSurfaceResponse
from backend.services.cache_service import LRUCache


class HestonService:
    """Service for computing option price surfaces using Heston FFT."""

    def __init__(self, cache_size: int = 1000):
        self.cache = LRUCache(max_size=cache_size)

    def compute_price_surface(self, request: PriceSurfaceRequest) -> PriceSurfaceResponse:
        """
        Compute option price surface across strikes and maturities.

        Args:
            request: Price surface request with parameters and grid specification

        Returns:
            Price surface response with computed prices and metadata
        """
        start_time = time.time()

        # Check cache
        cache_key = self._create_cache_key(request)
        cached_result = self.cache.get(cache_key)

        if cached_result is not None:
            cached_result["cache_hit"] = True
            cached_result["computation_time_ms"] = (time.time() - start_time) * 1000
            return PriceSurfaceResponse(**cached_result)

        # Cache miss - compute surface
        heston = HestonFFT(
            v0=request.params.v0,
            theta=request.params.theta,
            kappa=request.params.kappa,
            sigma_v=request.params.sigma_v,
            rho=request.params.rho,
            r=request.params.r,
            q=request.params.q,
        )

        # Generate grid
        strikes = np.linspace(
            request.strike_range[0], request.strike_range[1], request.num_strikes
        )
        maturities = np.linspace(
            request.maturity_range[0], request.maturity_range[1], request.num_maturities
        )

        # Compute prices for each maturity
        prices: List[List[float]] = []
        for T in maturities:
            row_prices = heston.price_range(S=request.spot, strikes=strikes, T=T)
            prices.append(row_prices.tolist())

        computation_time_ms = (time.time() - start_time) * 1000

        # Build response
        result = {
            "strikes": strikes.tolist(),
            "maturities": maturities.tolist(),
            "prices": prices,
            "computation_time_ms": computation_time_ms,
            "cache_hit": False,
            "params": request.params.model_dump(),
            "spot": request.spot,
        }

        # Cache the result
        self.cache.put(cache_key, result)

        return PriceSurfaceResponse(**result)

    def _create_cache_key(self, request: PriceSurfaceRequest) -> str:
        """Create deterministic cache key from request."""
        cache_data = {
            "params": request.params.model_dump(),
            "spot": request.spot,
            "strike_range": request.strike_range,
            "maturity_range": request.maturity_range,
            "num_strikes": request.num_strikes,
            "num_maturities": request.num_maturities,
        }
        return LRUCache.hash_dict(cache_data)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {"size": self.cache.size(), "max_size": self.cache.max_size}
