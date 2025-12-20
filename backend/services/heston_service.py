"""
Heston option pricing service with caching.
"""

import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src path to import volatility_arbitrage modules
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from volatility_arbitrage.models.heston import HestonModel, HestonParameters
from volatility_arbitrage.core.types import OptionType

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
        heston_params = HestonParameters(
            v0=Decimal(str(request.params.v0)),
            theta=Decimal(str(request.params.theta)),
            kappa=Decimal(str(request.params.kappa)),
            xi=Decimal(str(request.params.sigma_v)),  # sigma_v -> xi
            rho=Decimal(str(request.params.rho)),
        )
        heston = HestonModel(heston_params)

        # Generate grid
        strikes = np.linspace(
            request.strike_range[0], request.strike_range[1], request.num_strikes
        )
        maturities = np.linspace(
            request.maturity_range[0], request.maturity_range[1], request.num_maturities
        )

        # Compute call prices for each maturity
        r_decimal = Decimal(str(request.params.r))
        prices: List[List[float]] = []
        for T in maturities:
            row_prices = []
            for K in strikes:
                price = heston.price(
                    S=Decimal(str(request.spot)),
                    K=Decimal(str(K)),
                    T=Decimal(str(T)),
                    r=r_decimal,
                    option_type=OptionType.CALL,
                )
                row_prices.append(float(price))
            prices.append(row_prices)

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
