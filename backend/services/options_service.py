"""
Unified options surface service.

Orchestrates surface computation across Heston, Black-Scholes, and Market IV modes.
"""

import time
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

import numpy as np

from backend.schemas.options import (
    SurfaceMode,
    SurfaceValueType,
    UnifiedSurfaceRequest,
    UnifiedSurfaceResponse,
    IVSurfaceResponse,
)
from backend.services.heston_service import HestonService
from backend.services.live_market_service import LiveMarketService
from volatility_arbitrage.core.types import OptionType
from volatility_arbitrage.models.black_scholes import BlackScholesModel


class OptionsSurfaceService:
    """
    Unified service for computing option surfaces.

    Supports three modes:
    - Heston: FFT-based stochastic volatility pricing
    - Black-Scholes: Classic constant volatility pricing
    - Market IV: Live implied volatility from Yahoo Finance
    """

    def __init__(self, cache_size: int = 1000):
        self.heston_service = HestonService(cache_size=cache_size)
        self.market_service = LiveMarketService()

    def compute_surface(self, request: UnifiedSurfaceRequest) -> UnifiedSurfaceResponse:
        """
        Compute option surface based on mode.

        Args:
            request: Unified surface request with mode and parameters

        Returns:
            Surface response with computed values
        """
        if request.mode == SurfaceMode.HESTON:
            return self._compute_heston_surface(request)
        elif request.mode == SurfaceMode.BLACK_SCHOLES:
            return self._compute_bs_surface(request)
        elif request.mode == SurfaceMode.MARKET_IV:
            return self._compute_market_iv_surface(request)
        else:
            raise ValueError(f"Unknown mode: {request.mode}")

    def _compute_heston_surface(
        self, request: UnifiedSurfaceRequest
    ) -> UnifiedSurfaceResponse:
        """Compute surface using Heston model."""
        from backend.schemas.heston import HestonParams, PriceSurfaceRequest

        start_time = time.time()

        # Build Heston request
        heston_params = request.heston_params or {}
        params = HestonParams(
            v0=heston_params.get("v0", 0.04),
            theta=heston_params.get("theta", 0.05),
            kappa=heston_params.get("kappa", 2.0),
            sigma_v=heston_params.get("sigma_v", 0.3),
            rho=heston_params.get("rho", -0.7),
            r=request.r,
        )

        spot = request.spot or 100.0
        strike_range = request.strike_range or [spot * 0.8, spot * 1.2]
        maturity_range = request.maturity_range or [0.25, 2.0]

        heston_request = PriceSurfaceRequest(
            params=params,
            spot=spot,
            strike_range=strike_range,
            maturity_range=maturity_range,
            num_strikes=request.num_strikes,
            num_maturities=request.num_maturities,
        )

        heston_response = self.heston_service.compute_price_surface(heston_request)

        # Convert to unified response
        values = heston_response.prices

        # If IV requested, invert prices to IV
        if request.value_type == SurfaceValueType.IV:
            values = self._prices_to_iv(
                values,
                heston_response.strikes,
                heston_response.maturities,
                spot,
                request.r,
            )

        return UnifiedSurfaceResponse(
            mode=request.mode.value,
            strikes=heston_response.strikes,
            maturities=heston_response.maturities,
            values=values,
            value_type=request.value_type.value,
            computation_time_ms=(time.time() - start_time) * 1000,
            cache_hit=heston_response.cache_hit,
            underlying_price=spot,
        )

    def _compute_bs_surface(
        self, request: UnifiedSurfaceRequest
    ) -> UnifiedSurfaceResponse:
        """Compute surface using Black-Scholes model."""
        start_time = time.time()

        spot = request.spot or 100.0
        sigma = request.bs_sigma or 0.20
        r = request.r

        strike_range = request.strike_range or [spot * 0.8, spot * 1.2]
        maturity_range = request.maturity_range or [0.25, 2.0]

        strikes = np.linspace(strike_range[0], strike_range[1], request.num_strikes)
        maturities = np.linspace(
            maturity_range[0], maturity_range[1], request.num_maturities
        )

        # Compute prices
        values: List[List[float]] = []
        for T in maturities:
            row = []
            for K in strikes:
                price = BlackScholesModel.price(
                    Decimal(str(spot)),
                    Decimal(str(K)),
                    Decimal(str(T)),
                    Decimal(str(r)),
                    Decimal(str(sigma)),
                    OptionType.CALL,
                )
                row.append(float(price))
            values.append(row)

        # If IV requested, return constant sigma (BS has constant vol)
        if request.value_type == SurfaceValueType.IV:
            values = [[sigma for _ in strikes] for _ in maturities]

        return UnifiedSurfaceResponse(
            mode=request.mode.value,
            strikes=strikes.tolist(),
            maturities=maturities.tolist(),
            values=values,
            value_type=request.value_type.value,
            computation_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False,
            underlying_price=spot,
        )

    def _compute_market_iv_surface(
        self, request: UnifiedSurfaceRequest
    ) -> UnifiedSurfaceResponse:
        """Compute IV surface from live market data."""
        start_time = time.time()

        symbol = request.symbol
        if not symbol:
            raise ValueError("Symbol required for market IV mode")

        # Get first chain to get available expiries and underlying price
        first_chain = self.market_service.get_option_chain(symbol)
        underlying_price = first_chain.underlying_price
        available_expiries = first_chain.available_expiries[: request.expiry_count]

        # Collect all strikes and IVs
        all_strikes = set()
        expiry_data = []

        for expiry in available_expiries:
            chain = self.market_service.get_option_chain(symbol, expiry)

            # Collect call IVs (more liquid typically)
            iv_by_strike = {}
            for call in chain.calls:
                if call.implied_volatility and call.implied_volatility > 0:
                    all_strikes.add(call.strike)
                    iv_by_strike[call.strike] = call.implied_volatility

            expiry_data.append(
                {"expiry": expiry, "ivs": iv_by_strike}
            )

        # Build uniform strike grid
        strikes = sorted(list(all_strikes))

        # Calculate days to expiry for each expiration
        today = datetime.now()
        maturities = []
        for data in expiry_data:
            expiry_date = datetime.strptime(data["expiry"], "%Y-%m-%d")
            days = (expiry_date - today).days
            maturities.append(max(days / 365.0, 0.01))

        # Build IV matrix [maturity_idx][strike_idx]
        values: List[List[float]] = []
        for data in expiry_data:
            row = []
            for strike in strikes:
                iv = data["ivs"].get(strike)
                if iv is not None:
                    row.append(iv)
                else:
                    # Interpolate or use NaN
                    row.append(float("nan"))
            values.append(row)

        return UnifiedSurfaceResponse(
            mode=request.mode.value,
            strikes=strikes,
            maturities=maturities,
            values=values,
            value_type="iv",  # Market IV mode always returns IV
            computation_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False,
            symbol=symbol,
            underlying_price=underlying_price,
        )

    def get_iv_surface(
        self, symbol: str, expiry_count: int = 5
    ) -> IVSurfaceResponse:
        """
        Get live IV surface for a symbol.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            expiry_count: Number of expirations to include

        Returns:
            IV surface response
        """
        start_time = time.time()

        # Get first chain for expiries list
        first_chain = self.market_service.get_option_chain(symbol)
        underlying_price = first_chain.underlying_price
        available_expiries = first_chain.available_expiries[:expiry_count]

        # Collect strikes and IVs from all expiries
        all_strikes = set()
        expiry_ivs = []

        for expiry in available_expiries:
            chain = self.market_service.get_option_chain(symbol, expiry)

            # Use calls (typically more liquid for OTM strikes > underlying)
            iv_by_strike = {}
            for call in chain.calls:
                if call.implied_volatility and call.implied_volatility > 0:
                    all_strikes.add(call.strike)
                    iv_by_strike[call.strike] = call.implied_volatility

            expiry_ivs.append(iv_by_strike)

        strikes = sorted(list(all_strikes))

        # Convert expiries to maturities (years)
        today = datetime.now()
        maturities = []
        for expiry in available_expiries:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            days = (expiry_date - today).days
            maturities.append(max(days / 365.0, 0.01))

        # Build IV matrix
        ivs: List[List[float]] = []
        for iv_by_strike in expiry_ivs:
            row = []
            for strike in strikes:
                iv = iv_by_strike.get(strike)
                row.append(iv if iv is not None else float("nan"))
            ivs.append(row)

        return IVSurfaceResponse(
            symbol=symbol.upper(),
            underlying_price=underlying_price,
            strikes=strikes,
            maturities=maturities,
            expiry_dates=available_expiries,
            ivs=ivs,
            computation_time_ms=(time.time() - start_time) * 1000,
        )

    def _prices_to_iv(
        self,
        prices: List[List[float]],
        strikes: List[float],
        maturities: List[float],
        spot: float,
        r: float,
    ) -> List[List[float]]:
        """Convert price surface to IV surface via BS inversion."""
        ivs: List[List[float]] = []

        for mat_idx, T in enumerate(maturities):
            row = []
            for strike_idx, K in enumerate(strikes):
                price = prices[mat_idx][strike_idx]
                if price <= 0:
                    row.append(float("nan"))
                    continue

                iv = BlackScholesModel.calculate_implied_volatility(
                    Decimal(str(price)),
                    Decimal(str(spot)),
                    Decimal(str(K)),
                    Decimal(str(T)),
                    Decimal(str(r)),
                    OptionType.CALL,
                )

                if iv is not None:
                    row.append(float(iv))
                else:
                    row.append(float("nan"))
            ivs.append(row)

        return ivs
