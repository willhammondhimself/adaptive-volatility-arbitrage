"""Options pricing API endpoints for BS playground and unified surface."""

import time
from decimal import Decimal
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.schemas.options import (
    BSPnLHeatmapRequest,
    BSPnLHeatmapResponse,
    BSPriceRequest,
    BSPriceResponse,
    UnifiedSurfaceRequest,
    UnifiedSurfaceResponse,
    IVSurfaceResponse,
)
from backend.services.options_service import OptionsSurfaceService
from volatility_arbitrage.core.types import OptionType
from volatility_arbitrage.models.black_scholes import BlackScholesModel

router = APIRouter(prefix="/api/v1/options", tags=["options"])

# Initialize service (singleton)
options_service = OptionsSurfaceService(cache_size=1000)


@router.post("/bs/price", response_model=BSPriceResponse)
async def compute_bs_price(request: BSPriceRequest) -> BSPriceResponse:
    """
    Price a single option with all Greeks.

    Returns price, delta, gamma, theta, vega, rho.
    """
    opt_type = OptionType.CALL if request.option_type == "call" else OptionType.PUT

    S = Decimal(str(request.S))
    K = Decimal(str(request.K))
    T = Decimal(str(request.T))
    r = Decimal(str(request.r))
    sigma = Decimal(str(request.sigma))

    price = BlackScholesModel.price(S, K, T, r, sigma, opt_type)
    greeks = BlackScholesModel.greeks(S, K, T, r, sigma, opt_type)

    return BSPriceResponse(
        price=float(price),
        delta=float(greeks.delta),
        gamma=float(greeks.gamma),
        theta=float(greeks.theta),
        vega=float(greeks.vega),
        rho=float(greeks.rho),
    )


@router.post("/bs/pnl-heatmap", response_model=BSPnLHeatmapResponse)
async def compute_pnl_heatmap(request: BSPnLHeatmapRequest) -> BSPnLHeatmapResponse:
    """
    Compute P&L grid across spot × vol shocks.

    Returns a 2D grid of P&L values for different spot prices and volatilities.
    """
    opt_type = OptionType.CALL if request.option_type == "call" else OptionType.PUT

    # Generate spot and vol ranges
    spots = np.linspace(request.spot_range[0], request.spot_range[1], request.num_spots)
    vols = np.linspace(request.vol_range[0], request.vol_range[1], request.num_vols)

    K = Decimal(str(request.K))
    T = Decimal(str(request.T))
    r = Decimal(str(request.r))
    entry_price = request.entry_price

    # Compute P&L grid: [vol_idx][spot_idx]
    pnl: List[List[float]] = []
    for vol in vols:
        row = []
        for spot in spots:
            price = BlackScholesModel.price(
                Decimal(str(spot)),
                K,
                T,
                r,
                Decimal(str(vol)),
                opt_type,
            )
            row.append(float(price) - entry_price)
        pnl.append(row)

    return BSPnLHeatmapResponse(
        spots=spots.tolist(),
        vols=vols.tolist(),
        pnl=pnl,
    )


@router.post("/surface", response_model=UnifiedSurfaceResponse)
async def compute_unified_surface(
    request: UnifiedSurfaceRequest,
) -> UnifiedSurfaceResponse:
    """
    Compute option surface with model and value type switching.

    Modes:
    - heston: Stochastic volatility Heston model
    - black_scholes: Classic constant volatility model
    - market_iv: Live implied volatility from Yahoo Finance

    Value types:
    - price: Option prices
    - iv: Implied volatility surface
    """
    try:
        return options_service.compute_surface(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/iv-surface/{symbol}", response_model=IVSurfaceResponse)
async def get_iv_surface(symbol: str, expiry_count: int = 5) -> IVSurfaceResponse:
    """
    Get live implied volatility surface for a symbol.

    Fetches option chains from Yahoo Finance and extracts IV values.

    Args:
        symbol: Underlying symbol (e.g., SPY)
        expiry_count: Number of expirations to include (1-10)
    """
    try:
        return options_service.get_iv_surface(symbol, expiry_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
