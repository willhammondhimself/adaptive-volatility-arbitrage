"""
Live market data API endpoints.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from backend.schemas.market import (
    QuoteResponse,
    OptionChainResponse,
    VixResponse,
    MarketStatusResponse,
)
from backend.services.live_market_service import LiveMarketService

router = APIRouter(prefix="/api/v1/market", tags=["market"])

# Initialize service (singleton)
market_service = LiveMarketService()


@router.get("/quote/{symbol}", response_model=QuoteResponse)
async def get_quote(symbol: str) -> QuoteResponse:
    """
    Get stock quote.

    Fetches current price, bid/ask, and change from Yahoo Finance.
    Data is cached for 30 seconds.

    **Example:** GET /api/v1/market/quote/SPY
    """
    try:
        return market_service.get_quote(symbol)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch quote for {symbol}: {str(e)}",
        )


@router.get("/option-chain/{symbol}", response_model=OptionChainResponse)
async def get_option_chain(
    symbol: str,
    expiry: Optional[str] = Query(
        default=None, description="Expiration date (YYYY-MM-DD). Uses nearest if not specified."
    ),
) -> OptionChainResponse:
    """
    Get option chain.

    Fetches calls and puts with strikes, bids/asks, and implied volatility.
    Data is cached for 60 seconds.

    **Example:** GET /api/v1/market/option-chain/SPY?expiry=2025-01-17
    """
    try:
        return market_service.get_option_chain(symbol, expiry)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch option chain for {symbol}: {str(e)}",
        )


@router.get("/vix", response_model=VixResponse)
async def get_vix() -> VixResponse:
    """
    Get VIX quote.

    Fetches current VIX level and change from Yahoo Finance.
    Data is cached for 30 seconds.

    **Example:** GET /api/v1/market/vix
    """
    try:
        return market_service.get_vix()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch VIX: {str(e)}",
        )


@router.get("/status", response_model=MarketStatusResponse)
async def get_market_status() -> MarketStatusResponse:
    """
    Get market status.

    Returns whether US stock market is open and the current market phase
    (pre, regular, after, closed).

    **Example:** GET /api/v1/market/status
    """
    return market_service.get_market_status()


@router.delete("/cache")
async def clear_cache() -> dict:
    """
    Clear all market data caches.

    Useful for forcing fresh data fetch.

    **Example:** DELETE /api/v1/market/cache
    """
    return market_service.clear_cache()
