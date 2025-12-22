"""
Live market data service.

Wraps Yahoo Finance data fetchers with TTL caching for API responses.
"""

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import pytz
import pandas as pd


def _safe_int(value) -> Optional[int]:
    """Safely convert to int, handling NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if pd.isna(value):
        return None
    return int(value)


def _safe_float(value) -> Optional[float]:
    """Safely convert to float, handling NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)

# Add src path for data fetchers
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import yfinance as yf

from backend.services.ttl_cache import TTLCache
from backend.schemas.market import (
    QuoteResponse,
    OptionChainResponse,
    OptionContractResponse,
    VixResponse,
    MarketStatusResponse,
)


class LiveMarketService:
    """Service for fetching live market data with caching."""

    # TTL values in seconds
    QUOTE_TTL = 30
    OPTION_CHAIN_TTL = 60
    VIX_TTL = 30
    MARKET_STATUS_TTL = 60

    def __init__(self):
        self._quote_cache = TTLCache(default_ttl=self.QUOTE_TTL)
        self._chain_cache = TTLCache(default_ttl=self.OPTION_CHAIN_TTL)
        self._vix_cache = TTLCache(default_ttl=self.VIX_TTL)

    def get_quote(self, symbol: str) -> QuoteResponse:
        """
        Get stock quote with caching.

        Args:
            symbol: Ticker symbol (e.g., "SPY")

        Returns:
            Quote response with price data
        """
        cache_key = f"quote:{symbol.upper()}"
        cached, is_stale = self._quote_cache.get(cache_key)

        if cached and not is_stale:
            return cached

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get quote data
            price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
            prev_close = info.get("regularMarketPreviousClose", price)
            change = price - prev_close if prev_close else None
            change_pct = (change / prev_close * 100) if prev_close and change else None

            response = QuoteResponse(
                symbol=symbol.upper(),
                price=float(price),
                bid=float(info.get("bid", 0)) if info.get("bid") else None,
                ask=float(info.get("ask", 0)) if info.get("ask") else None,
                change=round(change, 2) if change else None,
                change_percent=round(change_pct, 2) if change_pct else None,
                volume=info.get("regularMarketVolume"),
                timestamp=datetime.now(),
                is_stale=False,
            )

            self._quote_cache.set(cache_key, response)
            return response

        except Exception:
            # Return stale cache if available
            if cached:
                cached.is_stale = True
                return cached
            raise

    def get_option_chain(
        self, symbol: str, expiry: Optional[str] = None
    ) -> OptionChainResponse:
        """
        Get option chain with caching.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date (YYYY-MM-DD). Uses nearest if not specified.

        Returns:
            Option chain response with calls and puts
        """
        cache_key = f"chain:{symbol.upper()}:{expiry or 'nearest'}"
        cached, is_stale = self._chain_cache.get(cache_key)

        if cached and not is_stale:
            return cached

        try:
            ticker = yf.Ticker(symbol)

            # Get available expirations
            available_expiries = list(ticker.options) if ticker.options else []

            if not available_expiries:
                raise ValueError(f"No options available for {symbol}")

            # Use specified expiry or nearest
            if expiry and expiry in available_expiries:
                selected_expiry = expiry
            else:
                selected_expiry = available_expiries[0]

            # Get option chain
            chain = ticker.option_chain(selected_expiry)

            # Get underlying price
            info = ticker.info
            underlying_price = info.get("regularMarketPrice") or info.get(
                "currentPrice", 0
            )

            # Convert calls
            calls = []
            for _, row in chain.calls.iterrows():
                calls.append(
                    OptionContractResponse(
                        strike=float(row["strike"]),
                        bid=_safe_float(row["bid"]),
                        ask=_safe_float(row["ask"]),
                        last_price=_safe_float(row["lastPrice"]),
                        volume=_safe_int(row["volume"]),
                        open_interest=_safe_int(row["openInterest"]),
                        implied_volatility=_safe_float(row["impliedVolatility"]),
                    )
                )

            # Convert puts
            puts = []
            for _, row in chain.puts.iterrows():
                puts.append(
                    OptionContractResponse(
                        strike=float(row["strike"]),
                        bid=_safe_float(row["bid"]),
                        ask=_safe_float(row["ask"]),
                        last_price=_safe_float(row["lastPrice"]),
                        volume=_safe_int(row["volume"]),
                        open_interest=_safe_int(row["openInterest"]),
                        implied_volatility=_safe_float(row["impliedVolatility"]),
                    )
                )

            response = OptionChainResponse(
                symbol=symbol.upper(),
                expiry=selected_expiry,
                underlying_price=float(underlying_price),
                risk_free_rate=0.043,  # Could fetch from ^TNX
                calls=calls,
                puts=puts,
                available_expiries=available_expiries,
                timestamp=datetime.now(),
                is_stale=False,
            )

            self._chain_cache.set(cache_key, response)
            return response

        except Exception:
            if cached:
                cached.is_stale = True
                return cached
            raise

    def get_vix(self) -> VixResponse:
        """
        Get VIX quote with caching.

        Returns:
            VIX response with current level
        """
        cache_key = "vix"
        cached, is_stale = self._vix_cache.get(cache_key)

        if cached and not is_stale:
            return cached

        try:
            ticker = yf.Ticker("^VIX")
            info = ticker.info

            level = info.get("regularMarketPrice") or info.get("currentPrice", 0)
            prev_close = info.get("regularMarketPreviousClose", level)
            change = level - prev_close if prev_close else None
            change_pct = (change / prev_close * 100) if prev_close and change else None

            response = VixResponse(
                level=float(level),
                change=round(change, 2) if change else None,
                change_percent=round(change_pct, 2) if change_pct else None,
                timestamp=datetime.now(),
                is_stale=False,
            )

            self._vix_cache.set(cache_key, response)
            return response

        except Exception:
            if cached:
                cached.is_stale = True
                return cached
            raise

    def get_market_status(self) -> MarketStatusResponse:
        """
        Get US stock market status.

        Returns:
            Market status response with open/closed status
        """
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)

        # Weekend
        if now.weekday() >= 5:
            return MarketStatusResponse(
                is_open=False,
                market_phase="closed",
                current_time=now,
                next_open=None,
                next_close=None,
            )

        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        pre_market = now.replace(hour=4, minute=0, second=0, microsecond=0)
        after_hours_end = now.replace(hour=20, minute=0, second=0, microsecond=0)

        if now < pre_market:
            phase = "closed"
            is_open = False
        elif now < market_open:
            phase = "pre"
            is_open = False
        elif now < market_close:
            phase = "regular"
            is_open = True
        elif now < after_hours_end:
            phase = "after"
            is_open = False
        else:
            phase = "closed"
            is_open = False

        return MarketStatusResponse(
            is_open=is_open,
            market_phase=phase,
            current_time=now,
            next_open=market_open if now < market_open else None,
            next_close=market_close if now < market_close else None,
        )

    def clear_cache(self) -> dict:
        """Clear all caches."""
        return {
            "quotes_cleared": self._quote_cache.clear(),
            "chains_cleared": self._chain_cache.clear(),
            "vix_cleared": self._vix_cache.clear(),
        }
