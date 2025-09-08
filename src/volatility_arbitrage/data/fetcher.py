"""
Abstract data fetcher interface.

Defines the contract for data sources to implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Optional

import pandas as pd

from volatility_arbitrage.core.types import TickData, OptionChain


class DataFetcherError(Exception):
    """Base exception for data fetching errors."""

    pass


class DataNotFoundError(DataFetcherError):
    """Raised when requested data is not available."""

    pass


class DataFetcher(ABC):
    """
    Abstract base class for data fetchers.

    All data sources must implement this interface to be used with the backtest engine.
    """

    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a symbol.

        Args:
            symbol: Ticker symbol
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            DataNotFoundError: If data is not available
            DataFetcherError: For other fetching errors
        """
        pass

    @abstractmethod
    def fetch_option_chain(
        self,
        symbol: str,
        timestamp: datetime,
        expiry: Optional[datetime] = None,
    ) -> OptionChain:
        """
        Fetch option chain for a symbol.

        Args:
            symbol: Underlying ticker symbol
            timestamp: Time to fetch chain for
            expiry: Specific expiration date (if None, fetches nearest)

        Returns:
            OptionChain with calls and puts

        Raises:
            DataNotFoundError: If option chain is not available
            DataFetcherError: For other fetching errors
        """
        pass

    @abstractmethod
    def fetch_current_price(self, symbol: str) -> Decimal:
        """
        Fetch current price for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Current price

        Raises:
            DataNotFoundError: If price is not available
            DataFetcherError: For other fetching errors
        """
        pass

    @abstractmethod
    def fetch_risk_free_rate(self, date: datetime) -> Decimal:
        """
        Fetch risk-free rate for a given date.

        Args:
            date: Date to fetch rate for

        Returns:
            Annual risk-free rate (e.g., 0.05 for 5%)

        Raises:
            DataFetcherError: For fetching errors
        """
        pass

    def get_tick_data(
        self,
        symbol: str,
        timestamp: datetime,
        price: Decimal,
        volume: int = 0,
    ) -> TickData:
        """
        Create TickData from fetched information.

        Helper method to construct TickData objects consistently.

        Args:
            symbol: Ticker symbol
            timestamp: Timestamp for the tick
            price: Price at timestamp
            volume: Trading volume

        Returns:
            TickData instance
        """
        return TickData(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            volume=volume,
        )
