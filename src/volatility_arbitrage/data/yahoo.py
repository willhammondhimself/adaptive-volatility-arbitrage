"""
Yahoo Finance data fetcher implementation.

Fetches historical data and option chains using yfinance library.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd
import yfinance as yf

from volatility_arbitrage.core.types import OptionChain, OptionContract, OptionType
from volatility_arbitrage.data.fetcher import (
    DataFetcher,
    DataFetcherError,
    DataNotFoundError,
)
from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


class YahooFinanceFetcher(DataFetcher):
    """
    Yahoo Finance data fetcher.

    Implements DataFetcher interface using yfinance library.
    """

    def __init__(self, cache: bool = True) -> None:
        """
        Initialize Yahoo Finance fetcher.

        Args:
            cache: Whether to cache downloaded data
        """
        self.cache = cache
        self._cached_tickers: dict[str, yf.Ticker] = {}
        logger.info("Initialized YahooFinanceFetcher", extra={"cache": cache})

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """
        Get or create ticker object.

        Args:
            symbol: Ticker symbol

        Returns:
            yfinance Ticker object
        """
        symbol = symbol.upper()
        if self.cache and symbol in self._cached_tickers:
            return self._cached_tickers[symbol]

        ticker = yf.Ticker(symbol)
        if self.cache:
            self._cached_tickers[symbol] = ticker

        return ticker

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            DataNotFoundError: If no data available
            DataFetcherError: For API or network errors
        """
        try:
            logger.info(
                f"Fetching historical data for {symbol}",
                extra={
                    "symbol": symbol,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )

            ticker = self._get_ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise DataNotFoundError(
                    f"No historical data found for {symbol} "
                    f"between {start_date} and {end_date}"
                )

            # Standardize column names
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Rename date column to timestamp
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})

            # Select and order columns
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            df = df[required_cols]

            logger.info(
                f"Fetched {len(df)} rows for {symbol}",
                extra={"symbol": symbol, "rows": len(df)},
            )

            return df

        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error fetching historical data: {e}",
                extra={"symbol": symbol, "error": str(e)},
            )
            raise DataFetcherError(f"Failed to fetch data for {symbol}: {e}") from e

    def fetch_option_chain(
        self,
        symbol: str,
        timestamp: datetime,
        expiry: Optional[datetime] = None,
    ) -> OptionChain:
        """
        Fetch option chain from Yahoo Finance.

        Args:
            symbol: Underlying symbol
            timestamp: Current timestamp
            expiry: Target expiration (if None, uses nearest available)

        Returns:
            OptionChain with calls and puts

        Raises:
            DataNotFoundError: If no options available
            DataFetcherError: For API errors
        """
        try:
            logger.info(
                f"Fetching option chain for {symbol}",
                extra={"symbol": symbol, "expiry": expiry.isoformat() if expiry else None},
            )

            ticker = self._get_ticker(symbol)

            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                raise DataNotFoundError(f"No options available for {symbol}")

            # Select expiration date
            if expiry is None:
                # Use nearest expiration
                expiry_str = expirations[0]
            else:
                # Find closest matching expiration
                expiry_date = expiry.strftime("%Y-%m-%d")
                if expiry_date in expirations:
                    expiry_str = expiry_date
                else:
                    # Find nearest
                    expiry_dates = [datetime.strptime(d, "%Y-%m-%d") for d in expirations]
                    nearest = min(expiry_dates, key=lambda x: abs((x - expiry).days))
                    expiry_str = nearest.strftime("%Y-%m-%d")

            # Fetch option chain
            opt_chain = ticker.option_chain(expiry_str)

            # Get current underlying price
            current_price = self.fetch_current_price(symbol)

            # Parse calls
            calls = []
            if not opt_chain.calls.empty:
                for _, row in opt_chain.calls.iterrows():
                    try:
                        contract = OptionContract(
                            symbol=symbol,
                            option_type=OptionType.CALL,
                            strike=Decimal(str(row["strike"])),
                            expiry=datetime.strptime(expiry_str, "%Y-%m-%d"),
                            price=Decimal(str(row.get("lastPrice", 0))),
                            bid=Decimal(str(row.get("bid", 0)))
                            if pd.notna(row.get("bid"))
                            else None,
                            ask=Decimal(str(row.get("ask", 0)))
                            if pd.notna(row.get("ask"))
                            else None,
                            volume=int(row.get("volume", 0))
                            if pd.notna(row.get("volume"))
                            else 0,
                            open_interest=int(row.get("openInterest", 0))
                            if pd.notna(row.get("openInterest"))
                            else 0,
                            implied_volatility=Decimal(str(row.get("impliedVolatility", 0)))
                            if pd.notna(row.get("impliedVolatility"))
                            else None,
                        )
                        calls.append(contract)
                    except Exception as e:
                        logger.warning(
                            f"Skipping invalid call option: {e}",
                            extra={"strike": row.get("strike"), "error": str(e)},
                        )

            # Parse puts
            puts = []
            if not opt_chain.puts.empty:
                for _, row in opt_chain.puts.iterrows():
                    try:
                        contract = OptionContract(
                            symbol=symbol,
                            option_type=OptionType.PUT,
                            strike=Decimal(str(row["strike"])),
                            expiry=datetime.strptime(expiry_str, "%Y-%m-%d"),
                            price=Decimal(str(row.get("lastPrice", 0))),
                            bid=Decimal(str(row.get("bid", 0)))
                            if pd.notna(row.get("bid"))
                            else None,
                            ask=Decimal(str(row.get("ask", 0)))
                            if pd.notna(row.get("ask"))
                            else None,
                            volume=int(row.get("volume", 0))
                            if pd.notna(row.get("volume"))
                            else 0,
                            open_interest=int(row.get("openInterest", 0))
                            if pd.notna(row.get("openInterest"))
                            else 0,
                            implied_volatility=Decimal(str(row.get("impliedVolatility", 0)))
                            if pd.notna(row.get("impliedVolatility"))
                            else None,
                        )
                        puts.append(contract)
                    except Exception as e:
                        logger.warning(
                            f"Skipping invalid put option: {e}",
                            extra={"strike": row.get("strike"), "error": str(e)},
                        )

            # Get risk-free rate
            risk_free_rate = self.fetch_risk_free_rate(timestamp)

            chain = OptionChain(
                symbol=symbol,
                timestamp=timestamp,
                expiry=datetime.strptime(expiry_str, "%Y-%m-%d"),
                underlying_price=current_price,
                calls=calls,
                puts=puts,
                risk_free_rate=risk_free_rate,
            )

            logger.info(
                f"Fetched option chain for {symbol}",
                extra={
                    "symbol": symbol,
                    "calls": len(calls),
                    "puts": len(puts),
                    "expiry": expiry_str,
                },
            )

            return chain

        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error fetching option chain: {e}",
                extra={"symbol": symbol, "error": str(e)},
            )
            raise DataFetcherError(f"Failed to fetch option chain for {symbol}: {e}") from e

    def fetch_current_price(self, symbol: str) -> Decimal:
        """
        Fetch current price for symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Current price

        Raises:
            DataNotFoundError: If price not available
        """
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info

            # Try multiple price fields
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            if price is None:
                # Fallback: get latest close from history
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]

            if price is None:
                raise DataNotFoundError(f"No price available for {symbol}")

            return Decimal(str(price))

        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error fetching current price: {e}",
                extra={"symbol": symbol, "error": str(e)},
            )
            raise DataFetcherError(f"Failed to fetch price for {symbol}: {e}") from e

    def fetch_risk_free_rate(self, date: datetime) -> Decimal:
        """
        Fetch risk-free rate (using 10-year Treasury yield as proxy).

        Args:
            date: Date to fetch rate for

        Returns:
            Annual risk-free rate

        Note:
            Uses ^TNX (10-year Treasury) as proxy for risk-free rate.
            Falls back to default 5% if not available.
        """
        try:
            # Fetch 10-year Treasury yield (^TNX)
            ticker = yf.Ticker("^TNX")
            hist = ticker.history(start=date - timedelta(days=7), end=date + timedelta(days=1))

            if not hist.empty:
                # Treasury yield is already in percentage, convert to decimal
                rate = Decimal(str(hist["Close"].iloc[-1])) / Decimal("100")
                logger.debug(f"Fetched risk-free rate: {rate}", extra={"rate": float(rate)})
                return rate

            # Fallback to default
            logger.warning("Could not fetch risk-free rate, using default 5%")
            return Decimal("0.05")

        except Exception as e:
            logger.warning(
                f"Error fetching risk-free rate, using default: {e}",
                extra={"error": str(e)},
            )
            return Decimal("0.05")
