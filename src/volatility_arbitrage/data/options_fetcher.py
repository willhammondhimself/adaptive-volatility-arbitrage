"""
Enhanced options chain fetching with implied volatility extraction.

Extends YahooFinanceFetcher to provide comprehensive options data including
IV calculation, data quality validation, and liquidity filtering.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd

from volatility_arbitrage.core.types import OptionChain, OptionContract, OptionType
from volatility_arbitrage.data.yahoo import YahooFinanceFetcher
from volatility_arbitrage.models.black_scholes import calculate_implied_volatility
from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


class OptionsDataQuality:
    """
    Data quality metrics for option chains.

    Tracks issues and provides quality score.
    """

    def __init__(self) -> None:
        """Initialize quality tracker."""
        self.total_options = 0
        self.missing_bid_ask = 0
        self.wide_spreads = 0
        self.low_volume = 0
        self.put_call_parity_violations = 0
        self.failed_iv_calculations = 0

    @property
    def quality_score(self) -> float:
        """
        Calculate overall quality score (0-1).

        Returns:
            Quality score where 1.0 is perfect quality
        """
        if self.total_options == 0:
            return 0.0

        issues = (
            self.missing_bid_ask
            + self.wide_spreads
            + self.low_volume
            + self.put_call_parity_violations
            + self.failed_iv_calculations
        )

        return max(0.0, 1.0 - (issues / self.total_options))

    def to_dict(self) -> dict:
        """Export quality metrics as dictionary."""
        return {
            "total_options": self.total_options,
            "missing_bid_ask": self.missing_bid_ask,
            "wide_spreads": self.wide_spreads,
            "low_volume": self.low_volume,
            "put_call_parity_violations": self.put_call_parity_violations,
            "failed_iv_calculations": self.failed_iv_calculations,
            "quality_score": self.quality_score,
        }


class YahooOptionsChainFetcher(YahooFinanceFetcher):
    """
    Enhanced Yahoo Finance fetcher for options data.

    Extends base fetcher with:
    - Implied volatility calculation
    - Data quality validation
    - Liquidity filtering
    - Put-call parity checks
    """

    def __init__(
        self,
        cache: bool = True,
        min_volume: int = 10,
        min_open_interest: int = 50,
        max_spread_pct: float = 0.20,  # 20% max bid-ask spread
    ) -> None:
        """
        Initialize enhanced options fetcher.

        Args:
            cache: Whether to cache data
            min_volume: Minimum daily volume filter
            min_open_interest: Minimum open interest filter
            max_spread_pct: Maximum bid-ask spread as % of mid price
        """
        super().__init__(cache=cache)
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
        self.max_spread_pct = max_spread_pct

        logger.info(
            "Initialized YahooOptionsChainFetcher",
            extra={
                "min_volume": min_volume,
                "min_open_interest": min_open_interest,
                "max_spread_pct": max_spread_pct,
            },
        )

    def fetch_option_chain_with_iv(
        self,
        symbol: str,
        timestamp: datetime,
        expiry: Optional[datetime] = None,
        calculate_iv: bool = True,
    ) -> tuple[OptionChain, OptionsDataQuality]:
        """
        Fetch option chain and calculate implied volatility.

        Args:
            symbol: Underlying symbol
            timestamp: Current timestamp
            expiry: Target expiration (None for nearest)
            calculate_iv: Whether to calculate IV from market prices

        Returns:
            Tuple of (OptionChain with IVs, DataQuality metrics)

        Raises:
            DataNotFoundError: If no options available
            DataFetcherError: For API errors
        """
        # Fetch base option chain
        chain = self.fetch_option_chain(symbol, timestamp, expiry)

        quality = OptionsDataQuality()
        quality.total_options = len(chain.calls) + len(chain.puts)

        if not calculate_iv:
            return chain, quality

        # Calculate IV for all options
        updated_calls = []
        for option in chain.calls:
            updated_option, option_quality = self._calculate_option_iv(
                option, chain.underlying_price, chain.time_to_expiry, chain.risk_free_rate
            )
            updated_calls.append(updated_option)

            # Update quality metrics
            if option_quality.get("missing_bid_ask"):
                quality.missing_bid_ask += 1
            if option_quality.get("wide_spread"):
                quality.wide_spreads += 1
            if option_quality.get("low_volume"):
                quality.low_volume += 1
            if option_quality.get("failed_iv"):
                quality.failed_iv_calculations += 1

        updated_puts = []
        for option in chain.puts:
            updated_option, option_quality = self._calculate_option_iv(
                option, chain.underlying_price, chain.time_to_expiry, chain.risk_free_rate
            )
            updated_puts.append(updated_option)

            # Update quality metrics
            if option_quality.get("missing_bid_ask"):
                quality.missing_bid_ask += 1
            if option_quality.get("wide_spread"):
                quality.wide_spreads += 1
            if option_quality.get("low_volume"):
                quality.low_volume += 1
            if option_quality.get("failed_iv"):
                quality.failed_iv_calculations += 1

        # Check put-call parity
        parity_violations = self._check_put_call_parity(
            updated_calls, updated_puts, chain.underlying_price, chain.risk_free_rate, chain.time_to_expiry
        )
        quality.put_call_parity_violations = parity_violations

        # Create updated chain
        updated_chain = OptionChain(
            symbol=chain.symbol,
            timestamp=chain.timestamp,
            expiry=chain.expiry,
            underlying_price=chain.underlying_price,
            calls=updated_calls,
            puts=updated_puts,
            risk_free_rate=chain.risk_free_rate,
        )

        logger.info(
            f"Fetched option chain with IVs for {symbol}",
            extra={
                "symbol": symbol,
                "calls": len(updated_calls),
                "puts": len(updated_puts),
                "quality_score": quality.quality_score,
            },
        )

        return updated_chain, quality

    def _calculate_option_iv(
        self,
        option: OptionContract,
        underlying_price: Decimal,
        time_to_expiry: Decimal,
        risk_free_rate: Decimal,
    ) -> tuple[OptionContract, dict]:
        """
        Calculate implied volatility for an option.

        Args:
            option: Option contract
            underlying_price: Current underlying price
            time_to_expiry: Time to expiration (years)
            risk_free_rate: Risk-free rate

        Returns:
            Tuple of (updated OptionContract with IV, quality dict)
        """
        quality = {
            "missing_bid_ask": False,
            "wide_spread": False,
            "low_volume": False,
            "failed_iv": False,
        }

        # Check for missing bid/ask
        if option.bid is None or option.ask is None:
            quality["missing_bid_ask"] = True
            return option, quality

        # Calculate mid price
        mid_price = option.mid_price

        # Check spread
        spread_pct = (option.ask - option.bid) / mid_price if mid_price > 0 else Decimal("1")
        if spread_pct > Decimal(str(self.max_spread_pct)):
            quality["wide_spread"] = True

        # Check volume
        if option.volume < self.min_volume:
            quality["low_volume"] = True

        # Calculate IV
        try:
            iv = calculate_implied_volatility(
                market_price=mid_price,
                S=underlying_price,
                K=option.strike,
                T=time_to_expiry,
                r=risk_free_rate,
                option_type=option.option_type,
                initial_guess=0.25,
            )

            if iv is not None:
                # Create updated option with IV
                updated_option = OptionContract(
                    symbol=option.symbol,
                    option_type=option.option_type,
                    strike=option.strike,
                    expiry=option.expiry,
                    price=option.price,
                    bid=option.bid,
                    ask=option.ask,
                    volume=option.volume,
                    open_interest=option.open_interest,
                    implied_volatility=iv,
                )
                return updated_option, quality
            else:
                quality["failed_iv"] = True
                return option, quality

        except Exception as e:
            logger.warning(
                f"IV calculation failed for {option.symbol} {option.strike}",
                extra={"error": str(e)},
            )
            quality["failed_iv"] = True
            return option, quality

    def _check_put_call_parity(
        self,
        calls: list[OptionContract],
        puts: list[OptionContract],
        underlying_price: Decimal,
        risk_free_rate: Decimal,
        time_to_expiry: Decimal,
        tolerance: Decimal = Decimal("0.05"),  # 5% tolerance
    ) -> int:
        """
        Check put-call parity violations.

        Put-Call Parity: C - P = S - K*e^(-rT)

        Args:
            calls: Call options
            puts: Put options
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate
            time_to_expiry: Time to expiration
            tolerance: Acceptable deviation as fraction

        Returns:
            Number of parity violations
        """
        violations = 0

        # Match calls and puts by strike
        call_dict = {c.strike: c for c in calls}
        put_dict = {p.strike: p for p in puts}

        common_strikes = set(call_dict.keys()) & set(put_dict.keys())

        import math

        for strike in common_strikes:
            call = call_dict[strike]
            put = put_dict[strike]

            # Skip if no IV calculated
            if call.implied_volatility is None or put.implied_volatility is None:
                continue

            # Calculate both sides of parity
            left_side = call.mid_price - put.mid_price

            import math

            pv_strike = strike * Decimal(str(math.exp(-float(risk_free_rate) * float(time_to_expiry))))
            right_side = underlying_price - pv_strike

            # Check if within tolerance
            deviation = abs(left_side - right_side) / abs(right_side) if right_side != 0 else Decimal("1")

            if deviation > tolerance:
                violations += 1
                logger.debug(
                    f"Put-call parity violation at strike {strike}",
                    extra={
                        "strike": float(strike),
                        "deviation": float(deviation),
                        "left_side": float(left_side),
                        "right_side": float(right_side),
                    },
                )

        return violations

    def filter_liquid_options(
        self,
        chain: OptionChain,
    ) -> OptionChain:
        """
        Filter option chain for liquid options only.

        Args:
            chain: Option chain to filter

        Returns:
            Filtered option chain
        """
        filtered_calls = [
            opt
            for opt in chain.calls
            if opt.volume >= self.min_volume and opt.open_interest >= self.min_open_interest
        ]

        filtered_puts = [
            opt
            for opt in chain.puts
            if opt.volume >= self.min_volume and opt.open_interest >= self.min_open_interest
        ]

        logger.info(
            f"Filtered option chain for {chain.symbol}",
            extra={
                "original_calls": len(chain.calls),
                "filtered_calls": len(filtered_calls),
                "original_puts": len(chain.puts),
                "filtered_puts": len(filtered_puts),
            },
        )

        return OptionChain(
            symbol=chain.symbol,
            timestamp=chain.timestamp,
            expiry=chain.expiry,
            underlying_price=chain.underlying_price,
            calls=filtered_calls,
            puts=filtered_puts,
            risk_free_rate=chain.risk_free_rate,
        )

    def get_atm_options(
        self,
        chain: OptionChain,
        num_strikes: int = 5,
    ) -> tuple[list[OptionContract], list[OptionContract]]:
        """
        Get ATM (at-the-money) options centered around underlying price.

        Args:
            chain: Option chain
            num_strikes: Number of strikes to return on each side

        Returns:
            Tuple of (ATM calls, ATM puts)
        """
        atm_strike = chain.get_atm_strike()
        if atm_strike is None:
            return [], []

        # Sort by strike
        sorted_calls = sorted(chain.calls, key=lambda x: x.strike)
        sorted_puts = sorted(chain.puts, key=lambda x: x.strike)

        # Find ATM index
        call_idx = next((i for i, c in enumerate(sorted_calls) if c.strike >= atm_strike), 0)
        put_idx = next((i for i, p in enumerate(sorted_puts) if p.strike >= atm_strike), 0)

        # Get surrounding strikes
        start_call = max(0, call_idx - num_strikes // 2)
        end_call = min(len(sorted_calls), start_call + num_strikes)
        atm_calls = sorted_calls[start_call:end_call]

        start_put = max(0, put_idx - num_strikes // 2)
        end_put = min(len(sorted_puts), start_put + num_strikes)
        atm_puts = sorted_puts[start_put:end_put]

        return atm_calls, atm_puts
