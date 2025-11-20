"""
Black-Scholes option pricing model and Greeks calculation.

Implements the classic Black-Scholes-Merton formula for European options.
"""

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from scipy import optimize
from scipy.stats import norm

from volatility_arbitrage.core.types import OptionType
from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Greeks:
    """
    Option Greeks - risk measures.

    All Greeks are dimensionless except Vega (sensitivity to 1% vol change).
    """

    delta: Decimal  # Price change per $1 underlying move
    gamma: Decimal  # Delta change per $1 underlying move
    theta: Decimal  # Price change per day
    vega: Decimal  # Price change per 1% volatility change
    rho: Decimal  # Price change per 1% interest rate change


class BlackScholesModel:
    """
    Black-Scholes option pricing model.

    Calculates theoretical option prices and Greeks for European options.
    """

    @staticmethod
    def _d1(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Calculate d1 parameter.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility

        Returns:
            d1 value
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def _d2(d1: float, sigma: float, T: float) -> float:
        """
        Calculate d2 parameter.

        Args:
            d1: d1 value
            sigma: Volatility
            T: Time to expiration (years)

        Returns:
            d2 value
        """
        if T <= 0:
            return d1

        return d1 - sigma * math.sqrt(T)

    @classmethod
    def price(
        cls,
        S: Decimal,
        K: Decimal,
        T: Decimal,
        r: Decimal,
        sigma: Decimal,
        option_type: OptionType,
    ) -> Decimal:
        """
        Calculate Black-Scholes option price.

        Args:
            S: Spot price of underlying
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: CALL or PUT

        Returns:
            Theoretical option price

        Example:
            >>> price = BlackScholesModel.price(
            ...     S=Decimal("100"),
            ...     K=Decimal("100"),
            ...     T=Decimal("1"),
            ...     r=Decimal("0.05"),
            ...     sigma=Decimal("0.2"),
            ...     option_type=OptionType.CALL
            ... )
        """
        # Convert to float for calculation
        S_f = float(S)
        K_f = float(K)
        T_f = float(T)
        r_f = float(r)
        sigma_f = float(sigma)

        # Handle edge cases
        if T_f <= 0:
            # At expiration
            if option_type == OptionType.CALL:
                return Decimal(str(max(S_f - K_f, 0)))
            else:
                return Decimal(str(max(K_f - S_f, 0)))

        if sigma_f <= 0:
            logger.warning("Volatility <= 0, using intrinsic value")
            if option_type == OptionType.CALL:
                return Decimal(str(max(S_f - K_f, 0)))
            else:
                return Decimal(str(max(K_f - S_f, 0)))

        # Calculate d1 and d2
        d1 = cls._d1(S_f, K_f, T_f, r_f, sigma_f)
        d2 = cls._d2(d1, sigma_f, T_f)

        # Calculate price
        if option_type == OptionType.CALL:
            price = S_f * norm.cdf(d1) - K_f * math.exp(-r_f * T_f) * norm.cdf(d2)
        else:  # PUT
            price = K_f * math.exp(-r_f * T_f) * norm.cdf(-d2) - S_f * norm.cdf(-d1)

        return Decimal(str(max(price, 0)))

    @classmethod
    def greeks(
        cls,
        S: Decimal,
        K: Decimal,
        T: Decimal,
        r: Decimal,
        sigma: Decimal,
        option_type: OptionType,
    ) -> Greeks:
        """
        Calculate option Greeks.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: CALL or PUT

        Returns:
            Greeks instance with all risk measures
        """
        # Convert to float
        S_f = float(S)
        K_f = float(K)
        T_f = float(T)
        r_f = float(r)
        sigma_f = float(sigma)

        # Handle edge cases
        if T_f <= 0 or sigma_f <= 0:
            return Greeks(
                delta=Decimal("0"),
                gamma=Decimal("0"),
                theta=Decimal("0"),
                vega=Decimal("0"),
                rho=Decimal("0"),
            )

        # Calculate d1, d2, and pdf
        d1 = cls._d1(S_f, K_f, T_f, r_f, sigma_f)
        d2 = cls._d2(d1, sigma_f, T_f)
        pdf_d1 = norm.pdf(d1)

        # Delta
        if option_type == OptionType.CALL:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = pdf_d1 / (S_f * sigma_f * math.sqrt(T_f))

        # Vega (same for calls and puts) - per 1% change in vol
        vega = (S_f * pdf_d1 * math.sqrt(T_f)) / 100

        # Theta (per day)
        if option_type == OptionType.CALL:
            theta = (
                -(S_f * pdf_d1 * sigma_f) / (2 * math.sqrt(T_f))
                - r_f * K_f * math.exp(-r_f * T_f) * norm.cdf(d2)
            ) / 365
        else:
            theta = (
                -(S_f * pdf_d1 * sigma_f) / (2 * math.sqrt(T_f))
                + r_f * K_f * math.exp(-r_f * T_f) * norm.cdf(-d2)
            ) / 365

        # Rho (per 1% change in rate)
        if option_type == OptionType.CALL:
            rho = (K_f * T_f * math.exp(-r_f * T_f) * norm.cdf(d2)) / 100
        else:
            rho = (-K_f * T_f * math.exp(-r_f * T_f) * norm.cdf(-d2)) / 100

        return Greeks(
            delta=Decimal(str(delta)),
            gamma=Decimal(str(gamma)),
            theta=Decimal(str(theta)),
            vega=Decimal(str(vega)),
            rho=Decimal(str(rho)),
        )
    
    @staticmethod
    def calculate_implied_volatility(market_price, S, K, T, r, option_type):
        """
        Calculate implied volatility from market price using Brent's method.

        Args:
            market_price: Observed market price (Decimal)
            S: Spot price (Decimal)
            K: Strike price (Decimal)
            T: Time to expiration in years (Decimal)
            r: Risk-free rate (Decimal)
            option_type: CALL or PUT

        Returns:
            Implied volatility (Decimal) or None if calculation fails

        Note:
            Returns None for:
            - Zero or negative market price
            - Expired options (T <= 0)
            - When numerical solver fails to converge
        """
        from scipy.optimize import brentq

        # Validate inputs
        if market_price <= 0:
            return None

        if T <= 0:
            return None

        # Objective function: difference between model price and market price
        def objective(vol):
            return float(BlackScholesModel.price(
                S, K, T, r, Decimal(str(vol)), option_type
            )) - float(market_price)

        try:
            # Use Brent's method to find volatility that matches market price
            # Search range: 0.0001% to 500% annualized volatility
            result = brentq(objective, 1e-6, 5.0)
            return Decimal(str(result))
        except ValueError:
            # Optimization failed (no solution in range, non-monotonic, etc.)
            return None