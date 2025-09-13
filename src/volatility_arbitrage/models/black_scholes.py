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


def calculate_implied_volatility(
    market_price: Decimal,
    S: Decimal,
    K: Decimal,
    T: Decimal,
    r: Decimal,
    option_type: OptionType,
    initial_guess: float = 0.3,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Optional[Decimal]:
    """
    Calculate implied volatility using Newton-Raphson method.

    Args:
        market_price: Observed market price
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        option_type: CALL or PUT
        initial_guess: Starting volatility guess
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Implied volatility or None if convergence fails

    Example:
        >>> iv = calculate_implied_volatility(
        ...     market_price=Decimal("10.5"),
        ...     S=Decimal("100"),
        ...     K=Decimal("100"),
        ...     T=Decimal("1"),
        ...     r=Decimal("0.05"),
        ...     option_type=OptionType.CALL
        ... )
    """
    if T <= 0:
        logger.warning("Cannot calculate IV for expired option")
        return None

    if market_price <= 0:
        logger.warning("Market price <= 0, cannot calculate IV")
        return None

    def objective(sigma: float) -> float:
        """Objective function: difference between BS price and market price."""
        if sigma <= 0:
            return float("inf")

        bs_price = BlackScholesModel.price(
            S, K, T, r, Decimal(str(sigma)), option_type
        )
        return float(bs_price) - float(market_price)

    def vega_func(sigma: float) -> float:
        """Derivative of price with respect to sigma (vega)."""
        if sigma <= 0:
            return 1e-10

        greeks = BlackScholesModel.greeks(
            S, K, T, r, Decimal(str(sigma)), option_type
        )
        # Convert vega from per 1% to per 1 unit
        return float(greeks.vega) * 100

    try:
        # Use Newton-Raphson method via scipy
        result = optimize.newton(
            objective,
            x0=initial_guess,
            fprime=vega_func,
            maxiter=max_iterations,
            tol=tolerance,
        )

        if result > 0 and result < 5.0:  # Sanity check: IV between 0% and 500%
            return Decimal(str(result))
        else:
            logger.warning(
                f"IV out of reasonable range: {result}",
                extra={"iv": result}
            )
            return None

    except (RuntimeError, ValueError) as e:
        logger.warning(
            f"IV calculation failed: {e}",
            extra={
                "market_price": float(market_price),
                "S": float(S),
                "K": float(K),
                "T": float(T),
                "error": str(e),
            },
        )
        return None
