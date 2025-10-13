"""
Unit tests for Black-Scholes pricing model.
"""

from decimal import Decimal

import pytest

from volatility_arbitrage.core.types import OptionType
from volatility_arbitrage.models.black_scholes import (
    BlackScholesModel,
    Greeks,
    calculate_implied_volatility,
)


@pytest.mark.unit
class TestBlackScholesModel:
    """Tests for Black-Scholes pricing model."""

    def test_atm_call_price(self):
        """Test ATM call option pricing."""
        price = BlackScholesModel.price(
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.CALL,
        )

        # ATM call with 1 year to expiry should be around $10-12
        assert Decimal("8") < price < Decimal("15")

    def test_atm_put_price(self):
        """Test ATM put option pricing."""
        price = BlackScholesModel.price(
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.PUT,
        )

        # ATM put should be positive
        assert price > 0

    def test_itm_call_price(self):
        """Test in-the-money call pricing."""
        price = BlackScholesModel.price(
            S=Decimal("110"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.CALL,
        )

        # ITM call should be worth at least intrinsic value
        intrinsic = Decimal("10")
        assert price >= intrinsic

    def test_otm_put_price(self):
        """Test out-of-the-money put pricing."""
        price = BlackScholesModel.price(
            S=Decimal("110"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.PUT,
        )

        # OTM put should have time value
        assert price > 0
        assert price < Decimal("10")  # Less than if it were ITM

    def test_zero_time_to_expiry_call(self):
        """Test call at expiration."""
        price = BlackScholesModel.price(
            S=Decimal("110"),
            K=Decimal("100"),
            T=Decimal("0"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.CALL,
        )

        # Should equal intrinsic value
        assert price == Decimal("10")

    def test_zero_time_to_expiry_put(self):
        """Test put at expiration."""
        price = BlackScholesModel.price(
            S=Decimal("90"),
            K=Decimal("100"),
            T=Decimal("0"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.PUT,
        )

        # Should equal intrinsic value
        assert price == Decimal("10")

    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        S = Decimal("100")
        K = Decimal("100")
        T = Decimal("1")
        r = Decimal("0.05")
        sigma = Decimal("0.2")

        call_price = BlackScholesModel.price(S, K, T, r, sigma, OptionType.CALL)
        put_price = BlackScholesModel.price(S, K, T, r, sigma, OptionType.PUT)

        # Put-Call Parity: C - P = S - K*e^(-rT)
        import math

        pv_strike = K * Decimal(str(math.exp(-float(r) * float(T))))
        left_side = call_price - put_price
        right_side = S - pv_strike

        # Should be approximately equal
        assert abs(left_side - right_side) < Decimal("0.01")


@pytest.mark.unit
class TestGreeks:
    """Tests for Greeks calculation."""

    def test_call_delta_range(self):
        """Test that call delta is between 0 and 1."""
        greeks = BlackScholesModel.greeks(
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.CALL,
        )

        assert Decimal("0") <= greeks.delta <= Decimal("1")

    def test_put_delta_range(self):
        """Test that put delta is between -1 and 0."""
        greeks = BlackScholesModel.greeks(
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.PUT,
        )

        assert Decimal("-1") <= greeks.delta <= Decimal("0")

    def test_gamma_positive(self):
        """Test that gamma is always positive."""
        greeks = BlackScholesModel.greeks(
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.CALL,
        )

        assert greeks.gamma > 0

    def test_vega_positive(self):
        """Test that vega is always positive."""
        greeks = BlackScholesModel.greeks(
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.CALL,
        )

        assert greeks.vega > 0

    def test_call_theta_negative(self):
        """Test that call theta is typically negative."""
        greeks = BlackScholesModel.greeks(
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.CALL,
        )

        # ATM call theta is usually negative (time decay)
        assert greeks.theta < 0

    def test_atm_delta_approximately_half(self):
        """Test that ATM call delta is approximately 0.5 when r=0."""
        # When r=0, ATM call delta is close to 0.5 (affected by vol term)
        greeks = BlackScholesModel.greeks(
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0"),  # Risk-free rate = 0
            sigma=Decimal("0.2"),
            option_type=OptionType.CALL,
        )

        # Delta ≈ 0.54 due to sigma²/2 term in d1 formula
        assert abs(greeks.delta - Decimal("0.5")) < Decimal("0.05")

    def test_atm_delta_with_positive_rate(self):
        """Test that ATM call delta > 0.5 when r > 0."""
        # With positive r, forward is above spot, so ATM call delta > 0.5
        greeks = BlackScholesModel.greeks(
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            sigma=Decimal("0.2"),
            option_type=OptionType.CALL,
        )

        # Delta should be around 0.6368 (N(0.35))
        assert Decimal("0.63") < greeks.delta < Decimal("0.64")


@pytest.mark.unit
class TestImpliedVolatility:
    """Tests for implied volatility calculation."""

    def test_implied_vol_recovery(self):
        """Test that IV calculation recovers known volatility."""
        S = Decimal("100")
        K = Decimal("100")
        T = Decimal("1")
        r = Decimal("0.05")
        sigma = Decimal("0.25")

        # Calculate price with known vol
        price = BlackScholesModel.price(S, K, T, r, sigma, OptionType.CALL)

        # Recover implied vol
        iv = calculate_implied_volatility(price, S, K, T, r, OptionType.CALL)

        # Should recover original volatility
        assert iv is not None
        assert abs(iv - sigma) < Decimal("0.01")

    def test_implied_vol_itm_call(self):
        """Test IV calculation for ITM call."""
        S = Decimal("110")
        K = Decimal("100")
        T = Decimal("1")
        r = Decimal("0.05")
        sigma = Decimal("0.20")

        price = BlackScholesModel.price(S, K, T, r, sigma, OptionType.CALL)
        iv = calculate_implied_volatility(price, S, K, T, r, OptionType.CALL)

        assert iv is not None
        assert abs(iv - sigma) < Decimal("0.01")

    def test_implied_vol_zero_price(self):
        """Test that zero price returns None."""
        iv = calculate_implied_volatility(
            market_price=Decimal("0"),
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("1"),
            r=Decimal("0.05"),
            option_type=OptionType.CALL,
        )

        assert iv is None

    def test_implied_vol_expired_option(self):
        """Test that expired option returns None."""
        iv = calculate_implied_volatility(
            market_price=Decimal("10"),
            S=Decimal("100"),
            K=Decimal("100"),
            T=Decimal("0"),
            r=Decimal("0.05"),
            option_type=OptionType.CALL,
        )

        assert iv is None
