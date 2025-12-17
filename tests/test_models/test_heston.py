"""
Tests for Heston stochastic volatility model.
"""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from volatility_arbitrage.core.types import OptionType
from volatility_arbitrage.models.heston import (
    HestonCalibrator,
    HestonModel,
    HestonParameters,
    compare_to_black_scholes,
)


@pytest.mark.unit
class TestHestonParameters:
    """Tests for HestonParameters validation."""

    def test_valid_parameters(self):
        """Test creating valid Heston parameters."""
        params = HestonParameters(
            v0=Decimal("0.04"),
            theta=Decimal("0.04"),
            kappa=Decimal("2.0"),
            xi=Decimal("0.5"),
            rho=Decimal("-0.7"),
        )

        assert params.v0 == Decimal("0.04")
        assert params.theta == Decimal("0.04")
        assert params.kappa == Decimal("2.0")
        assert params.xi == Decimal("0.5")
        assert params.rho == Decimal("-0.7")

    def test_negative_variance_fails(self):
        """Test that negative variance raises error."""
        with pytest.raises(ValueError, match="v0 must be positive"):
            HestonParameters(
                v0=Decimal("-0.04"),
                theta=Decimal("0.04"),
                kappa=Decimal("2.0"),
                xi=Decimal("0.5"),
                rho=Decimal("-0.7"),
            )

    def test_invalid_correlation_fails(self):
        """Test that invalid correlation raises error."""
        with pytest.raises(ValueError, match="rho must be in"):
            HestonParameters(
                v0=Decimal("0.04"),
                theta=Decimal("0.04"),
                kappa=Decimal("2.0"),
                xi=Decimal("0.5"),
                rho=Decimal("1.5"),  # Invalid: > 1
            )

    def test_feller_condition_warning(self, caplog):
        """Test Feller condition violation warning."""
        # Violate Feller: 2κθ > ξ²
        # 2 * 0.5 * 0.01 = 0.01
        # 1.0² = 1.0
        # 0.01 < 1.0 → violation
        params = HestonParameters(
            v0=Decimal("0.04"),
            theta=Decimal("0.01"),  # Low long-term variance
            kappa=Decimal("0.5"),  # Low mean reversion
            xi=Decimal("1.0"),  # High vol of vol
            rho=Decimal("-0.5"),
        )

        assert "Feller condition violated" in caplog.text

    def test_to_dict(self):
        """Test parameter serialization."""
        params = HestonParameters(
            v0=Decimal("0.04"),
            theta=Decimal("0.04"),
            kappa=Decimal("2.0"),
            xi=Decimal("0.5"),
            rho=Decimal("-0.7"),
        )

        d = params.to_dict()

        assert d["v0"] == 0.04
        assert d["theta"] == 0.04
        assert d["kappa"] == 2.0
        assert d["xi"] == 0.5
        assert d["rho"] == -0.7

    def test_from_dict(self):
        """Test parameter deserialization."""
        params_dict = {
            "v0": 0.04,
            "theta": 0.04,
            "kappa": 2.0,
            "xi": 0.5,
            "rho": -0.7,
        }

        params = HestonParameters.from_dict(params_dict)

        assert params.v0 == Decimal("0.04")
        assert params.rho == Decimal("-0.7")


@pytest.mark.unit
class TestHestonModel:
    """Tests for Heston model pricing."""

    @pytest.fixture
    def standard_params(self):
        """Standard Heston parameters for testing."""
        return HestonParameters(
            v0=Decimal("0.04"),  # 20% initial vol
            theta=Decimal("0.04"),  # 20% long-term vol
            kappa=Decimal("2.0"),
            xi=Decimal("0.3"),
            rho=Decimal("-0.7"),
        )

    @pytest.fixture
    def heston_model(self, standard_params):
        """Heston model with standard parameters."""
        return HestonModel(standard_params)

    def test_characteristic_function(self, heston_model):
        """Test characteristic function calculation."""
        u = np.array([0.0, 1.0, 2.0, 5.0])
        S = 100.0
        T = 1.0
        r = 0.05

        cf = heston_model.characteristic_function(u, S, T, r)

        # Should return complex values
        assert len(cf) == len(u)
        assert all(isinstance(x, (complex, np.complex128)) for x in cf)

        # At u=0, should equal forward price (approximately)
        assert abs(cf[0] - np.exp(r * T + np.log(S))) < 0.01

    def test_atm_call_pricing(self, heston_model):
        """Test ATM call option pricing."""
        S = Decimal("100")
        K = Decimal("100")  # ATM
        T = Decimal("1.0")
        r = Decimal("0.05")

        price = heston_model.price(S, K, T, r, OptionType.CALL)

        # ATM call with 20% vol should be approximately $10-15
        assert Decimal("8") < price < Decimal("20")

    def test_put_call_parity(self, heston_model):
        """Test put-call parity relationship."""
        S = Decimal("100")
        K = Decimal("100")
        T = Decimal("1.0")
        r = Decimal("0.05")

        call_price = heston_model.price(S, K, T, r, OptionType.CALL)
        put_price = heston_model.price(S, K, T, r, OptionType.PUT)

        # Put-call parity: C - P = S - K*e^(-rT)
        parity_diff = call_price - put_price
        expected_diff = S - K * Decimal(str(np.exp(-float(r) * float(T))))

        # Allow small numerical error
        assert abs(parity_diff - expected_diff) < Decimal("0.50")

    def test_deep_otm_option_near_zero(self, heston_model):
        """Test that deep OTM options are near zero."""
        S = Decimal("100")
        K = Decimal("200")  # Deep OTM call
        T = Decimal("0.25")
        r = Decimal("0.05")

        price = heston_model.price(S, K, T, r, OptionType.CALL)

        # Should be very small
        assert price < Decimal("1.0")

    def test_expiry_option_intrinsic_value(self, heston_model):
        """Test option at expiry equals intrinsic value."""
        S = Decimal("110")
        K = Decimal("100")
        T = Decimal("0")  # At expiry
        r = Decimal("0.05")

        call_price = heston_model.price(S, K, T, r, OptionType.CALL)
        put_price = heston_model.price(S, K, T, r, OptionType.PUT)

        # Call: max(S - K, 0) = 10
        assert abs(call_price - Decimal("10")) < Decimal("0.01")

        # Put: max(K - S, 0) = 0
        assert abs(put_price - Decimal("0")) < Decimal("0.01")

    def test_implied_volatility_surface(self, heston_model):
        """Test IV surface generation."""
        S = 100.0
        r = 0.05
        strikes = np.array([90, 95, 100, 105, 110])
        expiries = np.array([0.25, 0.5, 1.0])

        iv_surface = heston_model.implied_volatility_surface(S, r, strikes, expiries)

        # Check shape
        assert iv_surface.shape == (len(expiries), len(strikes))

        # IVs should be reasonable (10%-50%)
        valid_ivs = iv_surface[~np.isnan(iv_surface)]
        assert all(0.05 < iv < 0.8 for iv in valid_ivs)


@pytest.mark.integration
class TestHestonCalibrator:
    """Tests for Heston calibration."""

    def test_calibrator_initialization(self):
        """Test calibrator initialization."""
        calibrator = HestonCalibrator(loss_function="rmse")

        assert calibrator.loss_function == "rmse"
        assert "v0" in calibrator.bounds
        assert calibrator.bounds["rho"] == (-0.99, 0.99)

    def test_calibration_simple(self):
        """Test calibration to synthetic market data."""
        # Create synthetic market data from known parameters
        true_params = HestonParameters(
            v0=Decimal("0.04"),
            theta=Decimal("0.04"),
            kappa=Decimal("2.0"),
            xi=Decimal("0.3"),
            rho=Decimal("-0.5"),
        )

        model = HestonModel(true_params)

        # Generate market prices
        S = Decimal("100")
        r = Decimal("0.05")

        market_data = []
        for K_val in [95, 100, 105]:
            for T_val in [0.25, 0.5]:
                price = model.price(
                    S,
                    Decimal(str(K_val)),
                    Decimal(str(T_val)),
                    r,
                    OptionType.CALL,
                )

                market_data.append({
                    "strike": K_val,
                    "expiry": T_val,
                    "price": float(price),
                    "option_type": OptionType.CALL,
                })

        market_prices = pd.DataFrame(market_data)

        # Calibrate
        calibrator = HestonCalibrator(loss_function="rmse")

        # Use a different initial guess to test calibration (not the true params)
        initial_guess = HestonParameters(
            v0=Decimal("0.06"),  # Different from true: 0.04
            theta=Decimal("0.05"),  # Different from true: 0.04
            kappa=Decimal("1.5"),  # Different from true: 2.0
            xi=Decimal("0.4"),  # Different from true: 0.3
            rho=Decimal("-0.6"),  # Different from true: -0.5
        )

        calibrated_params, diagnostics = calibrator.calibrate(
            S=S,
            r=r,
            market_prices=market_prices,
            initial_guess=initial_guess,
        )

        # Check calibration succeeded
        assert diagnostics["success"]
        assert diagnostics["final_loss"] < 1.0  # Low pricing error

        # Calibrated parameters should be close to true parameters
        assert abs(float(calibrated_params.v0) - 0.04) < 0.02
        assert abs(float(calibrated_params.theta) - 0.04) < 0.02

    def test_invalid_rho_bounds_rejected(self):
        """Ensure calibrator rejects invalid rho bounds."""
        with pytest.raises(ValueError, match="rho bounds must be within"):
            HestonCalibrator(
                bounds={
                    "v0": (0.001, 1.0),
                    "theta": (0.001, 1.0),
                    "kappa": (0.001, 10.0),
                    "xi": (0.001, 2.0),
                    "rho": (1.1, 2.0),  # Invalid: > 1
                }
            )

    def test_nan_bounds_rejected(self):
        """Ensure calibrator rejects NaN bounds."""
        with pytest.raises(ValueError, match="non-finite values"):
            HestonCalibrator(
                bounds={
                    "v0": (float("nan"), 1.0),  # Invalid: NaN
                    "theta": (0.001, 1.0),
                    "kappa": (0.001, 10.0),
                    "xi": (0.001, 2.0),
                    "rho": (-0.99, 0.99),
                }
            )

    def test_missing_bounds_rejected(self):
        """Ensure calibrator rejects incomplete bounds."""
        with pytest.raises(ValueError, match="Missing bounds for parameter"):
            HestonCalibrator(
                bounds={
                    "v0": (0.001, 1.0),
                    # Missing theta, kappa, xi, rho
                }
            )

    def test_inverted_bounds_rejected(self):
        """Ensure calibrator rejects bounds where lower >= upper."""
        with pytest.raises(ValueError, match="lower.*>= upper"):
            HestonCalibrator(
                bounds={
                    "v0": (1.0, 0.001),  # Invalid: lower > upper
                    "theta": (0.001, 1.0),
                    "kappa": (0.001, 10.0),
                    "xi": (0.001, 2.0),
                    "rho": (-0.99, 0.99),
                }
            )


@pytest.mark.unit
class TestComparisonUtilities:
    """Tests for comparison utilities."""

    def test_compare_to_black_scholes(self):
        """Test Heston vs Black-Scholes comparison."""
        params = HestonParameters(
            v0=Decimal("0.04"),
            theta=Decimal("0.04"),
            kappa=Decimal("2.0"),
            xi=Decimal("0.3"),  # Low vol of vol
            rho=Decimal("-0.5"),
        )

        model = HestonModel(params)

        results = compare_to_black_scholes(
            heston_model=model,
            S=Decimal("100"),
            r=Decimal("0.05"),
            strikes=[Decimal("95"), Decimal("100"), Decimal("105")],
            expiries=[Decimal("0.25"), Decimal("0.5")],
            option_type=OptionType.CALL,
        )

        # Check DataFrame structure
        assert len(results) == 6  # 3 strikes x 2 expiries
        assert "heston_price" in results.columns
        assert "bs_price" in results.columns
        assert "difference" in results.columns
        assert "pct_difference" in results.columns

        # For low vol of vol, Heston should be close to BS
        assert all(abs(results["pct_difference"]) < 10)  # <10% difference
