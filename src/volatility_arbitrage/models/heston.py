"""
Heston Stochastic Volatility Model Implementation.

Implements the Heston (1993) model for option pricing with stochastic volatility.
Includes calibration to market option prices using L-BFGS optimization.

Mathematical Foundation:
    dS_t = μS_t dt + √v_t S_t dW_1
    dv_t = κ(θ - v_t)dt + ξ√v_t dW_2
    dW_1 dW_2 = ρ dt

where:
    v_t: instantaneous variance
    θ: long-term variance
    κ: mean reversion speed
    ξ: volatility of volatility
    ρ: correlation between stock and volatility

References:
    Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility.
    The Review of Financial Studies, 6(2), 327-343.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd
from scipy import integrate, optimize
from scipy.interpolate import RegularGridInterpolator

from volatility_arbitrage.core.types import OptionType
from volatility_arbitrage.models.black_scholes import BlackScholesModel
from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HestonParameters:
    """
    Heston model parameters.

    Attributes:
        v0: Initial variance (σ₀²)
        theta: Long-term variance (θ)
        kappa: Mean reversion speed (κ)
        xi: Volatility of volatility (ξ)
        rho: Correlation between stock and volatility (ρ)
    """

    v0: Decimal  # Initial variance
    theta: Decimal  # Long-term variance
    kappa: Decimal  # Mean reversion speed
    xi: Decimal  # Volatility of volatility (vol of vol)
    rho: Decimal  # Correlation stock-vol

    def __post_init__(self) -> None:
        """Validate parameters."""
        # Feller condition: 2κθ > ξ² ensures variance stays positive
        feller = 2 * float(self.kappa) * float(self.theta)
        xi_sq = float(self.xi) ** 2

        if feller <= xi_sq:
            logger.warning(
                f"Feller condition violated: 2κθ={feller:.4f} <= ξ²={xi_sq:.4f}. "
                "Variance may reach zero."
            )

        # Parameter bounds
        if float(self.v0) <= 0:
            raise ValueError(f"v0 must be positive, got {self.v0}")
        if float(self.theta) <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if float(self.kappa) <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if float(self.xi) <= 0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not -1 <= float(self.rho) <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    def to_dict(self) -> dict[str, float]:
        """Convert parameters to dictionary."""
        return {
            "v0": float(self.v0),
            "theta": float(self.theta),
            "kappa": float(self.kappa),
            "xi": float(self.xi),
            "rho": float(self.rho),
        }

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> "HestonParameters":
        """Create parameters from dictionary."""
        return cls(
            v0=Decimal(str(params["v0"])),
            theta=Decimal(str(params["theta"])),
            kappa=Decimal(str(params["kappa"])),
            xi=Decimal(str(params["xi"])),
            rho=Decimal(str(params["rho"])),
        )


class HestonModel:
    """
    Heston stochastic volatility model for option pricing.

    Uses the characteristic function approach for fast and accurate pricing.
    """

    def __init__(self, params: HestonParameters):
        """
        Initialize Heston model.

        Args:
            params: Heston model parameters
        """
        self.params = params

    def characteristic_function(
        self, u: np.ndarray, S: float, T: float, r: float
    ) -> np.ndarray:
        """
        Heston characteristic function.

        Implements the characteristic function φ(u) from Heston (1993).

        Args:
            u: Input values for characteristic function
            S: Current stock price
            T: Time to expiry in years
            r: Risk-free rate

        Returns:
            Characteristic function values
        """
        # Convert parameters to float
        v0 = float(self.params.v0)
        theta = float(self.params.theta)
        kappa = float(self.params.kappa)
        xi = float(self.params.xi)
        rho = float(self.params.rho)

        # Intermediate calculations
        i = complex(0, 1)
        d = np.sqrt((kappa - i * rho * xi * u) ** 2 + xi**2 * (i * u + u**2))

        g = (kappa - i * rho * xi * u - d) / (kappa - i * rho * xi * u + d)

        # Characteristic function components
        C = (
            i * u * (np.log(S) + r * T)
            + (kappa * theta / xi**2)
            * ((kappa - i * rho * xi * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        )

        D = ((kappa - i * rho * xi * u - d) / xi**2) * (
            (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        )

        return np.exp(C + D * v0)

    def price(
        self,
        S: Decimal,
        K: Decimal,
        T: Decimal,
        r: Decimal,
        option_type: OptionType,
    ) -> Decimal:
        """
        Price European option using Heston model.

        Uses Carr-Madan FFT formula for efficient pricing.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            option_type: CALL or PUT

        Returns:
            Option price
        """
        # Convert to float for calculations
        S_f = float(S)
        K_f = float(K)
        T_f = float(T)
        r_f = float(r)

        if T_f <= 0:
            # Handle expiry
            if option_type == OptionType.CALL:
                return Decimal(str(max(S_f - K_f, 0)))
            else:
                return Decimal(str(max(K_f - S_f, 0)))

        # Use numerical integration for pricing
        # P1: probability of finishing in-the-money under stock measure
        # P2: probability of finishing in-the-money under money market measure

        def integrand_P1(u: float) -> float:
            """Integrand for P1 probability."""
            phi = self.characteristic_function(u - 1j, S_f, T_f, r_f)
            return np.real(
                np.exp(-1j * u * np.log(K_f)) * phi / (1j * u * self.characteristic_function(-1j, S_f, T_f, r_f))
            )

        def integrand_P2(u: float) -> float:
            """Integrand for P2 probability."""
            phi = self.characteristic_function(u, S_f, T_f, r_f)
            return np.real(np.exp(-1j * u * np.log(K_f)) * phi / (1j * u))

        # Numerical integration
        P1 = 0.5 + (1 / np.pi) * integrate.quad(integrand_P1, 0, 100)[0]
        P2 = 0.5 + (1 / np.pi) * integrate.quad(integrand_P2, 0, 100)[0]

        # Option price
        if option_type == OptionType.CALL:
            price = S_f * P1 - K_f * np.exp(-r_f * T_f) * P2
        else:
            # Put-call parity
            call_price = S_f * P1 - K_f * np.exp(-r_f * T_f) * P2
            price = call_price - S_f + K_f * np.exp(-r_f * T_f)

        return Decimal(str(max(price, 0)))

    def implied_volatility_surface(
        self,
        S: float,
        r: float,
        strikes: np.ndarray,
        expiries: np.ndarray,
    ) -> np.ndarray:
        """
        Generate implied volatility surface from Heston model.

        Args:
            S: Current stock price
            r: Risk-free rate
            strikes: Array of strike prices
            expiries: Array of time to expiries (in years)

        Returns:
            2D array of implied volatilities [expiries x strikes]
        """
        iv_surface = np.zeros((len(expiries), len(strikes)))

        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                # Price with Heston
                heston_price = self.price(
                    Decimal(str(S)),
                    Decimal(str(K)),
                    Decimal(str(T)),
                    Decimal(str(r)),
                    OptionType.CALL,
                )

                # Invert to get implied vol
                try:
                    iv = BlackScholesModel.calculate_implied_volatility(
                        market_price=heston_price,
                        S=Decimal(str(S)),
                        K=Decimal(str(K)),
                        T=Decimal(str(T)),
                        r=Decimal(str(r)),
                        option_type=OptionType.CALL,
                    )
                    iv_surface[i, j] = float(iv)
                except (ValueError, RuntimeError):
                    # If IV calculation fails, use NaN
                    iv_surface[i, j] = np.nan

        return iv_surface


class HestonCalibrator:
    """
    Calibrate Heston model to market option prices.

    Uses L-BFGS-B optimization to minimize pricing errors.
    """

    def __init__(
        self,
        loss_function: str = "rmse",
        bounds: Optional[dict[str, tuple[float, float]]] = None,
    ):
        """
        Initialize calibrator.

        Args:
            loss_function: Loss function to minimize ('rmse' or 'mape')
            bounds: Parameter bounds as {param_name: (lower, upper)}
        """
        self.loss_function = loss_function

        # Default parameter bounds
        self.bounds = bounds or {
            "v0": (0.001, 1.0),  # Initial variance
            "theta": (0.001, 1.0),  # Long-term variance
            "kappa": (0.001, 10.0),  # Mean reversion speed
            "xi": (0.001, 2.0),  # Vol of vol
            "rho": (-0.99, 0.99),  # Correlation
        }

    def calibrate(
        self,
        S: Decimal,
        r: Decimal,
        market_prices: pd.DataFrame,
        initial_guess: Optional[HestonParameters] = None,
    ) -> tuple[HestonParameters, dict[str, float]]:
        """
        Calibrate Heston model to market option prices.

        Args:
            S: Current stock price
            r: Risk-free rate
            market_prices: DataFrame with columns [strike, expiry, price, option_type]
            initial_guess: Initial parameter guess (default: reasonable starting point)

        Returns:
            Tuple of (calibrated_parameters, diagnostics)
        """
        logger.info(f"Calibrating Heston model to {len(market_prices)} market prices")

        # Initial guess
        if initial_guess is None:
            initial_params = np.array([0.04, 0.04, 2.0, 0.5, -0.5])  # v0, theta, kappa, xi, rho
        else:
            initial_params = np.array([
                float(initial_guess.v0),
                float(initial_guess.theta),
                float(initial_guess.kappa),
                float(initial_guess.xi),
                float(initial_guess.rho),
            ])

        # Parameter bounds for L-BFGS-B
        bounds_list = [
            self.bounds["v0"],
            self.bounds["theta"],
            self.bounds["kappa"],
            self.bounds["xi"],
            self.bounds["rho"],
        ]

        def objective(params: np.ndarray) -> float:
            """Objective function to minimize."""
            try:
                # Create Heston parameters
                heston_params = HestonParameters(
                    v0=Decimal(str(params[0])),
                    theta=Decimal(str(params[1])),
                    kappa=Decimal(str(params[2])),
                    xi=Decimal(str(params[3])),
                    rho=Decimal(str(params[4])),
                )

                model = HestonModel(heston_params)

                # Calculate pricing errors
                errors = []
                for _, row in market_prices.iterrows():
                    model_price = model.price(
                        S=S,
                        K=Decimal(str(row["strike"])),
                        T=Decimal(str(row["expiry"])),
                        r=r,
                        option_type=row["option_type"],
                    )

                    market_price = Decimal(str(row["price"]))

                    if self.loss_function == "rmse":
                        errors.append((float(model_price - market_price)) ** 2)
                    elif self.loss_function == "mape":
                        errors.append(abs(float(model_price - market_price)) / float(market_price))

                # Return RMSE or MAPE
                if self.loss_function == "rmse":
                    return np.sqrt(np.mean(errors))
                else:
                    return np.mean(errors) * 100

            except Exception as e:
                logger.warning(f"Error in objective function: {e}")
                return 1e10  # Large penalty

        # Optimize
        result = optimize.minimize(
            objective,
            initial_params,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        # Extract calibrated parameters
        calibrated_params = HestonParameters(
            v0=Decimal(str(result.x[0])),
            theta=Decimal(str(result.x[1])),
            kappa=Decimal(str(result.x[2])),
            xi=Decimal(str(result.x[3])),
            rho=Decimal(str(result.x[4])),
        )

        # Diagnostics
        diagnostics = {
            "success": result.success,
            "iterations": result.nit,
            "final_loss": result.fun,
            "message": result.message,
        }

        logger.info(
            f"Calibration {'succeeded' if result.success else 'failed'}: "
            f"loss={result.fun:.6f}, iterations={result.nit}"
        )

        return calibrated_params, diagnostics


def compare_to_black_scholes(
    heston_model: HestonModel,
    S: Decimal,
    r: Decimal,
    strikes: list[Decimal],
    expiries: list[Decimal],
    option_type: OptionType = OptionType.CALL,
) -> pd.DataFrame:
    """
    Compare Heston prices to Black-Scholes prices.

    Args:
        heston_model: Calibrated Heston model
        S: Current stock price
        r: Risk-free rate
        strikes: List of strike prices
        expiries: List of time to expiries
        option_type: Option type

    Returns:
        DataFrame with comparison results
    """
    results = []

    # Use average volatility from Heston parameters
    avg_vol = Decimal(str(np.sqrt(float(heston_model.params.theta))))

    for T in expiries:
        for K in strikes:
            # Heston price
            heston_price = heston_model.price(S, K, T, r, option_type)

            # Black-Scholes price
            bs_price = BlackScholesModel.price(S, K, T, r, avg_vol, option_type)

            # Price difference
            diff = heston_price - bs_price
            pct_diff = (diff / bs_price * 100) if bs_price > 0 else Decimal("0")

            results.append({
                "strike": float(K),
                "expiry": float(T),
                "heston_price": float(heston_price),
                "bs_price": float(bs_price),
                "difference": float(diff),
                "pct_difference": float(pct_diff),
            })

    return pd.DataFrame(results)
