"""
Volatility forecasting models.

Implements various methods for estimating future volatility:
- Historical volatility
- EWMA (Exponentially Weighted Moving Average)
- GARCH(1,1) (Generalized Autoregressive Conditional Heteroskedasticity)
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize

from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


class VolatilityForecaster(ABC):
    """Abstract base class for volatility forecasting models."""

    @abstractmethod
    def forecast(self, returns: pd.Series, horizon: int = 1) -> Decimal:
        """
        Forecast volatility for given return series.

        Args:
            returns: Time series of returns
            horizon: Forecast horizon (days)

        Returns:
            Annualized volatility forecast
        """
        pass


class HistoricalVolatility(VolatilityForecaster):
    """
    Historical volatility estimator.

    Uses standard deviation of historical returns with optional windowing.
    """

    def __init__(self, window: Optional[int] = None) -> None:
        """
        Initialize historical volatility estimator.

        Args:
            window: Lookback window in days (None = use all data)
        """
        self.window = window
        logger.debug(f"Initialized HistoricalVolatility with window={window}")

    def forecast(self, returns: pd.Series, horizon: int = 1) -> Decimal:
        """
        Calculate historical volatility.

        Args:
            returns: Daily returns series
            horizon: Forecast horizon (days)

        Returns:
            Annualized volatility forecast
        """
        if len(returns) < 2:
            logger.warning("Insufficient data for historical volatility")
            return Decimal("0.2")  # Default 20% volatility

        # Apply window if specified
        if self.window is not None:
            returns = returns.iloc[-self.window :]

        # Calculate standard deviation
        std_dev = returns.std()

        # Annualize (assuming 252 trading days)
        annualized_vol = std_dev * np.sqrt(252)

        # Scale for horizon (simplified: assumes independence)
        horizon_vol = annualized_vol * np.sqrt(horizon / 252)

        return Decimal(str(horizon_vol))


class EWMAVolatility(VolatilityForecaster):
    """
    EWMA (Exponentially Weighted Moving Average) volatility estimator.

    Gives more weight to recent observations using exponential decay.
    """

    def __init__(self, lambda_param: Decimal = Decimal("0.94")) -> None:
        """
        Initialize EWMA volatility estimator.

        Args:
            lambda_param: Decay factor (typically 0.94 for daily data)
        """
        self.lambda_param = float(lambda_param)
        logger.debug(f"Initialized EWMAVolatility with lambda={self.lambda_param}")

    def forecast(self, returns: pd.Series, horizon: int = 1) -> Decimal:
        """
        Calculate EWMA volatility.

        Args:
            returns: Daily returns series
            horizon: Forecast horizon (days)

        Returns:
            Annualized volatility forecast
        """
        if len(returns) < 2:
            logger.warning("Insufficient data for EWMA volatility")
            return Decimal("0.2")

        # Calculate squared returns
        squared_returns = returns**2

        # Initialize variance with first squared return
        variance = squared_returns.iloc[0]

        # Apply EWMA recursion
        for sq_ret in squared_returns.iloc[1:]:
            variance = self.lambda_param * variance + (1 - self.lambda_param) * sq_ret

        # Convert to standard deviation
        volatility = np.sqrt(variance)

        # Annualize
        annualized_vol = volatility * np.sqrt(252)

        # Scale for horizon
        horizon_vol = annualized_vol * np.sqrt(horizon / 252)

        return Decimal(str(horizon_vol))


class GARCHVolatility(VolatilityForecaster):
    """
    GARCH(p,q) volatility estimator.

    Generalized Autoregressive Conditional Heteroskedasticity model.
    Default is GARCH(1,1) which is most commonly used in practice.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        max_iter: int = 1000,
    ) -> None:
        """
        Initialize GARCH volatility estimator.

        Args:
            p: Number of lag variance terms
            q: Number of lag residual terms
            max_iter: Maximum iterations for optimization
        """
        self.p = p
        self.q = q
        self.max_iter = max_iter
        self._params: Optional[np.ndarray] = None
        logger.debug(f"Initialized GARCHVolatility with p={p}, q={q}")

    def _estimate_params(self, returns: pd.Series) -> tuple[float, float, float]:
        """
        Estimate GARCH(1,1) parameters using MLE.

        Args:
            returns: Return series

        Returns:
            Tuple of (omega, alpha, beta) parameters
        """

        def garch_likelihood(params: np.ndarray) -> float:
            """Negative log-likelihood for GARCH(1,1)."""
            omega, alpha, beta = params

            # Constraint violations
            if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
                return 1e10

            # Initialize variance
            variance = np.var(returns)
            log_likelihood = 0

            # Calculate log-likelihood
            for ret in returns:
                variance = omega + alpha * ret**2 + beta * variance
                if variance <= 0:
                    return 1e10
                log_likelihood += np.log(variance) + ret**2 / variance

            return log_likelihood

        # Initial parameter guess
        initial_params = np.array([0.01, 0.05, 0.90])

        # Bounds: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        bounds = [(1e-6, 1), (0, 1), (0, 1)]

        try:
            result = optimize.minimize(
                garch_likelihood,
                initial_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iter},
            )

            if result.success:
                omega, alpha, beta = result.x
                logger.debug(
                    f"GARCH parameters: omega={omega:.6f}, alpha={alpha:.6f}, beta={beta:.6f}"
                )
                return omega, alpha, beta
            else:
                logger.warning("GARCH optimization did not converge, using defaults")
                return 0.01, 0.05, 0.90

        except Exception as e:
            logger.warning(f"GARCH parameter estimation failed: {e}, using defaults")
            return 0.01, 0.05, 0.90

    def forecast(self, returns: pd.Series, horizon: int = 1) -> Decimal:
        """
        Calculate GARCH volatility forecast.

        Args:
            returns: Daily returns series
            horizon: Forecast horizon (days)

        Returns:
            Annualized volatility forecast
        """
        if len(returns) < max(self.p, self.q) + 10:
            logger.warning("Insufficient data for GARCH, falling back to historical vol")
            fallback = HistoricalVolatility()
            return fallback.forecast(returns, horizon)

        # Estimate parameters
        omega, alpha, beta = self._estimate_params(returns)

        # Calculate current variance
        current_variance = np.var(returns)
        last_return_sq = returns.iloc[-1] ** 2

        # One-step-ahead forecast
        forecast_variance = omega + alpha * last_return_sq + beta * current_variance

        # Multi-step forecast (simplified persistence model)
        if horizon > 1:
            # Long-run variance
            long_run_var = omega / (1 - alpha - beta)

            # Forecast using exponential decay to long-run
            persistence = alpha + beta
            forecast_variance = (
                long_run_var
                + (forecast_variance - long_run_var) * persistence ** (horizon - 1)
            )

        # Convert to volatility
        volatility = np.sqrt(forecast_variance)

        # Annualize
        annualized_vol = volatility * np.sqrt(252)

        return Decimal(str(annualized_vol))


def calculate_returns(prices: pd.Series, method: str = "log") -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        prices: Price series
        method: Return calculation method ('log' or 'simple')

    Returns:
        Returns series
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}")

    return returns.dropna()


def realized_volatility(
    returns: pd.Series,
    window: int = 30,
    annualize: bool = True,
) -> pd.Series:
    """
    Calculate rolling realized volatility.

    Args:
        returns: Return series
        window: Rolling window size
        annualize: Whether to annualize the volatility

    Returns:
        Rolling volatility series
    """
    rolling_std = returns.rolling(window=window).std()

    if annualize:
        rolling_std = rolling_std * np.sqrt(252)

    return rolling_std
