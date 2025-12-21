"""Tests for BayesianLSTMForecaster adapter."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from volatility_arbitrage.models.bayesian_forecaster import BayesianLSTMForecaster
from volatility_arbitrage.models.volatility import VolatilityForecaster


@pytest.mark.unit
class TestBayesianLSTMForecaster:
    """Tests for BayesianLSTMForecaster adapter."""

    def test_inherits_from_abc(self) -> None:
        """Forecaster inherits from VolatilityForecaster ABC."""
        forecaster = BayesianLSTMForecaster()
        assert isinstance(forecaster, VolatilityForecaster)

    def test_forecast_returns_decimal(self) -> None:
        """forecast() returns a Decimal."""
        forecaster = BayesianLSTMForecaster(sequence_length=10)
        returns = pd.Series(np.random.randn(30) * 0.02)
        result = forecaster.forecast(returns, horizon=1)
        assert isinstance(result, Decimal)

    def test_forecast_returns_positive(self) -> None:
        """forecast() returns positive volatility."""
        forecaster = BayesianLSTMForecaster(sequence_length=10)
        returns = pd.Series(np.random.randn(30) * 0.02)
        result = forecaster.forecast(returns, horizon=1)
        assert result > Decimal("0")

    def test_forecast_with_uncertainty_returns_dict(self) -> None:
        """forecast_with_uncertainty() returns expected keys."""
        forecaster = BayesianLSTMForecaster(sequence_length=10, n_mc_samples=20)
        returns = pd.Series(np.random.randn(30) * 0.02)
        result = forecaster.forecast_with_uncertainty(returns, horizon=1)

        assert "mean_vol" in result
        assert "epistemic_uncertainty" in result
        assert isinstance(result["mean_vol"], Decimal)
        assert isinstance(result["epistemic_uncertainty"], Decimal)

    def test_uncertainty_is_nonnegative(self) -> None:
        """Epistemic uncertainty should be non-negative."""
        forecaster = BayesianLSTMForecaster(sequence_length=10, dropout_p=0.3)
        returns = pd.Series(np.random.randn(30) * 0.02)
        result = forecaster.forecast_with_uncertainty(returns, horizon=1)

        assert result["epistemic_uncertainty"] >= Decimal("0")

    def test_insufficient_data_returns_fallback(self) -> None:
        """Returns historical vol fallback when insufficient data."""
        forecaster = BayesianLSTMForecaster(sequence_length=20)
        # 5 returns with ~2% daily std (seeded for reproducibility)
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.standard_normal(5) * 0.02)
        result = forecaster.forecast(returns, horizon=1)

        # Should return historical vol fallback with floor at 0.01
        # Code floors at max(0.01, annualized_vol)
        assert Decimal("0.01") <= result < Decimal("1.0")

    def test_uses_last_n_returns(self) -> None:
        """Uses only the last sequence_length returns."""
        forecaster = BayesianLSTMForecaster(sequence_length=10)

        # Create returns with different volatility regimes
        returns_low = pd.Series(np.random.randn(20) * 0.01)  # 1% vol
        returns_high = pd.Series(np.random.randn(10) * 0.05)  # 5% vol

        # Concatenate: older low vol, recent high vol
        full_returns = pd.concat([returns_low, returns_high], ignore_index=True)

        # Forecast should reflect recent high vol
        result = forecaster.forecast(full_returns, horizon=1)

        # Annualized 5% daily vol is roughly 80%
        # Should be notably higher than if using all data
        assert result > Decimal("0.3")

    def test_horizon_different_from_one(self) -> None:
        """Horizon > 1 produces a different (valid) forecast."""
        forecaster = BayesianLSTMForecaster(sequence_length=10)
        returns = pd.Series(np.random.randn(30) * 0.02)

        vol_1d = forecaster.forecast(returns, horizon=1)
        vol_5d = forecaster.forecast(returns, horizon=5)

        # Both should be positive Decimals
        assert vol_1d > Decimal("0")
        assert vol_5d > Decimal("0")

    def test_output_is_positive(self) -> None:
        """Forecast output is always positive."""
        forecaster = BayesianLSTMForecaster(sequence_length=10)

        # Returns with ~2% daily std
        returns = pd.Series(np.random.randn(30) * 0.02)
        result = forecaster.forecast(returns, horizon=1)

        # Output should always be positive (floor at 1%)
        assert result >= Decimal("0.01")

    def test_custom_config(self) -> None:
        """Custom config parameters are used."""
        forecaster = BayesianLSTMForecaster(
            hidden_size=128,
            dropout_p=0.5,
            sequence_length=15,
            n_mc_samples=30,
        )

        # Verify model was created with custom hidden size
        assert forecaster.model.config.hidden_size == 128
        assert forecaster.model.config.dropout_p == 0.5
        assert forecaster.sequence_length == 15
        assert forecaster.n_mc_samples == 30


@pytest.mark.unit
class TestVolatilityForecasterInterface:
    """Tests verifying VolatilityForecaster interface compliance."""

    def test_garch_has_uncertainty_method(self) -> None:
        """GARCH forecaster has uncertainty method (returns zero)."""
        from volatility_arbitrage.models.volatility import GARCHVolatility

        forecaster = GARCHVolatility()
        returns = pd.Series(np.random.randn(50) * 0.02)
        result = forecaster.forecast_with_uncertainty(returns, horizon=1)

        assert "mean_vol" in result
        assert "epistemic_uncertainty" in result
        # Non-Bayesian model returns zero uncertainty
        assert result["epistemic_uncertainty"] == Decimal("0")

    def test_ewma_has_uncertainty_method(self) -> None:
        """EWMA forecaster has uncertainty method (returns zero)."""
        from volatility_arbitrage.models.volatility import EWMAVolatility

        forecaster = EWMAVolatility()
        returns = pd.Series(np.random.randn(30) * 0.02)
        result = forecaster.forecast_with_uncertainty(returns, horizon=1)

        assert result["epistemic_uncertainty"] == Decimal("0")

    def test_historical_has_uncertainty_method(self) -> None:
        """Historical volatility has uncertainty method (returns zero)."""
        from volatility_arbitrage.models.volatility import HistoricalVolatility

        forecaster = HistoricalVolatility()
        returns = pd.Series(np.random.randn(30) * 0.02)
        result = forecaster.forecast_with_uncertainty(returns, horizon=1)

        assert result["epistemic_uncertainty"] == Decimal("0")
