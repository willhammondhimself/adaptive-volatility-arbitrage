"""Tests for uncertainty-adjusted position sizing."""

import pytest

from volatility_arbitrage.risk.uncertainty_sizing import (
    size_position_with_uncertainty,
    UncertaintySizer,
    UncertaintySizingConfig,
)


@pytest.mark.unit
class TestSizePositionWithUncertainty:
    """Tests for size_position_with_uncertainty function."""

    def test_zero_uncertainty_full_size(self) -> None:
        """Zero uncertainty should give full Kelly size."""
        size = size_position_with_uncertainty(
            signal_strength=1.0,
            uncertainty=0.0,
            capital=100_000,
            kelly_fraction=0.25,
            uncertainty_penalty=2.0,
        )
        # With zero uncertainty, confidence_scalar = 1.0
        # base_size = 0.25 * 100000 * 1.0 = 25000
        # But max is 0.15 * 100000 = 15000
        assert size == pytest.approx(15_000)

    def test_high_uncertainty_reduces_size(self) -> None:
        """High uncertainty should reduce position size."""
        size_low_unc = size_position_with_uncertainty(
            signal_strength=0.5,
            uncertainty=0.01,
            capital=100_000,
        )

        size_high_unc = size_position_with_uncertainty(
            signal_strength=0.5,
            uncertainty=0.50,
            capital=100_000,
        )

        assert size_high_unc < size_low_unc

    def test_uncertainty_scalar_formula(self) -> None:
        """Verify the uncertainty scalar formula."""
        capital = 100_000
        signal = 0.4
        uncertainty = 0.10
        penalty = 2.0
        kelly = 0.25

        size = size_position_with_uncertainty(
            signal_strength=signal,
            uncertainty=uncertainty,
            capital=capital,
            kelly_fraction=kelly,
            uncertainty_penalty=penalty,
        )

        # Manual calculation
        base = kelly * capital * signal  # 0.25 * 100000 * 0.4 = 10000
        scalar = 1.0 / (1.0 + penalty * uncertainty)  # 1 / 1.2 = 0.833
        expected = base * scalar

        assert size == pytest.approx(expected)

    def test_respects_max_position(self) -> None:
        """Size should not exceed max_position_pct."""
        size = size_position_with_uncertainty(
            signal_strength=10.0,  # Very strong signal
            uncertainty=0.0,
            capital=100_000,
            max_position_pct=0.10,
        )
        assert size == pytest.approx(10_000)

    def test_respects_min_position(self) -> None:
        """Size should not go below min_position_pct."""
        size = size_position_with_uncertainty(
            signal_strength=0.001,  # Very weak signal
            uncertainty=0.99,  # Very high uncertainty
            capital=100_000,
            min_position_pct=0.02,
        )
        assert size == pytest.approx(2_000)

    def test_zero_capital_returns_zero(self) -> None:
        """Zero capital should return zero size."""
        size = size_position_with_uncertainty(
            signal_strength=1.0,
            uncertainty=0.1,
            capital=0,
        )
        assert size == 0.0

    def test_negative_capital_returns_zero(self) -> None:
        """Negative capital should return zero size."""
        size = size_position_with_uncertainty(
            signal_strength=1.0,
            uncertainty=0.1,
            capital=-10_000,
        )
        assert size == 0.0

    def test_absolute_signal_strength(self) -> None:
        """Signal strength should use absolute value."""
        size_positive = size_position_with_uncertainty(
            signal_strength=0.5,
            uncertainty=0.1,
            capital=100_000,
        )

        size_negative = size_position_with_uncertainty(
            signal_strength=-0.5,
            uncertainty=0.1,
            capital=100_000,
        )

        assert size_positive == size_negative

    def test_penalty_coefficient_effect(self) -> None:
        """Higher penalty should reduce size more aggressively."""
        size_low_penalty = size_position_with_uncertainty(
            signal_strength=0.5,
            uncertainty=0.2,
            capital=100_000,
            uncertainty_penalty=1.0,
        )

        size_high_penalty = size_position_with_uncertainty(
            signal_strength=0.5,
            uncertainty=0.2,
            capital=100_000,
            uncertainty_penalty=5.0,
        )

        assert size_high_penalty < size_low_penalty


@pytest.mark.unit
class TestUncertaintySizingConfig:
    """Tests for UncertaintySizingConfig."""

    def test_default_values(self) -> None:
        """Default config has expected values."""
        config = UncertaintySizingConfig()
        assert config.kelly_fraction == 0.25
        assert config.uncertainty_penalty == 2.0
        assert config.min_position_pct == 0.01
        assert config.max_position_pct == 0.15

    def test_custom_values(self) -> None:
        """Custom config values are preserved."""
        config = UncertaintySizingConfig(
            kelly_fraction=0.5,
            uncertainty_penalty=3.0,
        )
        assert config.kelly_fraction == 0.5
        assert config.uncertainty_penalty == 3.0


@pytest.mark.unit
class TestUncertaintySizer:
    """Tests for UncertaintySizer class."""

    def test_calculate_size(self) -> None:
        """calculate_size matches function output."""
        config = UncertaintySizingConfig(kelly_fraction=0.25)
        sizer = UncertaintySizer(config)

        size = sizer.calculate_size(
            signal_strength=0.5,
            uncertainty=0.1,
            capital=100_000,
        )

        expected = size_position_with_uncertainty(
            signal_strength=0.5,
            uncertainty=0.1,
            capital=100_000,
            kelly_fraction=0.25,
        )

        assert size == pytest.approx(expected)

    def test_calculate_size_pct(self) -> None:
        """calculate_size_pct returns fraction of capital."""
        sizer = UncertaintySizer()

        pct = sizer.calculate_size_pct(
            signal_strength=0.5,
            uncertainty=0.0,
        )

        # Should be between min and max
        assert 0.01 <= pct <= 0.15

    def test_default_config(self) -> None:
        """Sizer uses default config if none provided."""
        sizer = UncertaintySizer()
        assert sizer.config.kelly_fraction == 0.25

    def test_custom_config(self) -> None:
        """Sizer uses provided config."""
        config = UncertaintySizingConfig(kelly_fraction=0.5)
        sizer = UncertaintySizer(config)
        assert sizer.config.kelly_fraction == 0.5
