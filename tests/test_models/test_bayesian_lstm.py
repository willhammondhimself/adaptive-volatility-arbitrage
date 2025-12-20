"""Tests for Bayesian LSTM model."""

import pytest
import torch

from volatility_arbitrage.models.bayesian_lstm import (
    BayesianVolNet,
    BayesianVolNetConfig,
)


@pytest.mark.unit
class TestBayesianVolNetConfig:
    """Tests for BayesianVolNetConfig."""

    def test_default_values(self) -> None:
        """Default config has expected values."""
        config = BayesianVolNetConfig()
        assert config.input_size == 1
        assert config.hidden_size == 64
        assert config.dropout_p == 0.2
        assert config.n_mc_samples == 50

    def test_custom_values(self) -> None:
        """Custom config values are preserved."""
        config = BayesianVolNetConfig(
            input_size=5,
            hidden_size=128,
            dropout_p=0.5,
        )
        assert config.input_size == 5
        assert config.hidden_size == 128
        assert config.dropout_p == 0.5


@pytest.mark.unit
class TestBayesianVolNet:
    """Tests for BayesianVolNet."""

    def test_forward_output_shape(self) -> None:
        """Forward pass produces correct output shape."""
        model = BayesianVolNet()
        x = torch.randn(8, 20, 1)  # batch=8, seq_len=20, features=1
        output = model(x)
        assert output.shape == (8, 1)

    def test_forward_positive_output(self) -> None:
        """Forward pass produces positive outputs (volatility)."""
        model = BayesianVolNet()
        x = torch.randn(16, 20, 1)
        output = model(x)
        assert (output >= 0).all()

    def test_predict_with_uncertainty_returns_dict(self) -> None:
        """predict_with_uncertainty returns expected keys."""
        model = BayesianVolNet()
        x = torch.randn(1, 20, 1)
        result = model.predict_with_uncertainty(x, n_samples=10)

        assert "mean_vol" in result
        assert "epistemic_uncertainty" in result
        assert isinstance(result["mean_vol"], float)
        assert isinstance(result["epistemic_uncertainty"], float)

    def test_uncertainty_is_positive(self) -> None:
        """Epistemic uncertainty should be non-negative."""
        model = BayesianVolNet()
        x = torch.randn(1, 20, 1)
        result = model.predict_with_uncertainty(x, n_samples=50)

        assert result["epistemic_uncertainty"] >= 0

    def test_uncertainty_nonzero_with_dropout(self) -> None:
        """With dropout enabled, uncertainty should be > 0."""
        config = BayesianVolNetConfig(dropout_p=0.5)  # High dropout
        model = BayesianVolNet(config)
        x = torch.randn(1, 20, 1)
        result = model.predict_with_uncertainty(x, n_samples=100)

        # With 50% dropout, variance should be detectable
        assert result["epistemic_uncertainty"] > 0

    def test_more_samples_reduces_mean_variance(self) -> None:
        """More MC samples should give more stable mean estimates."""
        config = BayesianVolNetConfig(dropout_p=0.3)
        model = BayesianVolNet(config)
        x = torch.randn(1, 20, 1)

        # Run multiple times with few samples vs many samples
        means_few = []
        means_many = []

        for _ in range(10):
            result_few = model.predict_with_uncertainty(x, n_samples=5)
            result_many = model.predict_with_uncertainty(x, n_samples=100)
            means_few.append(result_few["mean_vol"])
            means_many.append(result_many["mean_vol"])

        # Variance of means should be lower with more samples
        import numpy as np

        var_few = np.var(means_few)
        var_many = np.var(means_many)

        assert var_many < var_few

    def test_predict_distribution_shape(self) -> None:
        """predict_distribution returns correct shape."""
        model = BayesianVolNet()
        x = torch.randn(4, 20, 1)  # batch=4
        samples = model.predict_distribution(x, n_samples=30)

        assert samples.shape == (30, 4, 1)

    def test_custom_input_size(self) -> None:
        """Model works with custom input size."""
        config = BayesianVolNetConfig(input_size=5)
        model = BayesianVolNet(config)
        x = torch.randn(2, 10, 5)  # 5 features
        output = model(x)
        assert output.shape == (2, 1)

    def test_custom_hidden_size(self) -> None:
        """Model works with custom hidden size."""
        config = BayesianVolNetConfig(hidden_size=128)
        model = BayesianVolNet(config)
        x = torch.randn(2, 10, 1)
        output = model(x)
        assert output.shape == (2, 1)

    def test_no_dropout_zero_uncertainty(self) -> None:
        """With dropout=0, uncertainty should be near zero."""
        config = BayesianVolNetConfig(dropout_p=0.0)
        model = BayesianVolNet(config)
        x = torch.randn(1, 20, 1)
        result = model.predict_with_uncertainty(x, n_samples=50)

        # Without dropout, all samples should be identical
        assert result["epistemic_uncertainty"] < 1e-6

    def test_deterministic_in_eval_mode(self) -> None:
        """In eval mode, dropout is disabled."""
        model = BayesianVolNet()
        model.eval()
        x = torch.randn(1, 20, 1)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)
