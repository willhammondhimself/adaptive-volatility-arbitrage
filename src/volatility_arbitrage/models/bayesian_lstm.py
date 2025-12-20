"""
Bayesian LSTM for Volatility Forecasting.

Uses Monte Carlo Dropout to estimate epistemic uncertainty:
run the forward pass multiple times with dropout enabled during inference,
then compute mean and standard deviation of predictions.

Reference: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class BayesianVolNetConfig:
    """Configuration for BayesianVolNet."""

    input_size: int = 1
    """Number of input features per timestep."""

    hidden_size: int = 64
    """LSTM hidden state dimension."""

    num_layers: int = 1
    """Number of LSTM layers."""

    dropout_p: float = 0.2
    """Dropout probability for MC Dropout."""

    n_mc_samples: int = 50
    """Default number of Monte Carlo samples for uncertainty estimation."""


class BayesianVolNet(nn.Module):
    """
    LSTM-based volatility forecaster with Monte Carlo Dropout.

    Architecture:
        LSTM(input_size, hidden_size) -> Dropout(p) -> Linear(hidden_size, 1)

    The dropout layer enables epistemic uncertainty estimation:
    by running multiple forward passes with dropout active,
    we approximate the posterior predictive distribution.

    High variance across samples indicates regions where the model
    is uncertain (e.g., out-of-distribution inputs, regime changes).
    """

    def __init__(self, config: Optional[BayesianVolNetConfig] = None) -> None:
        super().__init__()
        self.config = config or BayesianVolNetConfig()

        self.lstm = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=0.0,  # We apply dropout separately after LSTM
        )

        self.dropout = nn.Dropout(p=self.config.dropout_p)
        self.fc = nn.Linear(self.config.hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Volatility prediction of shape (batch, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Take last timestep output
        last_hidden = lstm_out[:, -1, :]

        # Dropout (active in train mode, used for MC sampling)
        dropped = self.dropout(last_hidden)

        # Final projection
        out = self.fc(dropped)

        # Ensure positive volatility via softplus
        return nn.functional.softplus(out)

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: int = 50,
    ) -> dict[str, float]:
        """
        Predict volatility with epistemic uncertainty via Monte Carlo Dropout.

        Runs the forward pass n_samples times with dropout enabled,
        then returns the mean prediction and standard deviation.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
               For single prediction, use batch=1.
            n_samples: Number of MC samples to draw.

        Returns:
            Dictionary with:
                - mean_vol: Mean predicted volatility (scalar)
                - epistemic_uncertainty: Std dev of predictions (scalar)
        """
        # Enable dropout for MC sampling
        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        # Stack predictions: (n_samples, batch, 1)
        preds_stacked = torch.stack(predictions, dim=0)

        # Compute statistics over MC samples
        mean_pred = preds_stacked.mean(dim=0)
        std_pred = preds_stacked.std(dim=0)

        # Return scalars (assuming batch=1 for typical use)
        return {
            "mean_vol": float(mean_pred.squeeze().item()),
            "epistemic_uncertainty": float(std_pred.squeeze().item()),
        }

    def predict_distribution(
        self,
        x: Tensor,
        n_samples: int = 50,
    ) -> Tensor:
        """
        Return raw MC samples for downstream analysis.

        Args:
            x: Input tensor
            n_samples: Number of samples

        Returns:
            Tensor of shape (n_samples, batch, 1)
        """
        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        return torch.stack(predictions, dim=0)
