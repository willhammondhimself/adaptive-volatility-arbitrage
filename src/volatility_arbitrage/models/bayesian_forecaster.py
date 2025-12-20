"""
Bayesian LSTM Volatility Forecaster.

Adapter that wraps BayesianVolNet to implement the VolatilityForecaster interface,
enabling integration with the existing strategy and backtest pipeline.
"""

from decimal import Decimal
from typing import Optional
import math

import numpy as np
import pandas as pd
import torch

from volatility_arbitrage.models.volatility import VolatilityForecaster
from volatility_arbitrage.models.bayesian_lstm import BayesianVolNet, BayesianVolNetConfig
from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


class BayesianLSTMForecaster(VolatilityForecaster):
    """
    Volatility forecaster using Bayesian LSTM with MC Dropout.

    Wraps BayesianVolNet to provide both point forecasts and
    epistemic uncertainty estimates via the VolatilityForecaster interface.

    The model expects sequences of daily returns and outputs
    annualized volatility forecasts.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        dropout_p: float = 0.2,
        sequence_length: int = 20,
        n_mc_samples: int = 50,
        pretrained_path: Optional[str] = None,
    ) -> None:
        """
        Initialize Bayesian LSTM forecaster.

        Args:
            hidden_size: LSTM hidden state dimension
            dropout_p: Dropout probability for MC Dropout
            sequence_length: Number of past returns to use as input
            n_mc_samples: Number of MC samples for uncertainty estimation
            pretrained_path: Optional path to pretrained model weights
        """
        self.sequence_length = sequence_length
        self.n_mc_samples = n_mc_samples

        config = BayesianVolNetConfig(
            input_size=1,
            hidden_size=hidden_size,
            dropout_p=dropout_p,
            n_mc_samples=n_mc_samples,
        )

        self.model = BayesianVolNet(config)

        if pretrained_path is not None:
            try:
                self.model.load_state_dict(torch.load(pretrained_path))
                logger.info(f"Loaded pretrained weights from {pretrained_path}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")

        # Track if model has been fitted
        self._is_fitted = pretrained_path is not None

        logger.debug(
            f"Initialized BayesianLSTMForecaster: "
            f"hidden={hidden_size}, dropout={dropout_p}, seq_len={sequence_length}"
        )

    def _prepare_input(self, returns: pd.Series) -> torch.Tensor:
        """
        Prepare input tensor from return series.

        Args:
            returns: Daily returns series

        Returns:
            Tensor of shape (1, seq_len, 1)
        """
        # Take last sequence_length returns
        if len(returns) < self.sequence_length:
            # Pad with zeros if insufficient data
            padded = np.zeros(self.sequence_length)
            padded[-len(returns):] = returns.values
            data = padded
        else:
            data = returns.iloc[-self.sequence_length:].values

        # Reshape to (batch=1, seq_len, features=1)
        x = torch.tensor(data, dtype=torch.float32).reshape(1, -1, 1)
        return x

    def forecast(self, returns: pd.Series, horizon: int = 1) -> Decimal:
        """
        Forecast volatility using Bayesian LSTM.

        Args:
            returns: Daily returns series
            horizon: Forecast horizon (days) - used for scaling

        Returns:
            Annualized volatility forecast
        """
        result = self.forecast_with_uncertainty(returns, horizon)
        return result["mean_vol"]

    def forecast_with_uncertainty(
        self, returns: pd.Series, horizon: int = 1
    ) -> dict[str, Decimal]:
        """
        Forecast volatility with epistemic uncertainty via MC Dropout.

        Args:
            returns: Daily returns series
            horizon: Forecast horizon (days)

        Returns:
            Dictionary with 'mean_vol' and 'epistemic_uncertainty'
        """
        # Fall back to historical vol if insufficient data
        if len(returns) < self.sequence_length:
            logger.warning("Insufficient data for Bayesian LSTM, using historical fallback")
            if len(returns) >= 2:
                # Use historical volatility
                daily_vol = returns.std()
                annualized_vol = float(daily_vol) * math.sqrt(252)
                return {
                    "mean_vol": Decimal(str(round(max(0.01, annualized_vol), 6))),
                    "epistemic_uncertainty": Decimal("0.05"),  # High uncertainty
                }
            else:
                return {
                    "mean_vol": Decimal("0.20"),
                    "epistemic_uncertainty": Decimal("0.10"),
                }

        try:
            x = self._prepare_input(returns)

            # Get prediction with uncertainty
            result = self.model.predict_with_uncertainty(x, n_samples=self.n_mc_samples)

            # Model outputs daily vol, annualize it
            # sqrt(252) for annualization
            annualization_factor = math.sqrt(252)
            mean_vol = result["mean_vol"] * annualization_factor
            uncertainty = result["epistemic_uncertainty"] * annualization_factor

            # Scale for horizon (simplified)
            if horizon > 1:
                horizon_factor = math.sqrt(horizon / 252)
                mean_vol *= horizon_factor
                uncertainty *= horizon_factor

            # Ensure non-negative
            mean_vol = max(0.01, mean_vol)  # Floor at 1% vol
            uncertainty = max(0.0, uncertainty)

            return {
                "mean_vol": Decimal(str(round(mean_vol, 6))),
                "epistemic_uncertainty": Decimal(str(round(uncertainty, 6))),
            }

        except Exception as e:
            logger.warning(f"Bayesian LSTM forecast failed: {e}, using fallback")
            return {
                "mean_vol": Decimal("0.20"),
                "epistemic_uncertainty": Decimal("0.05"),
            }

    def fit(
        self,
        returns: pd.Series,
        realized_vol: pd.Series,
        epochs: int = 10,
        lr: float = 0.01,
    ) -> list[float]:
        """
        Fit the model to historical data.

        Args:
            returns: Daily returns series
            realized_vol: Target realized volatility series
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            List of training losses per epoch
        """
        if len(returns) < self.sequence_length + 10:
            logger.warning("Insufficient data for training")
            return []

        # Prepare training data
        X, y = self._prepare_training_data(returns, realized_vol)

        if len(X) == 0:
            return []

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        losses = []
        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self._is_fitted = True
        logger.info(f"Fitted BayesianLSTMForecaster: final_loss={losses[-1]:.6f}")

        return losses

    def _prepare_training_data(
        self, returns: pd.Series, realized_vol: pd.Series
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare training sequences from returns and realized vol."""
        X_list = []
        y_list = []

        for t in range(self.sequence_length, min(len(returns), len(realized_vol)) - 1):
            # Input: past returns
            x_seq = returns.iloc[t - self.sequence_length : t].values
            # Target: next realized vol (daily, not annualized)
            y_val = realized_vol.iloc[t + 1] / math.sqrt(252)

            X_list.append(x_seq)
            y_list.append(y_val)

        if not X_list:
            return torch.tensor([]), torch.tensor([])

        X = np.array(X_list).reshape(-1, self.sequence_length, 1)
        y = np.array(y_list).reshape(-1, 1)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted or loaded."""
        return self._is_fitted
