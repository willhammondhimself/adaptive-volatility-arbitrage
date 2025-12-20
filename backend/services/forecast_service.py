"""
Volatility forecast service using Bayesian LSTM with MC Dropout.
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add src path to import volatility_arbitrage modules
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from volatility_arbitrage.models.bayesian_lstm import BayesianVolNet, BayesianVolNetConfig

from backend.schemas.forecast import ForecastRequest, ForecastResponse


class ForecastService:
    """Service for volatility forecasting with uncertainty estimation."""

    def __init__(self, default_hidden_size: int = 64, default_dropout_p: float = 0.2):
        self.default_hidden_size = default_hidden_size
        self.default_dropout_p = default_dropout_p
        self._model_cache: dict[tuple, BayesianVolNet] = {}

    def _get_model(self, hidden_size: int, dropout_p: float) -> BayesianVolNet:
        """Get or create a model with the specified configuration."""
        key = (hidden_size, dropout_p)
        if key not in self._model_cache:
            config = BayesianVolNetConfig(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=1,
                dropout_p=dropout_p,
            )
            self._model_cache[key] = BayesianVolNet(config)
        return self._model_cache[key]

    def predict(self, request: ForecastRequest) -> ForecastResponse:
        """
        Generate volatility forecast with uncertainty bounds.

        Args:
            request: Forecast request with returns and parameters

        Returns:
            Forecast response with mean, uncertainty, and bounds
        """
        start_time = time.time()

        # Get model
        model = self._get_model(request.hidden_size, request.dropout_p)

        # Prepare input tensor
        returns = np.array(request.returns, dtype=np.float32)
        x = torch.tensor(returns, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # Shape: (1, seq_len, 1)

        # Run MC dropout inference
        result = model.predict_with_uncertainty(x, n_samples=request.n_samples)

        mean_vol = result["mean_vol"]
        uncertainty = result["epistemic_uncertainty"]

        # Annualize if model outputs daily vol (multiply by sqrt(252))
        # Note: The raw model outputs are in the same scale as input
        # For now, assume returns are daily and we want annualized vol
        annualization_factor = np.sqrt(252)
        mean_vol_ann = mean_vol * annualization_factor
        uncertainty_ann = uncertainty * annualization_factor

        # Compute bounds (95% CI = mean +/- 2*std)
        lower_bound = max(0.0, mean_vol_ann - 2 * uncertainty_ann)
        upper_bound = mean_vol_ann + 2 * uncertainty_ann

        # Compute confidence scalar for position sizing
        # 1 / (1 + penalty * uncertainty) where uncertainty is relative
        relative_uncertainty = uncertainty / mean_vol if mean_vol > 0 else 1.0
        confidence_scalar = 1.0 / (1.0 + request.uncertainty_penalty * relative_uncertainty)
        confidence_scalar = max(0.0, min(1.0, confidence_scalar))

        computation_time_ms = (time.time() - start_time) * 1000

        return ForecastResponse(
            mean_vol=mean_vol_ann,
            epistemic_uncertainty=uncertainty_ann,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_scalar=confidence_scalar,
            computation_time_ms=computation_time_ms,
        )
