"""
Pydantic schemas for volatility forecast API endpoints.
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    """Request for volatility forecast with uncertainty."""

    returns: List[float] = Field(
        ..., min_length=20, description="Historical returns (min 20 points)"
    )
    horizon: int = Field(default=1, ge=1, le=30, description="Forecast horizon in days")
    n_samples: int = Field(
        default=50, ge=10, le=200, description="Number of MC dropout samples"
    )
    hidden_size: int = Field(default=64, ge=16, le=256, description="LSTM hidden size")
    dropout_p: float = Field(
        default=0.2, ge=0.0, le=0.5, description="Dropout probability"
    )
    uncertainty_penalty: float = Field(
        default=2.0, ge=0.0, le=10.0, description="Penalty for uncertainty in sizing"
    )

    @field_validator("returns")
    @classmethod
    def validate_returns(cls, v: List[float]) -> List[float]:
        """Validate returns are finite."""
        if any(abs(r) > 1.0 for r in v):
            raise ValueError("Returns should be in decimal form (e.g., 0.01 for 1%)")
        return v


class ForecastResponse(BaseModel):
    """Response containing volatility forecast with uncertainty bounds."""

    mean_vol: float = Field(..., description="Mean volatility forecast (annualized)")
    epistemic_uncertainty: float = Field(
        ..., description="Epistemic uncertainty from MC dropout"
    )
    lower_bound: float = Field(..., description="Lower 95% CI (mean - 2*std)")
    upper_bound: float = Field(..., description="Upper 95% CI (mean + 2*std)")
    confidence_scalar: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Position sizing scalar: 1 / (1 + penalty * uncertainty)",
    )
    computation_time_ms: float = Field(..., description="Time to compute forecast")
