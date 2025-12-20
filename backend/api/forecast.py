"""
Volatility forecast API endpoints.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.forecast import ForecastRequest, ForecastResponse
from backend.services.forecast_service import ForecastService

router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])

# Initialize service (singleton)
forecast_service = ForecastService()


@router.post("/predict", response_model=ForecastResponse)
async def predict_volatility(request: ForecastRequest) -> ForecastResponse:
    """
    Generate volatility forecast with uncertainty estimation.

    Uses Bayesian LSTM with Monte Carlo Dropout to provide:
    - Mean volatility forecast (annualized)
    - Epistemic uncertainty from MC samples
    - 95% confidence interval bounds
    - Position sizing confidence scalar

    **Input:**
    - returns: List of historical returns (min 20 points, decimal form e.g., 0.01 = 1%)
    - horizon: Forecast horizon in days (1-30)
    - n_samples: Number of MC dropout samples (10-200)

    **Performance:**
    - Typical: 50-200ms depending on n_samples
    """
    try:
        return forecast_service.predict(request)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating forecast: {str(e)}"
        )
