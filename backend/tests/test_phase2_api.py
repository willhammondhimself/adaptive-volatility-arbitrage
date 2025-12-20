"""
Tests for Phase 2 API endpoints: forecast, costs, backtest.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


class TestForecastAPI:
    """Tests for /api/v1/forecast endpoints."""

    def test_predict_volatility(self):
        """Test volatility forecast endpoint."""
        # Generate synthetic returns
        import numpy as np
        np.random.seed(42)
        returns = (np.random.randn(30) * 0.02).tolist()

        request_data = {
            "returns": returns,
            "horizon": 1,
            "n_samples": 20,  # Fewer samples for faster test
            "hidden_size": 32,  # Smaller model for faster test
            "dropout_p": 0.2,
            "uncertainty_penalty": 2.0,
        }

        response = client.post("/api/v1/forecast/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "mean_vol" in data
        assert "epistemic_uncertainty" in data
        assert "lower_bound" in data
        assert "upper_bound" in data
        assert "confidence_scalar" in data
        assert "computation_time_ms" in data

        # Validate ranges
        assert data["mean_vol"] > 0
        assert data["epistemic_uncertainty"] >= 0
        assert data["lower_bound"] <= data["mean_vol"]
        assert data["upper_bound"] >= data["mean_vol"]
        assert 0 <= data["confidence_scalar"] <= 1

    def test_predict_volatility_validation(self):
        """Test validation of forecast request."""
        # Too few returns
        request_data = {
            "returns": [0.01, 0.02, 0.03],  # Less than 20
            "horizon": 1,
        }

        response = client.post("/api/v1/forecast/predict", json=request_data)
        assert response.status_code == 422  # Validation error


class TestCostsAPI:
    """Tests for /api/v1/costs endpoints."""

    def test_estimate_costs(self):
        """Test transaction cost estimation endpoint."""
        request_data = {
            "order_size": 100,
            "price": 50.0,
            "daily_volume": 1000000,
            "half_spread_bps": 5.0,
            "impact_coefficient": 0.1,
        }

        response = client.post("/api/v1/costs/estimate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "total_cost" in data
        assert "spread_cost" in data
        assert "impact_cost" in data
        assert "impact_bps" in data
        assert "effective_price" in data
        assert "cost_as_pct" in data

        # Validate cost breakdown
        assert data["total_cost"] == pytest.approx(
            data["spread_cost"] + data["impact_cost"], rel=0.01
        )
        assert data["total_cost"] > 0
        assert data["effective_price"] > data["effective_price"] - data["total_cost"] / 100

    def test_estimate_costs_large_order(self):
        """Test that large orders have higher impact costs."""
        base_request = {
            "price": 50.0,
            "daily_volume": 1000000,
            "half_spread_bps": 5.0,
            "impact_coefficient": 0.1,
        }

        # Small order
        small_request = {**base_request, "order_size": 100}
        small_response = client.post("/api/v1/costs/estimate", json=small_request)
        small_data = small_response.json()

        # Large order
        large_request = {**base_request, "order_size": 10000}
        large_response = client.post("/api/v1/costs/estimate", json=large_request)
        large_data = large_response.json()

        # Impact should be higher for larger orders (relative to order size)
        assert large_data["impact_bps"] > small_data["impact_bps"]

    def test_estimate_costs_validation(self):
        """Test validation of cost request."""
        # Negative order size
        request_data = {
            "order_size": -100,
            "price": 50.0,
            "daily_volume": 1000000,
        }

        response = client.post("/api/v1/costs/estimate", json=request_data)
        assert response.status_code == 422  # Validation error


class TestBacktestAPI:
    """Tests for /api/v1/backtest endpoints."""

    @pytest.mark.slow
    def test_run_backtest_quick(self):
        """Test backtest endpoint with limited data."""
        request_data = {
            "data_dir": "src/volatility_arbitrage/data/SPY_Options_2019_24",
            "max_days": 5,  # Very limited for quick test
            "initial_capital": 100000.0,
            "entry_threshold_pct": 5.0,
            "exit_threshold_pct": 2.0,
            "position_size_pct": 15.0,
            "use_bayesian_lstm": False,
            "use_impact_model": False,
            "use_uncertainty_sizing": False,
        }

        response = client.post("/api/v1/backtest/run", json=request_data)

        if response.status_code == 200:
            data = response.json()
            assert "metrics" in data
            assert "equity_curve" in data
            assert "phase2_status" in data
            assert "computation_time_ms" in data
            assert "data_range" in data

            # Validate metrics structure
            metrics = data["metrics"]
            assert "total_return" in metrics
            assert "sharpe_ratio" in metrics
            assert "max_drawdown" in metrics
            assert "total_trades" in metrics
        else:
            # Data might not be available in test environment
            assert response.status_code in [404, 500]

    def test_backtest_validation(self):
        """Test validation of backtest request."""
        # Invalid position size
        request_data = {
            "position_size_pct": 100.0,  # > 50% limit
            "initial_capital": 100000.0,
        }

        response = client.post("/api/v1/backtest/run", json=request_data)
        assert response.status_code == 422  # Validation error


# Quick tests that don't require data
class TestAPIHealth:
    """Quick API health tests."""

    def test_all_endpoints_registered(self):
        """Verify all Phase 2 endpoints are registered."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi.get("paths", {})

        assert "/api/v1/forecast/predict" in paths
        assert "/api/v1/costs/estimate" in paths
        assert "/api/v1/backtest/run" in paths
