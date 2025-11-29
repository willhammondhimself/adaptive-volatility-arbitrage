"""
Tests for Heston API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Volatility Arbitrage Dashboard API" in response.json()["message"]


def test_compute_price_surface():
    """Test price surface computation."""
    request_data = {
        "params": {
            "v0": 0.04,
            "theta": 0.05,
            "kappa": 2.0,
            "sigma_v": 0.3,
            "rho": -0.7,
            "r": 0.05,
            "q": 0.0,
        },
        "spot": 100.0,
        "strike_range": [90, 110],
        "maturity_range": [0.5, 1.0],
        "num_strikes": 5,
        "num_maturities": 3,
    }

    response = client.post("/api/v1/heston/price-surface", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "strikes" in data
    assert "maturities" in data
    assert "prices" in data
    assert len(data["strikes"]) == 5
    assert len(data["maturities"]) == 3
    assert len(data["prices"]) == 3  # One row per maturity
    assert len(data["prices"][0]) == 5  # One price per strike
    assert "computation_time_ms" in data
    assert "cache_hit" in data


def test_cache_functionality():
    """Test that caching works correctly."""
    request_data = {
        "params": {
            "v0": 0.04,
            "theta": 0.05,
            "kappa": 2.0,
            "sigma_v": 0.3,
            "rho": -0.7,
            "r": 0.05,
        },
        "spot": 100.0,
        "strike_range": [95, 105],
        "maturity_range": [0.5, 1.0],
        "num_strikes": 3,
        "num_maturities": 2,
    }

    # First call - cache miss
    response1 = client.post("/api/v1/heston/price-surface", json=request_data)
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["cache_hit"] is False

    # Second call - should be cache hit
    response2 = client.post("/api/v1/heston/price-surface", json=request_data)
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cache_hit"] is True

    # Computation time should be much faster for cache hit
    assert data2["computation_time_ms"] < data1["computation_time_ms"]


def test_invalid_parameters():
    """Test validation of invalid parameters."""
    # Negative v0
    request_data = {
        "params": {
            "v0": -0.04,
            "theta": 0.05,
            "kappa": 2.0,
            "sigma_v": 0.3,
            "rho": -0.7,
            "r": 0.05,
        },
        "spot": 100.0,
        "strike_range": [90, 110],
        "maturity_range": [0.5, 1.0],
    }

    response = client.post("/api/v1/heston/price-surface", json=request_data)
    assert response.status_code == 422  # Validation error


def test_cache_stats():
    """Test cache statistics endpoint."""
    response = client.get("/api/v1/heston/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "size" in data
    assert "max_size" in data


def test_clear_cache():
    """Test cache clearing."""
    response = client.delete("/api/v1/heston/cache")
    assert response.status_code == 200
    assert "message" in response.json()

    # Verify cache is empty
    stats = client.get("/api/v1/heston/cache/stats")
    assert stats.json()["size"] == 0
