# Adaptive Volatility Arbitrage Trading System

**Quantitative finance platform** for options pricing, volatility arbitrage, and interactive market analysis.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
[![Tests](https://img.shields.io/badge/Tests-71%20passing-success.svg)]()

---

## Overview

Volatility arbitrage system that exploits mispricing between implied and realized volatility.

**Core components**:
- Heston FFT pricer (0.00-0.03% error after fixing Carr-Madan implementation bugs)
- Event-driven backtester with portfolio Greeks
- Interactive dashboard for parameter exploration

---

## Getting Started

### Backend API
```bash
PYTHONPATH=./src:. python3 backend/main.py
# Backend at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Dashboard
```bash
cd frontend
npm install  # First time only
npm run dev
# Dashboard at http://localhost:5173
```

### Tests
```bash
PYTHONPATH=./src:. python3 -m pytest tests/ -v
```

---

## Background: Volatility Arbitrage

Volatility arbitrage trades the *magnitude* of price movement rather than direction. The edge comes from forecasting realized volatility more accurately than the market's implied volatility.

**Basic mechanics**:
- Buy straddles when your forecast > implied vol
- Delta hedge daily to stay directionally neutral
- Profit via gamma scalping as the underlying moves

**P&L driver**: `realized_vol > implied_vol + theta_cost`

---

## Architecture

### Heston FFT Pricer

Heston (1993) stochastic volatility model with Carr-Madan (1999) FFT pricing. Implementation at [research/lib/heston_fft.py](research/lib/heston_fft.py).

**Accuracy** (after fixing Carr-Madan bugs, see below):
- ATM: 0.0000% error
- ITM/OTM: 0.0002-0.03% error
- Speed: 10-100x faster than numerical integration

```python
from research.lib.heston_fft import HestonFFT

heston = HestonFFT(
    v0=0.04, theta=0.05, kappa=2.0,
    sigma_v=0.3, rho=-0.7, r=0.05, q=0.02
)
call = heston.price(S=100, K=110, T=1.0, option_type="call")
```

### Dashboard

FastAPI backend + React 18 + Plotly.js frontend. Real-time parameter exploration with LRU caching (80% hit rate, <5ms cache hits).

**API**:
```
POST   /api/v1/heston/price-surface
GET    /api/v1/heston/cache/stats
DELETE /api/v1/heston/cache
```

### Backtester

Event-driven multi-asset engine at [src/volatility_arbitrage/backtest/multi_asset_engine.py](src/volatility_arbitrage/backtest/multi_asset_engine.py).

- Portfolio Greeks (delta, gamma, vega, theta)
- Option expiration handling
- Commission and slippage modeling

```python
from volatility_arbitrage.backtest.multi_asset_engine import MultiAssetEngine

engine = MultiAssetEngine(initial_capital=100000, commission=0.50, slippage=0.01)
engine.execute_trade(symbol="SPY", asset_type="option", quantity=10, price=12.50, ...)
greeks = engine.portfolio_greeks(spot=450, vol=0.20, r=0.05, q=0.02)
```

### Strategy Framework

Base class at [src/volatility_arbitrage/strategy/base.py](src/volatility_arbitrage/strategy/base.py).

**Included strategies**:
- Volatility mean reversion (z-score extremes)
- Realized vs implied (classic vol arb)
- Term structure arbitrage
- Delta-neutral vol trading (gamma scalping)

---

## Repository Structure

```
backend/                 FastAPI REST API
frontend/                React + Plotly.js dashboard
src/volatility_arbitrage/
  ├── backtest/          Event-driven backtesting
  ├── strategy/          Trading strategies
  ├── models/            Heston, Black-Scholes
  └── core/              Types, config
research/lib/            Pricing implementations (heston_fft.py)
tests/                   71 tests
config/                  YAML configurations
```

---

## Installation

```bash
# Backend
pip install fastapi uvicorn numpy pandas scipy pydantic pyyaml
# or: poetry install

# Frontend
cd frontend && npm install
```

---

## Performance

| Component | Latency |
|-----------|---------|
| FFT pricing (cache miss) | 150-300ms |
| FFT pricing (cache hit) | <5ms |
| Greeks calculation | <1ms/position |
| Backtest (1 year daily) | 2-5s |

---

## Testing

```bash
PYTHONPATH=./src:. python3 -m pytest tests/ -v
```

71 tests: multi-asset engine (14), core types (24), Black-Scholes (16), Heston (17).

---

## Known Limitations

**Pricing model**:
- Market impact modeled linearly (no square-root impact)
- No discrete dividend handling (assumes continuous yield)
- Heston calibration assumes stationary vol-of-vol

**Backtester**:
- Fill simulation assumes immediate execution at mid
- No bid-ask spread modeling for options
- Greeks computed at trade time, not continuously

**Data**:
- Historical IV from EOD snapshots, not tick-level
- No handling for option early exercise (American vs European)

---

## Changelog

### The Carr-Madan Sign Bug (Nov 2024)

Spent three days chasing a bug where deep OTM puts were pricing 50-200% wrong.
The FFT output looked fine, the characteristic function matched Gatheral's book,
but prices were garbage.

Turned out to be a sign error in the damping factor: `exp(-1j*b*v)` should be
`exp(+1j*b*v)`. The original Carr-Madan paper has `exp(-alpha*k)` for the call
transform, but when you work through the inverse FFT, the sign flips. I was
copying from a QuantLib forum post that had it wrong.

Also found the grid spacing was off. Used `b = lambda/2` (from some tutorial)
instead of `b = pi/eta`. After fixing both, errors dropped from 30-200% to
under 0.03%.

See `research/lib/heston_fft.py:L142-L168` for the corrected implementation.

### Dec 2024: QV Strategy

6-signal consensus strategy for volatility arbitrage. Walk-forward validated
across 5 folds (2019-2024), Monte Carlo bootstrap confirms Sharpe 95% CI
entirely above 1.0.

```bash
PYTHONPATH=./src:. python scripts/run_backtest.py
PYTHONPATH=./src:. python scripts/run_walkforward.py
PYTHONPATH=./src:. python scripts/run_monte_carlo.py --block-bootstrap
```

### Nov 2024: Dashboard

FastAPI + React dashboard for Heston parameter exploration. LRU cache gives
80% hit rate on repeated queries.

---

## Data

Historical options data from public sources (OptionMetrics, CBOE). Raw files excluded from version control.

---

## References

- Heston (1993), "A Closed-Form Solution for Options with Stochastic Volatility"
- Carr & Madan (1999), "Option Valuation using the Fast Fourier Transform"
- Lord & Kahl (2006), "Optimal Fourier Inversion in Semi-Analytical Option Pricing"

---

## License

MIT

