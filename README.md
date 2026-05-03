# Adaptive Volatility Arbitrage Trading System

**Quantitative finance platform** for options pricing, volatility arbitrage, and interactive market analysis.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
[![Tests](https://img.shields.io/badge/Tests-274%20passing-success.svg)]()

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

FastAPI backend + React 18 + Plotly.js frontend with dark mode. Real-time parameter exploration with LRU caching (80% hit rate, <5ms cache hits).

**Pages**:
| Route | Page |
|-------|------|
| `/` | Surface Explorer (3D vol surface, parameter sliders) |
| `/options` | Heston Explorer (price surface + live IV solver) |
| `/options/bs` | Black-Scholes Playground (Greeks visualization, P&L heatmaps) |
| `/trading` | Backtest Dashboard (equity / drawdown, Greeks evolution, trade log) |
| `/trading/paper` | Paper Trading (mock-gateway order entry + position tracking) |
| `/delta-hedged` | Delta-Hedged Backtest (event-driven replay with portfolio Greeks) |

**API** (24+ endpoints across 8 routers — full schema at `/docs`):
| Router | Purpose |
|--------|---------|
| `heston` | Price surface, cache stats, IV solve |
| `options` | Black-Scholes pricing, P&L heatmaps, IV surfaces |
| `backtest` | Run backtest, Monte Carlo, delta-hedged variant |
| `market` | Live quote, option chain, VIX, market status |
| `paper_trading` | Start/stop session, trade list, P&L stats |
| `snapshots` | Capture / list / fetch market snapshots |
| `forecast` | GARCH + Bayesian-LSTM volatility forecasts |
| `costs` | Slippage and commission estimation |

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

### Strategy

`VolatilityArbitrageStrategy` ([src/volatility_arbitrage/strategy/volatility_arbitrage.py](src/volatility_arbitrage/strategy/volatility_arbitrage.py)) supports two modes selected via `use_qv_strategy`:

- **Classic IV-vs-RV** — forecasts realized vol with GARCH(1,1) or a Bayesian LSTM, sells straddles when implied exceeds the forecast by a regime-aware threshold, holds delta-neutral via daily rebalancing, exits on convergence or stop-loss.
- **QV 6-signal consensus** — weighted score over IV skew, put/call ratio, IV-premium percentile, term-structure slope, volume ratio, and near-term sentiment. Trades on z-score extremes of the smoothed (EMA) consensus.

Risk controls (apply to both modes): tiered profit-taking (25/50/75% of P&L → 33/33/34% close), stop-loss on option-leg P&L only, delta rebalancing with 100x option multiplier and per-leg strike lookup, holding-period gate for discretionary exits (stop-loss bypasses), regime-adaptive sizing (HMM-classified vol regime), uncertainty-scaled sizing from Bayesian-LSTM epistemic variance.

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
tests/                   274 tests across 10 subdirectories
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

274 tests across pricing models (Heston, Black-Scholes, Bayesian LSTM), the multi-asset backtest engine, strategy logic (entries, exits, profit-taking, delta rebalancing, EMA smoothing), the real options data loader cache, and execution / risk modules.

---

## Up-Next

**Pricing model**:
- Discrete dividend handling (currently assumes continuous yield)
- American-option early-exercise premium

**Backtester**:
- Bid-ask spread modeling for options (currently fills at mid)
- Square-root market impact for size-aware fills
- Continuous Greeks tracking between trades (currently at trade time + daily MTM)

**Data**:
- Extend SPY options coverage to 2022-2023 and to additional underlyings
- Tick-level IV reconstruction (currently EOD snapshots)

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

