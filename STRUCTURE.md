# Repository Structure

## Overview

```
.
├── backend/              # FastAPI REST API
│   ├── api/             # API endpoints
│   ├── services/        # Business logic
│   ├── schemas/         # Request/response models
│   ├── core/            # Core utilities
│   └── tests/           # API tests
│
├── frontend/            # React dashboard
│   ├── src/
│   │   ├── api/        # API client
│   │   ├── components/ # UI components
│   │   ├── pages/      # Page components
│   │   ├── store/      # State management
│   │   └── hooks/      # Custom hooks
│   └── public/
│
├── src/                 # Core trading engine
│   └── volatility_arbitrage/
│       ├── backtest/   # Backtesting engine
│       ├── strategy/   # Trading strategies
│       ├── models/     # Pricing models
│       ├── data/       # Data fetchers
│       ├── core/       # Core infrastructure
│       └── utils/      # Utilities
│
├── research/            # Research & analysis
│   └── lib/
│       ├── heston_fft.py      # Heston FFT pricer
│       ├── validation.py      # Validation tools
│       ├── black_scholes.py   # BS model
│       └── surface_visualizer.py
│
├── execution/           # Execution systems
│   ├── gateway/        # C++ low-latency gateway
│   └── simulator/      # Backtesting execution
│
├── tests/              # Test suite
│   ├── test_backtest/
│   ├── test_core/
│   ├── test_data/
│   ├── test_models/
│   └── test_strategy/
│
├── config/             # Configuration files
│   ├── default.yaml
│   └── volatility_arb.yaml
│
├── docs/               # Documentation
│   ├── guides/        # User guides
│   ├── api/           # API documentation
│   └── PROJECT_STATUS.md
│
└── logs/              # Application logs
```

## Key Directories

### `/backend` - Web API
FastAPI-based REST API for the interactive dashboard.

### `/frontend` - Web Dashboard
React + Plotly.js interactive visualization.

### `/src/volatility_arbitrage` - Core Engine
Main trading system with backtesting, strategies, and models.

### `/research` - Research Code
Pricing libraries and validation tools.

### `/execution` - Execution Layer
Low-latency C++ gateway and execution simulators.

### `/tests` - Test Suite
Comprehensive unit and integration tests.

### `/docs` - Documentation
All project documentation and guides.