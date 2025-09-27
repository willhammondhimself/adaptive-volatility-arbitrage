# Adaptive Volatility Arbitrage Backtesting Engine

A production-grade quantitative finance system for backtesting volatility arbitrage strategies using Python 3.12.

## Features

- **Event-driven backtesting** with realistic transaction costs and slippage
- **Black-Scholes pricing** with full Greeks calculation
- **Volatility forecasting** using GARCH, EWMA, and historical methods
- **Comprehensive metrics** including Sharpe, Sortino, Calmar ratios and drawdown analysis
- **Type-safe** with Pydantic models and full type hints
- **Configuration-driven** using YAML with validation
- **Structured logging** in JSON format for production monitoring
- **Extensible architecture** with abstract interfaces for strategies and data sources

## Quick Start

### Installation

```bash
# Install dependencies (requires Python 3.12+)
pip install poetry
poetry install

# Or with pip
pip install -e .
```

### Basic Usage

```python
from datetime import datetime
from decimal import Decimal

from volatility_arbitrage.backtest import BacktestEngine
from volatility_arbitrage.core.config import load_config
from volatility_arbitrage.data import YahooFinanceFetcher
from volatility_arbitrage.strategy.base import BuyAndHoldStrategy

# Load configuration
config = load_config()

# Initialize components
data_fetcher = YahooFinanceFetcher()
strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
engine = BacktestEngine(config=config.backtest, strategy=strategy)

# Fetch historical data
market_data = data_fetcher.fetch_historical_data(
    symbol="SPY",
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 1, 1),
)

# Run backtest
result = engine.run(market_data)

# Print results
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Number of Trades: {result.num_trades}")
```

### Phase 1: Volatility Arbitrage Strategy

**NEW**: Run a complete volatility arbitrage backtest with a single command:

```bash
python scripts/run_vol_arb_backtest.py
```

This demonstrates the full system capabilities:
- Fetches real SPY data and option chains
- Runs delta-neutral volatility arbitrage strategy
- Compares GARCH-forecasted volatility to implied volatility
- Generates comprehensive results and visualizations

**Example Output:**

```
BACKTEST RESULTS SUMMARY
======================================================================

  Period: 2023-01-01 to 2023-06-30
  Trading Days: 126

  RETURNS
    Initial Capital:     $  100,000.00
    Final Capital:       $  106,250.00
    Total Return:                 6.25%
    Annualized Return:           12.87%

  RISK METRICS
    Volatility (ann.):            8.45%
    Max Drawdown:                -3.12%
    Sharpe Ratio:                 1.52
    Sortino Ratio:                2.18
    Calmar Ratio:                 4.12

  TRADING
    Number of Trades:               47
    Win Rate:                    62.0%
    Profit Factor:                1.84

  Results saved to: results/
======================================================================
```

**Generated Files:**
- `results/backtest_results.json` - Complete performance metrics
- `results/equity_curve.png` - Equity and drawdown charts
- `results/greeks_evolution.png` - Portfolio Greeks over time
- `results/summary_table.png` - Performance metrics table

### Strategy Overview

The volatility arbitrage strategy:

1. **Volatility Forecasting**: Uses GARCH(1,1) to forecast realized volatility
2. **Signal Generation**: Identifies mispricing when IV-RV spread > 5%
3. **Position Construction**:
   - Long volatility: Buy ATM straddles when IV < RV
   - Short volatility: Sell ATM straddles when IV > RV
4. **Delta Hedging**: Maintains delta-neutral exposure through dynamic hedging
5. **Exit Rules**: Closes positions when spread converges (< 2%)

**Key Parameters** (from `config/volatility_arb.yaml`):
- Entry threshold: 5% IV-RV spread
- Exit threshold: 2% spread convergence
- Time to expiry: 14-60 days
- Delta rebalance: ±0.10 threshold
- Position size: 5% of capital per trade

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/volatility_arbitrage --cov-report=html

# Run specific test categories
poetry run pytest -m unit
poetry run pytest -m integration
```

## Project Structure

```
adaptive-volatility-arbitrage/
├── config/
│   └── default.yaml          # Default configuration
├── src/volatility_arbitrage/
│   ├── core/                 # Core types and config
│   │   ├── types.py         # Pydantic models
│   │   └── config.py        # Configuration management
│   ├── data/                 # Data fetching
│   │   ├── fetcher.py       # Abstract interface
│   │   └── yahoo.py         # YahooFinance implementation
│   ├── models/               # Pricing and forecasting
│   │   ├── black_scholes.py # BS pricing + Greeks
│   │   └── volatility.py    # Volatility models
│   ├── strategy/             # Trading strategies
│   │   └── base.py          # Strategy interface
│   ├── backtest/             # Backtesting engine
│   │   ├── engine.py        # Core engine
│   │   └── metrics.py       # Performance metrics
│   └── utils/
│       └── logging.py        # Structured logging
└── tests/                    # Test suite
    ├── conftest.py          # Shared fixtures
    ├── test_core/
    └── test_models/
```

## Configuration

Configuration is managed through YAML files with Pydantic validation. See [`config/default.yaml`](config/default.yaml) for the default configuration.

### Key Configuration Sections

**Data Configuration**
```yaml
data:
  source: yahoo
  cache_dir: data/cache
  start_date: "2020-01-01"
  end_date: "2024-01-01"
  symbols:
    - SPY
    - QQQ
```

**Backtest Configuration**
```yaml
backtest:
  initial_capital: 100000.0
  commission_rate: 0.001     # 0.1%
  slippage: 0.0005          # 0.05%
  position_size_pct: 0.1    # 10% max per position
  max_positions: 10
  risk_free_rate: 0.05      # 5% annual
```

**Volatility Configuration**
```yaml
volatility:
  method: garch              # garch, ewma, or historical
  lookback_period: 30
  ewma_lambda: 0.94
  garch_p: 1
  garch_q: 1
```

## Core Concepts

### Data Types

All data types are immutable Pydantic models with full validation:

- **TickData**: Market tick data with bid/ask spreads
- **OptionContract**: Individual option specifications
- **OptionChain**: Complete chain with calls and puts
- **Trade**: Executed trade records
- **Position**: Current position tracking

### Black-Scholes Pricing

```python
from decimal import Decimal
from volatility_arbitrage.models.black_scholes import BlackScholesModel
from volatility_arbitrage.core.types import OptionType

price = BlackScholesModel.price(
    S=Decimal("100"),      # Spot price
    K=Decimal("100"),      # Strike price
    T=Decimal("0.25"),     # Time to expiry (years)
    r=Decimal("0.05"),     # Risk-free rate
    sigma=Decimal("0.20"), # Volatility
    option_type=OptionType.CALL
)

greeks = BlackScholesModel.greeks(...)
print(f"Delta: {greeks.delta}")
print(f"Gamma: {greeks.gamma}")
print(f"Vega: {greeks.vega}")
```

### Volatility Forecasting

```python
from volatility_arbitrage.models.volatility import (
    HistoricalVolatility,
    EWMAVolatility,
    GARCHVolatility,
)

# Historical volatility
hist_vol = HistoricalVolatility(window=30)
forecast = hist_vol.forecast(returns)

# EWMA
ewma_vol = EWMAVolatility(lambda_param=Decimal("0.94"))
forecast = ewma_vol.forecast(returns)

# GARCH(1,1)
garch_vol = GARCHVolatility(p=1, q=1)
forecast = garch_vol.forecast(returns)
```

### Creating Custom Strategies

```python
from volatility_arbitrage.strategy.base import Strategy, Signal

class MyStrategy(Strategy):
    def generate_signals(self, timestamp, market_data, positions):
        signals = []

        # Your strategy logic here
        if should_buy:
            signals.append(Signal(
                symbol="SPY",
                action="buy",
                quantity=100,
                reason="Entry signal"
            ))

        return signals
```

## Performance Metrics

The engine calculates comprehensive performance metrics:

- **Return Metrics**: Total return, annualized return
- **Risk Metrics**: Volatility, maximum drawdown
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Statistics**: Win rate, average win/loss, profit factor

```python
from volatility_arbitrage.backtest.metrics import (
    calculate_comprehensive_metrics,
    print_metrics,
)

metrics = calculate_comprehensive_metrics(
    equity_curve=result.equity_curve,
    initial_capital=config.initial_capital,
    final_capital=result.final_capital,
)

print_metrics(metrics)
```

## Development

### Code Quality

The project maintains high code quality standards:

- **Type hints** on all functions (mypy-compatible)
- **Docstrings** in Google style
- **Linting** with Ruff
- **Formatting** with Black
- **Testing** with pytest (~80%+ coverage target)

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type check
poetry run mypy src/
```

### Testing

Tests are organized by module with shared fixtures in `conftest.py`:

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_models/test_black_scholes.py

# Run with coverage report
poetry run pytest --cov=src/volatility_arbitrage --cov-report=html
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design documentation including:

- Data flow diagrams
- Component interactions
- Extension points
- Design patterns used

## Logging

The system uses structured JSON logging for production monitoring:

```python
from volatility_arbitrage.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(config.logging)

# Get logger
logger = get_logger(__name__)

# Log with structured data
logger.info(
    "Trade executed",
    extra={
        "symbol": "SPY",
        "quantity": 100,
        "price": 450.50,
    }
)
```

Logs are written to both console (configurable format) and files (JSON):

```json
{
  "timestamp": "2024-01-01T10:00:00",
  "level": "INFO",
  "logger": "volatility_arbitrage.backtest.engine",
  "message": "Trade executed",
  "symbol": "SPY",
  "quantity": 100,
  "price": 450.50
}
```

## Roadmap

Phase 0 (Current):
- [x] Core infrastructure and types
- [x] Black-Scholes pricing and Greeks
- [x] Volatility forecasting (GARCH, EWMA, Historical)
- [x] Basic backtest engine
- [x] Performance metrics
- [x] Test suite

Phase 1 (Next):
- [ ] Volatility arbitrage strategies
- [ ] Portfolio optimization
- [ ] Risk management system
- [ ] Advanced order types
- [ ] Multi-asset support

Phase 2 (Future):
- [ ] Live trading integration
- [ ] Web dashboard for results
- [ ] Real-time monitoring
- [ ] Strategy optimization framework

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass
- Code is formatted with Black
- Type hints are present
- Docstrings are complete
- Test coverage remains >80%

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

- Built with Python 3.12
- Uses Pydantic for data validation
- Powered by yfinance for market data
- scipy for numerical computations
