# Architecture Documentation

## System Overview

The Adaptive Volatility Arbitrage Backtesting Engine is designed as a modular, event-driven system for backtesting quantitative trading strategies with a focus on volatility arbitrage.

## Design Principles

### 1. Type Safety First
- Full type hints on all functions and methods
- Pydantic models for data validation
- Immutable data structures where appropriate
- mypy-compatible codebase

### 2. Configuration-Driven
- YAML configuration with Pydantic validation
- No hardcoded parameters
- Environment-specific configs supported
- Fail-fast validation on startup

### 3. Functional Core, Imperative Shell
- Pure functions for pricing and calculations
- Side effects isolated to I/O boundaries
- Testable business logic
- Deterministic behavior for backtesting

### 4. Observability
- Structured JSON logging
- Comprehensive metrics tracking
- Clear error messages
- Audit trail for all trades

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User / Strategy                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backtest Engine                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Daily Loop:                                          │  │
│  │  1. Update positions with market prices              │  │
│  │  2. Generate signals from strategy                   │  │
│  │  3. Execute trades with transaction costs            │  │
│  │  4. Track P&L and equity                             │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────┬────────────────────┬─────────────────────────────┘
           │                    │
           ▼                    ▼
┌──────────────────┐   ┌──────────────────────┐
│   Data Layer     │   │   Models Layer       │
│                  │   │                      │
│  • DataFetcher   │   │  • Black-Scholes    │
│  • YahooFinance  │   │  • Volatility       │
│  • Caching       │   │  • Greeks           │
└──────────────────┘   └──────────────────────┘
           │                    │
           ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Types                                │
│  TickData, OptionChain, Trade, Position, Config             │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Core Layer (`core/`)

**Purpose**: Foundation types and configuration

**Components**:

1. **types.py** - Pydantic models for all data structures
   - `TickData`: Market tick with bid/ask
   - `OptionContract`: Single option specification
   - `OptionChain`: Complete chain with calls/puts
   - `Trade`: Executed trade record
   - `Position`: Current position tracking
   - All models are immutable (frozen) except Position

2. **config.py** - Configuration management
   - `Config`: Root configuration container
   - `DataConfig`: Data source configuration
   - `BacktestConfig`: Engine parameters
   - `VolatilityConfig`: Forecasting parameters
   - `LoggingConfig`: Logging setup
   - YAML loading and validation
   - Type-safe configuration access

**Design Decisions**:
- Immutable data structures prevent accidental mutations
- Decimal type for financial calculations (avoid float precision issues)
- Comprehensive validation at construction time
- Self-documenting through type hints and docstrings

### Data Layer (`data/`)

**Purpose**: Market data acquisition and management

**Components**:

1. **fetcher.py** - Abstract data fetcher interface
   ```python
   class DataFetcher(ABC):
       @abstractmethod
       def fetch_historical_data(...)
       @abstractmethod
       def fetch_option_chain(...)
       @abstractmethod
       def fetch_current_price(...)
       @abstractmethod
       def fetch_risk_free_rate(...)
   ```

2. **yahoo.py** - YahooFinance implementation
   - Uses yfinance library
   - Implements caching for performance
   - Handles API errors gracefully
   - Validates data completeness

**Extension Points**:
- Implement `DataFetcher` interface for new data sources
- Examples: Bloomberg, Interactive Brokers, CSV files
- Caching strategy is pluggable

**Data Flow**:
```
YahooFinance API → yfinance library → YahooFinanceFetcher
                                              │
                                              ▼
                                    Validation + Caching
                                              │
                                              ▼
                                        TickData/OptionChain
```

### Models Layer (`models/`)

**Purpose**: Financial models and calculations

**Components**:

1. **black_scholes.py** - Option pricing
   - `BlackScholesModel.price()`: Calculate theoretical price
   - `BlackScholesModel.greeks()`: Calculate all Greeks
   - `calculate_implied_volatility()`: Newton-Raphson IV solver
   - Pure functions, no side effects
   - Uses scipy for numerical stability

2. **volatility.py** - Volatility forecasting
   - `HistoricalVolatility`: Simple historical std dev
   - `EWMAVolatility`: Exponentially weighted moving average
   - `GARCHVolatility`: GARCH(p,q) with MLE estimation
   - Abstract `VolatilityForecaster` base class

**Mathematical Foundations**:

Black-Scholes Formula:
```
C = S*N(d1) - K*e^(-rT)*N(d2)
P = K*e^(-rT)*N(-d2) - S*N(-d1)

where:
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

GARCH(1,1):
```
σ²(t+1) = ω + α*ε²(t) + β*σ²(t)

where ω > 0, α ≥ 0, β ≥ 0, α + β < 1
```

**Design Decisions**:
- Separated pricing from data structures
- All calculations use Decimal for precision
- Graceful degradation (fallbacks for edge cases)
- Comprehensive numerical stability checks

### Strategy Layer (`strategy/`)

**Purpose**: Trading signal generation

**Components**:

1. **base.py** - Strategy interface
   ```python
   class Strategy(ABC):
       @abstractmethod
       def generate_signals(timestamp, market_data, positions):
           """Return list of Signal objects"""
   ```

2. **Signal** - Trading signal data class
   - symbol: str
   - action: "buy" | "sell"
   - quantity: int
   - reason: str (optional explanation)

**Extension Points**:
- Implement `Strategy` interface for new strategies
- Examples:
  - Volatility arbitrage (buy underpriced, sell overpriced)
  - Delta-neutral strategies
  - Statistical arbitrage
  - Mean reversion

**Strategy Lifecycle**:
```
on_backtest_start()
    │
    ▼
Daily Loop: generate_signals()
    │
    ▼
on_trade_executed() (callback)
    │
    ▼
on_backtest_end()
```

### Backtest Layer (`backtest/`)

**Purpose**: Event-driven simulation engine

**Components**:

1. **engine.py** - Core backtest engine
   - Event-driven architecture
   - Daily processing loop
   - Position tracking
   - Transaction cost modeling
   - P&L calculation
   - Equity curve generation

2. **metrics.py** - Performance analytics
   - Return metrics (total, annualized)
   - Risk metrics (volatility, drawdown)
   - Risk-adjusted metrics (Sharpe, Sortino, Calmar)
   - Trade statistics (win rate, profit factor)

**Engine State Machine**:
```
[Initialize]
    │
    ▼
[Load Market Data]
    │
    ▼
┌───────────────────┐
│   Daily Loop      │
│                   │
│ Update Positions  │─┐
│        │          │ │
│        ▼          │ │
│ Generate Signals  │ │ Repeat for
│        │          │ │ each day
│        ▼          │ │
│ Execute Trades    │ │
│        │          │ │
│        ▼          │ │
│ Record Equity     │─┘
└───────────────────┘
    │
    ▼
[Close Positions]
    │
    ▼
[Calculate Metrics]
    │
    ▼
[Return Results]
```

**Transaction Cost Model**:
```python
execution_price = market_price * (1 ± slippage)
commission = trade_value * commission_rate
total_cost = execution_price * quantity ± commission
```

**Design Decisions**:
- Event-driven to match real trading
- Immutable trade records for audit trail
- Separate position tracking from trades
- Commission and slippage applied to all trades

### Multi-Asset Backtest Engine (`backtest/multi_asset_engine.py`)

**Purpose**: Extended backtest engine supporting simultaneous trading of stocks and options with portfolio-level Greeks tracking.

**Key Enhancements**:

1. **MultiAssetPosition Model** - Unified position tracking for stocks and options
   ```python
   class MultiAssetPosition(BaseModel):
       symbol: str
       asset_type: Literal["stock", "option"]
       quantity: int
       entry_price: Decimal
       current_price: Decimal
       last_update: datetime

       # Option-specific fields
       option_type: Optional[OptionType] = None
       strike: Optional[Decimal] = None
       expiry: Optional[datetime] = None
       underlying_price: Optional[Decimal] = None
       implied_volatility: Optional[Decimal] = None
       risk_free_rate: Optional[Decimal] = None
   ```

2. **PortfolioGreeks** - Aggregated risk metrics across all option positions
   ```python
   class PortfolioGreeks:
       delta: Decimal  # Directional exposure
       gamma: Decimal  # Delta sensitivity
       vega: Decimal   # Volatility sensitivity
       theta: Decimal  # Time decay
       rho: Decimal    # Interest rate sensitivity
   ```

**Multi-Asset Architecture**:

```
MultiAssetBacktestEngine
    │
    ├─► Stock Positions
    │   └─► Standard execution (commission + slippage)
    │
    ├─► Option Positions
    │   ├─► Options-specific execution (per-contract + slippage)
    │   ├─► Greeks calculation per position
    │   └─► Expiration monitoring
    │
    └─► Portfolio Aggregation
        ├─► Total P&L (stocks + options)
        ├─► Portfolio Greeks (sum of all options)
        └─► Equity curve with Greeks history
```

**Position-Level Greeks Calculation**:

For each option position, Greeks are calculated using Black-Scholes:
```python
def calculate_greeks(self) -> Optional[Greeks]:
    if not self.is_option:
        return None

    T = (self.expiry - self.last_update).days / 365

    return BlackScholesModel.greeks(
        S=self.underlying_price,
        K=self.strike,
        T=Decimal(str(T)),
        r=self.risk_free_rate,
        sigma=self.implied_volatility,
        option_type=self.option_type,
    )
```

**Portfolio Greeks Aggregation**:

Portfolio Greeks are the sum of all individual option position Greeks weighted by quantity:
```python
def _calculate_portfolio_greeks(self) -> PortfolioGreeks:
    total_delta = Decimal("0")
    total_gamma = Decimal("0")
    total_vega = Decimal("0")
    total_theta = Decimal("0")
    total_rho = Decimal("0")

    for position in self.multi_positions.values():
        greeks = position.calculate_greeks()
        if greeks:
            # Weight by quantity and option multiplier (100)
            multiplier = abs(position.quantity) * 100
            total_delta += greeks.delta * multiplier
            total_gamma += greeks.gamma * multiplier
            total_vega += greeks.vega * multiplier
            total_theta += greeks.theta * multiplier
            total_rho += greeks.rho * multiplier

    return PortfolioGreeks(
        delta=total_delta,
        gamma=total_gamma,
        vega=total_vega,
        theta=total_theta,
        rho=total_rho,
    )
```

**Options-Specific Execution Model**:

Options have different execution costs than stocks:

| Aspect | Stocks | Options |
|--------|--------|---------|
| **Commission** | Percentage of trade value | Per-contract flat fee ($0.65) |
| **Slippage** | Percentage of price (0.05%) | Percentage of premium (1%) |
| **Multiplier** | 1 | 100 (standard contract size) |
| **Market Value** | price × quantity | price × quantity × 100 |

```python
def _execute_option_signal(self, signal, current_price):
    quantity = signal.quantity
    premium = current_price

    # Apply slippage to premium
    if signal.action == "buy":
        execution_premium = premium * (1 + self.option_slippage_pct)
    else:
        execution_premium = premium * (1 - self.option_slippage_pct)

    # Calculate notional value (premium × quantity × 100)
    notional_value = execution_premium * abs(quantity) * 100

    # Per-contract commission
    commission = self.option_commission_per_contract * abs(quantity)

    # Total cost includes premium payment and commission
    if signal.action == "buy":
        total_cost = notional_value + commission
    else:
        total_cost = notional_value - commission

    return execution_premium, commission, total_cost
```

**Option Expiration Handling**:

Options are automatically closed 1 day before expiration to avoid exercise/assignment:

```python
def _handle_expirations(self, current_timestamp):
    """Close positions expiring within 1 day."""
    expiring_symbols = []

    for symbol, position in self.multi_positions.items():
        if position.is_option:
            days_to_expiry = (position.expiry - current_timestamp).days

            if days_to_expiry <= 1:
                expiring_symbols.append(symbol)

                # Create closing trade
                closing_signal = Signal(
                    symbol=symbol,
                    action="sell" if position.quantity > 0 else "buy",
                    quantity=abs(position.quantity),
                    reason=f"Option expiring on {position.expiry.date()}",
                )

                # Execute closing trade at current market price
                self._execute_signal(closing_signal, current_timestamp)

    # Remove expired positions
    for symbol in expiring_symbols:
        del self.multi_positions[symbol]
```

**Greeks Tracking Over Time**:

The engine maintains a historical record of portfolio Greeks alongside the equity curve:

```python
# During daily loop
portfolio_greeks = self._calculate_portfolio_greeks()
self.greeks_history.append({
    "timestamp": current_timestamp,
    "delta": portfolio_greeks.delta,
    "gamma": portfolio_greeks.gamma,
    "vega": portfolio_greeks.vega,
    "theta": portfolio_greeks.theta,
    "rho": portfolio_greeks.rho,
})

# Result includes Greeks in equity curve
result.equity_curve["portfolio_delta"] = greeks_df["delta"]
result.equity_curve["portfolio_vega"] = greeks_df["vega"]
```

**Market Value Calculation**:

Market value differs between asset types:

- **Stocks**: `price × quantity`
- **Options**: `price × quantity × 100` (standard multiplier)

```python
@property
def market_value(self) -> Decimal:
    if self.asset_type == "stock":
        return self.current_price * abs(self.quantity)
    else:  # option
        return self.current_price * abs(self.quantity) * Decimal("100")
```

**Unrealized P&L Calculation**:

P&L considers position direction (long/short) and asset type:

```python
@property
def unrealized_pnl(self) -> Decimal:
    price_diff = self.current_price - self.entry_price

    if self.asset_type == "stock":
        return price_diff * self.quantity
    else:  # option
        return price_diff * self.quantity * Decimal("100")
```

**Integration with Volatility Arbitrage Strategy**:

The multi-asset engine enables delta-neutral volatility arbitrage:

1. **Strategy generates option signals** (buy/sell straddles)
2. **Engine executes option trades** with appropriate costs
3. **Greeks are calculated** for each position
4. **Portfolio delta is monitored** for hedging needs
5. **Strategy generates hedge signals** (trade underlying stock)
6. **Engine maintains delta-neutral exposure**

This architecture enables sophisticated strategies that:
- Trade options for volatility exposure
- Hedge directional risk with underlying
- Monitor portfolio-level Greeks in real-time
- Close positions before expiration
- Track performance with full Greeks history

### Utilities Layer (`utils/`)

**Purpose**: Cross-cutting concerns

**Components**:

1. **logging.py** - Structured logging
   - `JSONFormatter`: Machine-readable logs
   - `TextFormatter`: Human-readable console output
   - `setup_logging()`: Configure logging system
   - `get_logger()`: Get logger instance
   - `get_contextual_logger()`: Logger with extra fields

**Log Structure**:
```json
{
  "timestamp": "ISO-8601",
  "level": "INFO|WARNING|ERROR",
  "logger": "module.name",
  "message": "Human readable message",
  "module": "filename",
  "function": "function_name",
  "line": 123,
  "extra_fields": {...}
}
```

## Data Flow

### Backtest Execution Flow

```
1. Configuration
   Config YAML → Pydantic Validation → Config Object

2. Data Loading
   YahooFinance API → DataFetcher → Historical Data → Validation

3. Backtest Loop (Daily)
   For each trading day:
   a. Market Data → Update Positions (mark-to-market)
   b. Strategy.generate_signals(data, positions) → Signals
   c. For each Signal:
      - Check cash availability
      - Apply slippage
      - Calculate commission
      - Execute trade
      - Update position
      - Record trade
   d. Calculate equity
   e. Record to equity curve

4. Results
   Trades + Equity Curve → Metrics Calculation → BacktestResult
```

### Option Pricing Flow

```
Market Data → Option Contract
                     │
                     ▼
              ┌──────────────┐
              │ Black-Scholes│
              └──────┬───────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   Theoretical Price         Greeks
         │                       │
         ▼                       ▼
   Compare to Market        Risk Management
         │
         ▼
   Trading Signal
```

### Volatility Forecasting Flow

```
Price Data → Calculate Returns → Select Method
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              Historical           EWMA              GARCH(1,1)
                    │                  │                  │
                    │                  │                  ▼
                    │                  │            MLE Estimation
                    │                  │                  │
                    └──────────────────┴──────────────────┘
                                       │
                                       ▼
                              Volatility Forecast
                                       │
                                       ▼
                              Option Pricing / Trading
```

## Extension Points

### Adding a New Data Source

1. Implement `DataFetcher` interface
2. Handle data validation
3. Implement caching if needed
4. Register in configuration

Example:
```python
class BloombergFetcher(DataFetcher):
    def fetch_historical_data(self, symbol, start, end):
        # Bloomberg API calls
        ...

    def fetch_option_chain(self, symbol, timestamp, expiry):
        # Bloomberg option data
        ...
```

### Adding a New Strategy

1. Inherit from `Strategy` base class
2. Implement `generate_signals()` method
3. Optionally implement lifecycle hooks
4. Configure in YAML

Example:
```python
class VolatilityArbitrageStrategy(Strategy):
    def generate_signals(self, timestamp, market_data, positions):
        signals = []

        # Calculate implied vs realized volatility
        # Generate buy/sell signals based on mispricing

        return signals
```

### Adding New Metrics

1. Add calculation function to `metrics.py`
2. Update `PerformanceMetrics` dataclass
3. Update `calculate_comprehensive_metrics()`
4. Update `print_metrics()` for display

## Performance Considerations

### Memory Management

- Immutable data structures prevent memory leaks
- Streaming processing for large datasets
- Lazy loading where appropriate
- Caching with TTL for data fetching

### Computational Efficiency

- Vectorized operations with NumPy/Pandas
- Compiled numerical libraries (scipy)
- Avoid Python loops in hot paths
- Profile before optimizing

### Scalability

Current design handles:
- ~1000 trading days
- ~100 instruments
- ~10,000 trades

For larger scale:
- Implement database backend for trades
- Use Dask for distributed processing
- Implement streaming backtest
- Parallelize across strategies

## Testing Strategy

### Test Pyramid

```
        /\
       /  \  E2E Tests (integration)
      /────\
     /      \ Integration Tests
    /────────\
   /          \ Unit Tests (80%+ coverage)
  /────────────\
```

### Test Categories

1. **Unit Tests** (`@pytest.mark.unit`)
   - Individual functions
   - Pydantic models
   - Pure calculations
   - Fast (<1ms each)

2. **Integration Tests** (`@pytest.mark.integration`)
   - Component interactions
   - Data fetching
   - Full backtest runs
   - Slower (<1s each)

3. **Fixtures** (`conftest.py`)
   - Sample data
   - Mock objects
   - Configuration
   - Reusable across tests

### Coverage Targets

- Overall: 80%+
- Core types: 95%+
- Models: 90%+
- Backtest engine: 85%+
- Utilities: 80%+

## Error Handling

### Error Hierarchy

```
Exception
  │
  ├── DataFetcherError
  │     └── DataNotFoundError
  │
  └── ValidationError (Pydantic)
```

### Error Handling Strategy

1. **Validation Errors**: Fail fast at boundaries
2. **Data Errors**: Retry with exponential backoff
3. **Calculation Errors**: Log and use fallback values
4. **System Errors**: Log and propagate

### Logging Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Unexpected but handled conditions
- **ERROR**: Error conditions that need attention
- **CRITICAL**: System-critical failures

## Security Considerations

1. **Data Validation**: All inputs validated with Pydantic
2. **Type Safety**: Type hints prevent type errors
3. **Immutability**: Prevents accidental data corruption
4. **Logging**: No sensitive data in logs
5. **Configuration**: Secrets in environment variables

## Future Enhancements

### Phase 1
- Portfolio-level risk management
- Multi-asset backtesting
- Advanced order types (limit, stop-loss)
- Optimization framework

### Phase 2
- Live trading integration
- Real-time monitoring dashboard
- Strategy optimization (walk-forward)
- Machine learning integration

### Phase 3
- Distributed backtesting
- Cloud deployment
- API for external access
- Advanced visualization

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity
- Sharpe, W. F. (1994). The Sharpe Ratio
