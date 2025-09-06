"""
Shared pytest fixtures for the test suite.

Provides reusable test data and configurations.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from volatility_arbitrage.core.config import (
    Config,
    BacktestConfig,
    DataConfig,
    LoggingConfig,
    VolatilityConfig,
)
from volatility_arbitrage.core.types import (
    TickData,
    OptionChain,
    OptionContract,
    OptionType,
    Trade,
    Position,
    TradeType,
)


@pytest.fixture
def sample_tick_data() -> TickData:
    """Sample tick data for testing."""
    return TickData(
        timestamp=datetime(2024, 1, 1, 10, 0, 0),
        symbol="SPY",
        price=Decimal("450.50"),
        volume=1000000,
        bid=Decimal("450.45"),
        ask=Decimal("450.55"),
    )


@pytest.fixture
def sample_option_contract() -> OptionContract:
    """Sample option contract for testing."""
    return OptionContract(
        symbol="SPY",
        option_type=OptionType.CALL,
        strike=Decimal("450"),
        expiry=datetime(2024, 3, 15),
        price=Decimal("10.50"),
        bid=Decimal("10.45"),
        ask=Decimal("10.55"),
        volume=500,
        open_interest=1000,
        implied_volatility=Decimal("0.20"),
    )


@pytest.fixture
def sample_option_chain() -> OptionChain:
    """Sample option chain for testing."""
    timestamp = datetime(2024, 1, 1, 10, 0, 0)
    expiry = datetime(2024, 3, 15)

    calls = [
        OptionContract(
            symbol="SPY",
            option_type=OptionType.CALL,
            strike=Decimal(str(strike)),
            expiry=expiry,
            price=Decimal("10.00"),
            volume=100,
            open_interest=500,
        )
        for strike in [440, 445, 450, 455, 460]
    ]

    puts = [
        OptionContract(
            symbol="SPY",
            option_type=OptionType.PUT,
            strike=Decimal(str(strike)),
            expiry=expiry,
            price=Decimal("8.00"),
            volume=100,
            open_interest=500,
        )
        for strike in [440, 445, 450, 455, 460]
    ]

    return OptionChain(
        symbol="SPY",
        timestamp=timestamp,
        expiry=expiry,
        underlying_price=Decimal("450"),
        calls=calls,
        puts=puts,
        risk_free_rate=Decimal("0.05"),
    )


@pytest.fixture
def sample_trade() -> Trade:
    """Sample trade for testing."""
    return Trade(
        timestamp=datetime(2024, 1, 1, 10, 0, 0),
        symbol="SPY",
        trade_type=TradeType.BUY,
        quantity=100,
        price=Decimal("450.50"),
        commission=Decimal("1.00"),
        trade_id="trade_001",
    )


@pytest.fixture
def sample_position() -> Position:
    """Sample position for testing."""
    return Position(
        symbol="SPY",
        quantity=100,
        avg_entry_price=Decimal("450.00"),
        current_price=Decimal("455.00"),
        last_update=datetime(2024, 1, 1, 10, 0, 0),
    )


@pytest.fixture
def default_config() -> Config:
    """Default configuration for testing."""
    return Config(
        data=DataConfig(
            source="yahoo",
            cache_dir=Path("data/test_cache"),
            symbols=["SPY", "QQQ"],
        ),
        backtest=BacktestConfig(
            initial_capital=Decimal("100000"),
            commission_rate=Decimal("0.001"),
            slippage=Decimal("0.0005"),
        ),
        volatility=VolatilityConfig(
            method="garch",
            lookback_period=30,
        ),
        logging=LoggingConfig(
            level="WARNING",  # Reduce noise in tests
            format="json",
            console_output=False,
        ),
    )


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Sample market data for backtesting."""
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")

    data = []
    for i, date in enumerate(dates):
        # Create simple trending price data
        price = 100 + i * 0.5
        data.append(
            {
                "timestamp": date,
                "symbol": "SPY",
                "open": price - 0.5,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000000,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Sample returns series for testing."""
    np.random.seed(42)
    import numpy as np

    # Generate random returns with slight positive drift
    returns = np.random.normal(0.001, 0.02, 252)
    return pd.Series(returns, index=pd.date_range(start="2024-01-01", periods=252, freq="D"))


@pytest.fixture
def sample_equity_curve() -> pd.DataFrame:
    """Sample equity curve for testing."""
    import numpy as np

    np.random.seed(42)

    dates = pd.date_range(start="2024-01-01", periods=252, freq="D")
    initial_capital = 100000

    # Generate equity curve with drift and volatility
    returns = np.random.normal(0.001, 0.02, 252)
    equity = initial_capital * (1 + returns).cumprod()

    return pd.DataFrame(
        {
            "timestamp": dates,
            "cash": initial_capital * 0.3,  # 30% cash
            "positions_value": equity * 0.7,  # 70% in positions
            "total_equity": equity,
        }
    )


@pytest.fixture
def temp_config_file(tmp_path: Path, default_config: Config) -> Path:
    """Create temporary config file for testing."""
    config_path = tmp_path / "test_config.yaml"
    default_config.to_yaml(config_path)
    return config_path


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
