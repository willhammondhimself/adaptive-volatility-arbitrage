"""
Configuration management for the backtesting engine.

Loads and validates configuration from YAML files using Pydantic.
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict
from decimal import Decimal


class DataConfig(BaseModel):
    """Data fetching configuration."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(default="yahoo", description="Data source (yahoo, csv, etc.)")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")
    start_date: Optional[str] = Field(None, description="Backtest start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Backtest end date (YYYY-MM-DD)")
    symbols: list[str] = Field(default_factory=list, description="Symbols to fetch")

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Ensure symbols are uppercase."""
        return [s.upper() for s in v]


class BacktestConfig(BaseModel):
    """Backtesting engine configuration."""

    model_config = ConfigDict(frozen=True)

    initial_capital: Decimal = Field(
        default=Decimal("100000"), description="Starting capital", gt=0
    )
    commission_rate: Decimal = Field(
        default=Decimal("0.001"), description="Commission rate (0.001 = 0.1%)", ge=0, le=1
    )
    slippage: Decimal = Field(
        default=Decimal("0.0005"), description="Slippage estimate", ge=0, le=1
    )
    position_size_pct: Decimal = Field(
        default=Decimal("0.1"), description="Max position size as % of capital", gt=0, le=1
    )
    max_positions: int = Field(
        default=10, description="Maximum concurrent positions", gt=0
    )
    risk_free_rate: Decimal = Field(
        default=Decimal("0.05"), description="Annual risk-free rate", ge=0, le=1
    )


class VolatilityConfig(BaseModel):
    """Volatility forecasting configuration."""

    model_config = ConfigDict(frozen=True)

    method: str = Field(
        default="garch", description="Forecasting method (garch, ewma, historical)"
    )
    lookback_period: int = Field(default=30, description="Historical lookback days", gt=0)
    ewma_lambda: Decimal = Field(
        default=Decimal("0.94"), description="EWMA decay factor", gt=0, lt=1
    )
    garch_p: int = Field(default=1, description="GARCH(p,q) p parameter", ge=1, le=5)
    garch_q: int = Field(default=1, description="GARCH(p,q) q parameter", ge=1, le=5)

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Ensure valid forecasting method."""
        valid_methods = {"garch", "ewma", "historical"}
        v_lower = v.lower()
        if v_lower not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v_lower


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(frozen=True)

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log format (json, text)")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    console_output: bool = Field(default=True, description="Enable console logging")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Ensure valid logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Level must be one of {valid_levels}")
        return v_upper

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Ensure valid log format."""
        valid_formats = {"json", "text"}
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(f"Format must be one of {valid_formats}")
        return v_lower


class Config(BaseModel):
    """
    Main configuration container.

    Aggregates all sub-configurations for the backtesting engine.
    """

    model_config = ConfigDict(frozen=True)

    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling Decimal and Path types
        data = self.model_dump(mode="python")

        def convert_for_yaml(obj: Any) -> Any:
            """Convert non-serializable types for YAML."""
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_yaml(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_yaml(item) for item in obj]
            return obj

        yaml_data = convert_for_yaml(data)

        with open(path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Optional path to config file. If None, uses default.yaml

    Returns:
        Validated Config instance
    """
    if config_path is None:
        config_path = Path("config/default.yaml")

    if config_path.exists():
        return Config.from_yaml(config_path)
    else:
        # Return default configuration
        return Config()
