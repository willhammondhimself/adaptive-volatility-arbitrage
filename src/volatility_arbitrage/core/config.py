"""
Configuration management for the backtesting engine.

Loads and validates configuration from YAML files using Pydantic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict
from decimal import Decimal


# Import strategy config types (avoid circular imports by using forward reference)
@dataclass
class RegimeParameters:
    """Regime-specific strategy parameters."""

    regime_id: int
    entry_threshold_pct: Decimal
    exit_threshold_pct: Decimal
    position_size_multiplier: Decimal = Decimal("1.0")
    max_vega_multiplier: Decimal = Decimal("1.0")


@dataclass
class VolatilityArbitrageConfig:
    """Configuration for volatility arbitrage strategy."""

    # Entry/exit thresholds (baseline, used when no regime detection)
    entry_threshold_pct: Decimal = Decimal("5.0")
    exit_threshold_pct: Decimal = Decimal("2.0")

    # Time constraints
    min_days_to_expiry: int = 14
    max_days_to_expiry: int = 60

    # Delta hedging
    delta_rebalance_threshold: Decimal = Decimal("0.10")
    delta_target: Decimal = Decimal("0.0")

    # Position sizing
    position_size_pct: Decimal = Decimal("5.0")
    max_vega_exposure: Decimal = Decimal("1000")
    max_positions: int = 5

    # Volatility forecasting
    vol_lookback_period: int = 30
    vol_forecast_method: str = "garch"

    # Risk management
    max_loss_pct: Decimal = Decimal("50.0")

    # Regime detection (optional)
    use_regime_detection: bool = False
    regime_params: Optional[dict[int, RegimeParameters]] = None
    regime_lookback_period: int = 60
    exit_on_regime_transition: bool = False

    # QV Strategy Toggle
    use_qv_strategy: bool = False

    # QV Feature Windows
    rv_window: int = 20
    feature_window: int = 60
    regime_window: int = 252

    # QV Signal Thresholds
    pc_ratio_threshold: Decimal = Decimal("1.0")
    skew_threshold: Decimal = Decimal("0.05")
    premium_threshold: Decimal = Decimal("0.10")
    term_structure_threshold: Decimal = Decimal("0.0")
    volume_spike_threshold: Decimal = Decimal("1.5")
    sentiment_threshold: Decimal = Decimal("-0.05")

    # QV Consensus Scoring
    consensus_threshold: Decimal = Decimal("0.2")

    # QV Signal Weights (must sum to 1.0)
    weight_pc_ratio: Decimal = Decimal("0.20")
    weight_iv_skew: Decimal = Decimal("0.20")
    weight_iv_premium: Decimal = Decimal("0.15")
    weight_term_structure: Decimal = Decimal("0.15")
    weight_volume_spike: Decimal = Decimal("0.15")
    weight_near_term_sentiment: Decimal = Decimal("0.15")

    # QV Regime Scalars
    regime_crisis_scalar: Decimal = Decimal("0.5")
    regime_elevated_scalar: Decimal = Decimal("0.75")
    regime_normal_scalar: Decimal = Decimal("1.0")
    regime_low_scalar: Decimal = Decimal("1.2")
    regime_extreme_low_scalar: Decimal = Decimal("1.5")

    # Bullish Base Exposure Parameters
    base_long_bias: Decimal = Decimal("0.8")
    signal_adjustment_factor: Decimal = Decimal("0.7")

    # Tiered Position Sizing
    use_tiered_sizing: bool = True
    min_consensus_threshold: Decimal = Decimal("0.15")
    position_scaling_method: str = "quadratic"
    min_holding_days: int = 5

    # Leverage Configuration (Phase 2)
    use_leverage: bool = False
    short_vol_leverage: Decimal = Decimal("1.3")
    long_vol_leverage: Decimal = Decimal("2.0")
    max_leveraged_notional_pct: Decimal = Decimal("0.80")
    leverage_drawdown_reduction: bool = True
    leverage_dd_threshold: Decimal = Decimal("0.10")

    # Bayesian LSTM Volatility Forecasting (Phase 2)
    bayesian_lstm_hidden_size: int = 64
    bayesian_lstm_dropout_p: float = 0.2
    bayesian_lstm_sequence_length: int = 20
    bayesian_lstm_n_mc_samples: int = 50

    # Uncertainty-Adjusted Position Sizing (Phase 2)
    use_uncertainty_sizing: bool = False
    uncertainty_penalty: float = 2.0
    uncertainty_min_position_pct: float = 0.01
    uncertainty_max_position_pct: float = 0.15


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
    # Stock trading costs
    commission_rate: Decimal = Field(
        default=Decimal("0.001"), description="Stock commission rate (0.001 = 0.1%)", ge=0, le=1
    )
    slippage: Decimal = Field(
        default=Decimal("0.001"), description="Stock slippage estimate (10 bps)", ge=0, le=1
    )
    # Option-specific trading costs (realistic for vol arb)
    option_spread: Decimal = Field(
        default=Decimal("0.05"), description="Option bid-ask spread as % of premium (5%)", ge=0, le=1
    )
    option_commission_per_contract: Decimal = Field(
        default=Decimal("0.65"), description="Per-contract option commission ($0.65)", ge=0
    )
    daily_hedge_cost: Decimal = Field(
        default=Decimal("0.0002"), description="Daily delta hedge rebalancing cost (2 bps)", ge=0, le=1
    )
    margin_rate: Decimal = Field(
        default=Decimal("0.05"), description="Annual margin financing rate (5%)", ge=0, le=1
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

    # Square-Root Impact Model (Phase 2)
    use_impact_model: bool = Field(
        default=False, description="Use square-root market impact model instead of fixed slippage"
    )
    impact_half_spread_bps: Decimal = Field(
        default=Decimal("5.0"), description="Half bid-ask spread in basis points", ge=0
    )
    impact_coefficient: Decimal = Field(
        default=Decimal("0.1"), description="Market impact coefficient", ge=0
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


def load_strategy_config(config_path: Optional[Path] = None) -> VolatilityArbitrageConfig:
    """
    Load volatility arbitrage strategy configuration from YAML file.

    Args:
        config_path: Optional path to config file. If None, uses volatility_arb.yaml

    Returns:
        VolatilityArbitrageConfig instance
    """
    if config_path is None:
        config_path = Path("config/volatility_arb.yaml")

    if not config_path.exists():
        # Return default configuration
        return VolatilityArbitrageConfig()

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None or "strategy" not in data:
        return VolatilityArbitrageConfig()

    strategy_data = data["strategy"]

    # Parse regime_params if present
    regime_params = None
    if "regime_params" in strategy_data and strategy_data.get("use_regime_detection", False):
        regime_params = {}
        for regime_id, params in strategy_data["regime_params"].items():
            regime_params[int(regime_id)] = RegimeParameters(
                regime_id=int(regime_id),
                entry_threshold_pct=Decimal(str(params.get("entry_threshold_pct", 5.0))),
                exit_threshold_pct=Decimal(str(params.get("exit_threshold_pct", 2.0))),
                position_size_multiplier=Decimal(str(params.get("position_size_multiplier", 1.0))),
                max_vega_multiplier=Decimal(str(params.get("max_vega_multiplier", 1.0))),
            )

    # Build config from YAML data
    config = VolatilityArbitrageConfig(
        # Entry/exit thresholds
        entry_threshold_pct=Decimal(str(strategy_data.get("entry_threshold_pct", 5.0))),
        exit_threshold_pct=Decimal(str(strategy_data.get("exit_threshold_pct", 2.0))),

        # Time constraints
        min_days_to_expiry=strategy_data.get("min_days_to_expiry", 14),
        max_days_to_expiry=strategy_data.get("max_days_to_expiry", 60),

        # Delta hedging
        delta_rebalance_threshold=Decimal(str(strategy_data.get("delta_rebalance_threshold", 0.10))),
        delta_target=Decimal(str(strategy_data.get("delta_target", 0.0))),

        # Position sizing
        position_size_pct=Decimal(str(strategy_data.get("position_size_pct", 5.0))),
        max_vega_exposure=Decimal(str(strategy_data.get("max_vega_exposure", 1000))),
        max_positions=strategy_data.get("max_positions", 5),

        # Volatility forecasting
        vol_lookback_period=strategy_data.get("vol_lookback_period", 30),
        vol_forecast_method=strategy_data.get("vol_forecast_method", "garch"),

        # Risk management
        max_loss_pct=Decimal(str(strategy_data.get("max_loss_pct", 50.0))),

        # Regime detection
        use_regime_detection=strategy_data.get("use_regime_detection", False),
        regime_params=regime_params,
        regime_lookback_period=strategy_data.get("regime_lookback_period", 60),
        exit_on_regime_transition=strategy_data.get("exit_on_regime_transition", False),

        # QV Strategy
        use_qv_strategy=strategy_data.get("use_qv_strategy", False),

        # QV Feature Windows
        rv_window=strategy_data.get("rv_window", 20),
        feature_window=strategy_data.get("feature_window", 60),
        regime_window=strategy_data.get("regime_window", 252),

        # QV Signal Thresholds
        pc_ratio_threshold=Decimal(str(strategy_data.get("pc_ratio_threshold", 1.0))),
        skew_threshold=Decimal(str(strategy_data.get("skew_threshold", 0.05))),
        premium_threshold=Decimal(str(strategy_data.get("premium_threshold", 0.10))),
        term_structure_threshold=Decimal(str(strategy_data.get("term_structure_threshold", 0.0))),
        volume_spike_threshold=Decimal(str(strategy_data.get("volume_spike_threshold", 1.5))),
        sentiment_threshold=Decimal(str(strategy_data.get("sentiment_threshold", -0.05))),

        # QV Consensus Scoring
        consensus_threshold=Decimal(str(strategy_data.get("consensus_threshold", 0.2))),

        # QV Signal Weights
        weight_pc_ratio=Decimal(str(strategy_data.get("weight_pc_ratio", 0.20))),
        weight_iv_skew=Decimal(str(strategy_data.get("weight_iv_skew", 0.20))),
        weight_iv_premium=Decimal(str(strategy_data.get("weight_iv_premium", 0.15))),
        weight_term_structure=Decimal(str(strategy_data.get("weight_term_structure", 0.15))),
        weight_volume_spike=Decimal(str(strategy_data.get("weight_volume_spike", 0.15))),
        weight_near_term_sentiment=Decimal(str(strategy_data.get("weight_near_term_sentiment", 0.15))),

        # QV Regime Scalars
        regime_crisis_scalar=Decimal(str(strategy_data.get("regime_crisis_scalar", 0.5))),
        regime_elevated_scalar=Decimal(str(strategy_data.get("regime_elevated_scalar", 0.75))),
        regime_normal_scalar=Decimal(str(strategy_data.get("regime_normal_scalar", 1.0))),
        regime_low_scalar=Decimal(str(strategy_data.get("regime_low_scalar", 1.2))),
        regime_extreme_low_scalar=Decimal(str(strategy_data.get("regime_extreme_low_scalar", 1.5))),

        # Bullish Base Exposure
        base_long_bias=Decimal(str(strategy_data.get("base_long_bias", 0.8))),
        signal_adjustment_factor=Decimal(str(strategy_data.get("signal_adjustment_factor", 0.7))),

        # Tiered Position Sizing
        use_tiered_sizing=strategy_data.get("use_tiered_sizing", True),
        min_consensus_threshold=Decimal(str(strategy_data.get("min_consensus_threshold", 0.15))),
        position_scaling_method=strategy_data.get("position_scaling_method", "quadratic"),
        min_holding_days=strategy_data.get("min_holding_days", 5),

        # Leverage Configuration (Phase 2)
        use_leverage=strategy_data.get("use_leverage", False),
        short_vol_leverage=Decimal(str(strategy_data.get("short_vol_leverage", 1.3))),
        long_vol_leverage=Decimal(str(strategy_data.get("long_vol_leverage", 2.0))),
        max_leveraged_notional_pct=Decimal(str(strategy_data.get("max_leveraged_notional_pct", 0.80))),
        leverage_drawdown_reduction=strategy_data.get("leverage_drawdown_reduction", True),
        leverage_dd_threshold=Decimal(str(strategy_data.get("leverage_dd_threshold", 0.10))),

        # Bayesian LSTM Volatility Forecasting (Phase 2)
        bayesian_lstm_hidden_size=strategy_data.get("bayesian_lstm_hidden_size", 64),
        bayesian_lstm_dropout_p=strategy_data.get("bayesian_lstm_dropout_p", 0.2),
        bayesian_lstm_sequence_length=strategy_data.get("bayesian_lstm_sequence_length", 20),
        bayesian_lstm_n_mc_samples=strategy_data.get("bayesian_lstm_n_mc_samples", 50),

        # Uncertainty-Adjusted Position Sizing (Phase 2)
        use_uncertainty_sizing=strategy_data.get("use_uncertainty_sizing", False),
        uncertainty_penalty=strategy_data.get("uncertainty_penalty", 2.0),
        uncertainty_min_position_pct=strategy_data.get("uncertainty_min_position_pct", 0.01),
        uncertainty_max_position_pct=strategy_data.get("uncertainty_max_position_pct", 0.15),
    )

    return config
