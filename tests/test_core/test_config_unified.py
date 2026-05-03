"""Regression test: VolatilityArbitrageConfig is a single class shared by
strategy and engine (F11)."""

from decimal import Decimal

from volatility_arbitrage.backtest.multi_asset_engine import MultiAssetBacktestEngine
from volatility_arbitrage.core.config import (
    BacktestConfig,
    VolatilityArbitrageConfig as CoreConfig,
)
from volatility_arbitrage.strategy import (
    VolatilityArbitrageConfig as StrategyConfig,
    VolatilityArbitrageStrategy,
)


def test_strategy_and_core_config_are_same_class():
    """The two import paths must resolve to the same class (no shadowing)."""
    assert StrategyConfig is CoreConfig


def test_config_carries_strategy_and_engine_fields():
    """A single config instance must satisfy both strategy and engine consumers."""
    cfg = StrategyConfig(
        use_real_options_data=False,
        use_signal_smoothing=False,
        max_loss_pct=Decimal("40.0"),
    )

    # Strategy-side fields
    assert cfg.use_signal_smoothing is False
    assert cfg.signal_smoothing_window == 3
    assert cfg.profit_take_levels == [Decimal("0.25"), Decimal("0.50"), Decimal("0.75")]
    assert cfg.adaptive_entry_high_vol == Decimal("7.0")

    # Engine-side fields
    assert cfg.use_real_options_data is False
    assert cfg.options_data_dir.endswith("SPY_Options_2019_24")


def test_strategy_and_engine_accept_same_config_instance():
    """Pass one config to both VolatilityArbitrageStrategy and MultiAssetBacktestEngine."""
    cfg = StrategyConfig(use_real_options_data=False)

    strategy = VolatilityArbitrageStrategy(config=cfg)
    engine = MultiAssetBacktestEngine(
        config=BacktestConfig(),
        strategy=strategy,
        strategy_config=cfg,
    )

    assert strategy.config is cfg
    assert engine.strategy_config is cfg
    assert engine.strategy_config.use_real_options_data is False
