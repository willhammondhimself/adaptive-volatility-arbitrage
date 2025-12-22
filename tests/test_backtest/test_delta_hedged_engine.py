"""Tests for DeltaHedgedBacktest engine."""

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from volatility_arbitrage.backtest.delta_hedged.engine import (
    BacktestMetrics,
    DeltaHedgedBacktest,
    generate_gbm_with_stochastic_vol,
    generate_vol_spike_scenario,
)
from volatility_arbitrage.backtest.delta_hedged.types import (
    HedgeConfig,
    RebalanceFrequency,
)
from volatility_arbitrage.models.black_scholes import OptionType


@pytest.mark.unit
class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_gbm_returns_dataframe(self) -> None:
        """GBM generator returns DataFrame with required columns."""
        df = generate_gbm_with_stochastic_vol(days=10, seed=42)

        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "spot" in df.columns
        assert "iv" in df.columns
        assert len(df) == 10 * 8  # 8 hours per day

    def test_generate_gbm_spot_positive(self) -> None:
        """GBM spot prices are always positive."""
        df = generate_gbm_with_stochastic_vol(days=30, seed=42)
        assert (df["spot"] > 0).all()

    def test_generate_gbm_iv_bounded(self) -> None:
        """GBM IV stays within reasonable bounds."""
        df = generate_gbm_with_stochastic_vol(days=30, seed=42)
        assert (df["iv"] >= 0.05).all()
        assert (df["iv"] < 1.0).all()

    def test_generate_gbm_reproducible(self) -> None:
        """Same seed produces same results."""
        df1 = generate_gbm_with_stochastic_vol(days=10, seed=42)
        df2 = generate_gbm_with_stochastic_vol(days=10, seed=42)

        assert np.allclose(df1["spot"].values, df2["spot"].values)
        assert np.allclose(df1["iv"].values, df2["iv"].values)

    def test_generate_vol_spike(self) -> None:
        """Vol spike scenario has expected IV jump."""
        df = generate_vol_spike_scenario(
            days=20,
            initial_iv=0.15,
            spike_day=10,
            spike_magnitude=0.15,
            seed=42,
        )

        # IV should spike around day 10
        iv_before = df[df["timestamp"] < datetime(2024, 1, 12)]["iv"].mean()
        iv_after = df[df["timestamp"] >= datetime(2024, 1, 12)]["iv"].iloc[0]

        assert iv_after > iv_before + 0.10  # At least 10% higher


@pytest.mark.unit
class TestDeltaHedgedBacktest:
    """Tests for backtest engine."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Generate sample data for testing."""
        return generate_gbm_with_stochastic_vol(days=20, seed=42)

    def test_run_returns_metrics(self, sample_data: pd.DataFrame) -> None:
        """Run returns BacktestMetrics."""
        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.DAILY)
        backtest = DeltaHedgedBacktest(config, option_position=10)

        metrics = backtest.run(
            data=sample_data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )

        assert isinstance(metrics, BacktestMetrics)
        assert metrics.num_periods > 0

    def test_attribution_df_has_required_columns(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Attribution DataFrame has all required columns."""
        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.DAILY)
        backtest = DeltaHedgedBacktest(config, option_position=10)

        backtest.run(
            data=sample_data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )

        df = backtest.get_attribution_df()

        required_cols = [
            "timestamp",
            "total_pnl",
            "delta_pnl",
            "gamma_pnl",
            "vega_pnl",
            "theta_pnl",
            "transaction_costs",
            "cum_total_pnl",
            "cum_delta_pnl",
            "cum_gamma_pnl",
            "cum_vega_pnl",
            "cum_theta_pnl",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_cumulative_columns_accumulate(self, sample_data: pd.DataFrame) -> None:
        """Cumulative columns are running totals."""
        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.DAILY)
        backtest = DeltaHedgedBacktest(config, option_position=10)

        backtest.run(
            data=sample_data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 2, 16),
            option_type=OptionType.CALL,
        )

        df = backtest.get_attribution_df()

        # Cumulative should be running sum
        assert np.allclose(
            df["cum_total_pnl"].values, df["total_pnl"].cumsum().values
        )


@pytest.mark.integration
class TestHedgeEffectiveness:
    """Integration tests for hedge effectiveness."""

    def test_delta_pnl_variance_low_with_frequent_hedging(self) -> None:
        """Frequent hedging keeps delta P&L variance low."""
        data = generate_gbm_with_stochastic_vol(days=30, seed=42)

        # Continuous hedging
        config = HedgeConfig(
            rebalance_frequency=RebalanceFrequency.CONTINUOUS,
            delta_threshold=Decimal("0.01"),
        )
        backtest = DeltaHedgedBacktest(config, option_position=10)

        metrics = backtest.run(
            data=data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 3, 1),
            option_type=OptionType.CALL,
        )

        # Delta P&L variance should be small relative to total
        assert metrics.delta_pnl_variance_ratio < 0.10

    def test_hedge_effectiveness_above_threshold(self) -> None:
        """Hedge effectiveness should be high with frequent hedging."""
        data = generate_gbm_with_stochastic_vol(days=30, seed=42)

        config = HedgeConfig(
            rebalance_frequency=RebalanceFrequency.HOURLY,
            delta_threshold=Decimal("0.05"),
        )
        backtest = DeltaHedgedBacktest(config, option_position=10)

        metrics = backtest.run(
            data=data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 3, 1),
            option_type=OptionType.CALL,
        )

        # Hedge effectiveness = 1 - delta_variance_ratio
        # Should be > 90% with frequent hedging
        assert metrics.hedge_effectiveness > 0.90

    def test_vega_pnl_positive_in_vol_spike(self) -> None:
        """Long vega profits from vol spike."""
        data = generate_vol_spike_scenario(
            days=20,
            initial_iv=0.15,
            spike_day=10,
            spike_magnitude=0.15,
            seed=42,
        )

        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.DAILY)
        backtest = DeltaHedgedBacktest(config, option_position=10)

        backtest.run(
            data=data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 3, 1),
            option_type=OptionType.CALL,
        )

        df = backtest.get_attribution_df()

        # Cumulative vega P&L should be positive (profited from vol spike)
        assert df["cum_vega_pnl"].iloc[-1] > 0

    def test_gamma_pnl_positive_for_large_moves(self) -> None:
        """Long gamma profits from large spot moves."""
        # Generate data with high volatility
        data = generate_gbm_with_stochastic_vol(
            days=30,
            initial_iv=0.30,
            vol_of_vol=0.8,
            seed=42,
        )

        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.DAILY)
        backtest = DeltaHedgedBacktest(config, option_position=10)

        backtest.run(
            data=data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 3, 1),
            option_type=OptionType.CALL,
        )

        df = backtest.get_attribution_df()

        # Cumulative gamma P&L should be positive (profited from moves)
        assert df["cum_gamma_pnl"].iloc[-1] > 0

    def test_theta_pnl_negative_for_long_options(self) -> None:
        """Long options have negative theta P&L."""
        data = generate_gbm_with_stochastic_vol(days=30, seed=42)

        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.DAILY)
        backtest = DeltaHedgedBacktest(config, option_position=10)

        backtest.run(
            data=data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 3, 1),
            option_type=OptionType.CALL,
        )

        df = backtest.get_attribution_df()

        # Cumulative theta P&L should be negative (time decay)
        assert df["cum_theta_pnl"].iloc[-1] < 0


@pytest.mark.unit
class TestMetricsCalculation:
    """Tests for metrics calculation."""

    def test_sharpe_calculation(self) -> None:
        """Sharpe ratio is calculated correctly."""
        data = generate_gbm_with_stochastic_vol(days=60, seed=42)

        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.DAILY)
        backtest = DeltaHedgedBacktest(config, option_position=10)

        metrics = backtest.run(
            data=data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 4, 1),
            option_type=OptionType.CALL,
        )

        # Sharpe should be a reasonable number
        assert -5 < metrics.total_sharpe < 5
        assert -5 < metrics.vega_gamma_sharpe < 5

    def test_transaction_costs_accumulate(self) -> None:
        """Transaction costs accumulate with rebalancing."""
        data = generate_gbm_with_stochastic_vol(days=30, seed=42)

        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.CONTINUOUS)
        backtest = DeltaHedgedBacktest(config, option_position=10)

        metrics = backtest.run(
            data=data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 3, 1),
            option_type=OptionType.CALL,
        )

        # Many rebalances should have significant costs
        assert metrics.rebalance_count > 100
        assert metrics.total_transaction_costs > 0

    def test_metrics_to_dict(self) -> None:
        """Metrics can be converted to dict."""
        data = generate_gbm_with_stochastic_vol(days=10, seed=42)

        config = HedgeConfig(rebalance_frequency=RebalanceFrequency.DAILY)
        backtest = DeltaHedgedBacktest(config, option_position=10)

        metrics = backtest.run(
            data=data,
            strike=Decimal("450.00"),
            expiry=datetime(2024, 3, 1),
            option_type=OptionType.CALL,
        )

        d = metrics.to_dict()

        assert "total_pnl" in d
        assert "total_sharpe" in d
        assert "vega_gamma_sharpe" in d
        assert "hedge_effectiveness" in d
