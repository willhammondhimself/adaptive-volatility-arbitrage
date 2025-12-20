"""
Backtest execution service.

Wraps the integrated backtest logic for API exposure.
"""

import sys
import time
import random
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

# Add src and scripts paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from run_integrated_backtest import (
    load_json_options_data,
    prepare_market_data,
)

from volatility_arbitrage.backtest.multi_asset_engine import MultiAssetBacktestEngine
from volatility_arbitrage.core.config import BacktestConfig
from volatility_arbitrage.strategy.volatility_arbitrage import (
    VolatilityArbitrageStrategy,
    VolatilityArbitrageConfig,
)

from backend.schemas.backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestMetrics,
    EquityPoint,
    Phase2Status,
)


class BacktestService:
    """Service for running backtests."""

    def _generate_demo_response(self, request: BacktestRequest) -> BacktestResponse:
        """Generate mock backtest data for UI testing."""
        start_time = time.time()

        # Generate realistic-looking equity curve
        num_days = request.max_days or 100
        initial = request.initial_capital
        equity = initial
        peak = initial
        max_dd = 0.0

        start_date = datetime(2019, 1, 2)
        equity_curve = []

        # Seed based on config so different parameters produce different results
        seed_value = hash((
            request.max_days,
            request.initial_capital,
            request.entry_threshold_pct,
            request.exit_threshold_pct,
            request.position_size_pct,
            request.max_positions,
            request.use_bayesian_lstm,
            request.use_impact_model,
            request.use_uncertainty_sizing,
            request.use_leverage,
        )) % (2**31)
        random.seed(seed_value)

        # Buy-and-hold benchmark (SPY-like: ~10% annual, ~15% vol)
        buy_hold = initial
        random.seed(12345)  # Fixed seed for consistent benchmark
        buy_hold_returns = [random.gauss(0.0004, 0.012) for _ in range(num_days)]
        random.seed(seed_value)  # Reset to strategy seed

        for i in range(num_days):
            date = start_date + timedelta(days=i)
            # Random walk with slight upward drift
            daily_return = random.gauss(0.0003, 0.015)
            equity *= (1 + daily_return)
            buy_hold *= (1 + buy_hold_returns[i])
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)

            equity_curve.append(
                EquityPoint(
                    date=date.strftime("%Y-%m-%d"),
                    equity=round(equity, 2),
                    drawdown=round(drawdown, 4),
                    buy_hold_equity=round(buy_hold, 2),
                )
            )

        total_return = (equity - initial) / initial
        # Approximate Sharpe based on return
        sharpe = total_return * 2.5 / (max_dd + 0.01) if max_dd > 0 else 1.5

        metrics = BacktestMetrics(
            total_return=round(total_return, 4),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd, 4),
            total_trades=random.randint(15, 35),
            win_rate=round(random.uniform(0.45, 0.65), 2),
        )

        phase2_status = Phase2Status(
            bayesian_lstm_active=request.use_bayesian_lstm,
            impact_model_active=request.use_impact_model,
            uncertainty_sizer_active=request.use_uncertainty_sizing,
            leverage_active=request.use_leverage,
        )

        computation_time_ms = (time.time() - start_time) * 1000

        return BacktestResponse(
            metrics=metrics,
            equity_curve=equity_curve,
            phase2_status=phase2_status,
            computation_time_ms=round(computation_time_ms, 2),
            data_range={
                "start": start_date.strftime("%Y-%m-%d"),
                "end": (start_date + timedelta(days=num_days - 1)).strftime("%Y-%m-%d"),
            },
        )

    def run(self, request: BacktestRequest) -> BacktestResponse:
        """
        Execute a backtest with the specified parameters.

        Args:
            request: Backtest configuration

        Returns:
            Backtest results with metrics and equity curve
        """
        # Return demo data instantly if demo_mode is enabled
        if request.demo_mode:
            return self._generate_demo_response(request)

        start_time = time.time()

        # Load options data
        options_df = load_json_options_data(request.data_dir)
        market_df = prepare_market_data(options_df)

        if request.max_days:
            market_df = market_df.head(request.max_days)

        # Create BacktestConfig
        backtest_config = BacktestConfig(
            initial_capital=Decimal(str(request.initial_capital)),
            commission_rate=Decimal("0.001"),
            slippage=Decimal("0.001"),
            option_spread=Decimal("0.05"),
            option_commission_per_contract=Decimal("0.65"),
            daily_hedge_cost=Decimal("0.0002"),
            margin_rate=Decimal("0.05"),
            position_size_pct=Decimal(str(request.position_size_pct / 100)),
            max_positions=request.max_positions,
            risk_free_rate=Decimal("0.05"),
            use_impact_model=request.use_impact_model,
            impact_half_spread_bps=Decimal("5.0"),
            impact_coefficient=Decimal("0.1"),
        )

        # Create VolatilityArbitrageConfig
        strategy_config = VolatilityArbitrageConfig(
            entry_threshold_pct=Decimal(str(request.entry_threshold_pct)),
            exit_threshold_pct=Decimal(str(request.exit_threshold_pct)),
            min_days_to_expiry=14,
            max_days_to_expiry=60,
            position_size_pct=Decimal(str(request.position_size_pct)),
            max_vega_exposure=Decimal("1000"),
            max_positions=request.max_positions,
            use_qv_strategy=True,
            consensus_threshold=Decimal("0.15"),
            base_long_bias=Decimal("0.0"),
            use_tiered_sizing=False,
            vol_forecast_method="bayesian_lstm" if request.use_bayesian_lstm else "garch",
            bayesian_lstm_hidden_size=64,
            bayesian_lstm_dropout_p=0.2,
            bayesian_lstm_sequence_length=20,
            bayesian_lstm_n_mc_samples=50,
            use_uncertainty_sizing=request.use_uncertainty_sizing,
            uncertainty_penalty=2.0,
            uncertainty_min_position_pct=0.01,
            uncertainty_max_position_pct=0.15,
            use_leverage=request.use_leverage,
            short_vol_leverage=Decimal("1.15"),
            long_vol_leverage=Decimal("1.5"),
        )

        # Create strategy and engine
        strategy = VolatilityArbitrageStrategy(config=strategy_config)
        engine = MultiAssetBacktestEngine(
            config=backtest_config,
            strategy=strategy,
            option_commission_per_contract=Decimal("0.65"),
            option_slippage_pct=Decimal("0.025"),
            margin_requirement_pct=Decimal("0.25"),
        )

        # Run backtest
        start_date = market_df["date"].min()
        end_date = market_df["date"].max()

        strategy.on_backtest_start(start_date, end_date)

        for _, row in market_df.iterrows():
            timestamp = row["date"]
            day_data = market_df[market_df["date"] == timestamp].copy()
            engine._process_day(timestamp, day_data)

        strategy.on_backtest_end()

        # Get results
        results = engine.get_results()

        computation_time_ms = (time.time() - start_time) * 1000

        # Build equity curve
        equity_curve = []
        if results.get("equity_curve"):
            peak_equity = request.initial_capital
            for point in results["equity_curve"]:
                equity = point["equity"]
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

                equity_curve.append(
                    EquityPoint(
                        date=point["date"].isoformat() if hasattr(point["date"], "isoformat") else str(point["date"]),
                        equity=float(equity),
                        drawdown=float(drawdown),
                    )
                )

        # Build metrics
        metrics = BacktestMetrics(
            total_return=results.get("total_return", 0.0),
            sharpe_ratio=results.get("sharpe_ratio", 0.0),
            max_drawdown=results.get("max_drawdown", 0.0),
            total_trades=results.get("total_trades", 0),
            win_rate=None,  # Not computed in current engine
        )

        # Phase 2 status
        phase2_status = Phase2Status(
            bayesian_lstm_active=strategy_config.vol_forecast_method == "bayesian_lstm",
            impact_model_active=engine.cost_model is not None,
            uncertainty_sizer_active=strategy.uncertainty_sizer is not None,
            leverage_active=strategy_config.use_leverage,
        )

        return BacktestResponse(
            metrics=metrics,
            equity_curve=equity_curve,
            phase2_status=phase2_status,
            computation_time_ms=computation_time_ms,
            data_range={
                "start": start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date),
                "end": end_date.isoformat() if hasattr(end_date, "isoformat") else str(end_date),
            },
        )
