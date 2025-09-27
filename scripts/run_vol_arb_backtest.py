#!/usr/bin/env python
"""
Volatility Arbitrage Backtest - Proof of Concept

Demonstrates complete volatility arbitrage strategy on SPY.

This script:
1. Fetches historical data and option chains
2. Runs volatility arbitrage strategy with delta hedging
3. Generates performance metrics and visualizations
4. Saves results to JSON and charts

Usage:
    python scripts/run_vol_arb_backtest.py

Outputs:
    - results/backtest_results.json
    - results/equity_curve.png
    - results/greeks_evolution.png
    - results/summary_table.png
"""

import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from volatility_arbitrage.backtest import (
    MultiAssetBacktestEngine,
    calculate_comprehensive_metrics,
)
from volatility_arbitrage.core.config import load_config
from volatility_arbitrage.data import YahooFinanceFetcher
from volatility_arbitrage.strategy import (
    VolatilityArbitrageStrategy,
    VolatilityArbitrageConfig,
)
from volatility_arbitrage.utils import (
    setup_logging,
    plot_equity_curve,
    plot_greeks_evolution,
    create_summary_table,
)


def main():
    """Run complete volatility arbitrage backtest."""

    print("\n" + "="*70)
    print("  VOLATILITY ARBITRAGE BACKTEST - PROOF OF CONCEPT")
    print("="*70 + "\n")

    # 1. Setup logging
    print("[1/8] Setting up logging...")
    from volatility_arbitrage.core.config import LoggingConfig
    setup_logging(LoggingConfig(level="INFO", console_output=True, format="text"))

    # 2. Load configuration
    print("[2/8] Loading configuration...")
    try:
        config = load_config(Path("config/volatility_arb.yaml"))
        print(f"   ✓ Loaded config: {config.backtest.initial_capital} initial capital")
    except FileNotFoundError:
        print("   ! Config file not found, using defaults")
        from volatility_arbitrage.core.config import Config
        config = Config()

    # 3. Initialize strategy
    print("[3/8] Initializing volatility arbitrage strategy...")
    strategy_config = VolatilityArbitrageConfig(
        entry_threshold_pct=Decimal("5.0"),
        exit_threshold_pct=Decimal("2.0"),
        min_days_to_expiry=14,
        max_days_to_expiry=60,
        delta_rebalance_threshold=Decimal("0.10"),
        vol_lookback_period=30,
        vol_forecast_method="garch",
    )

    strategy = VolatilityArbitrageStrategy(strategy_config)
    print(f"   ✓ Strategy configured: {strategy_config.vol_forecast_method.upper()} forecasting")

    # 4. Initialize backtest engine
    print("[4/8] Initializing multi-asset backtest engine...")
    engine = MultiAssetBacktestEngine(
        config=config.backtest,
        strategy=strategy,
        option_commission_per_contract=Decimal("0.65"),
        option_slippage_pct=Decimal("0.01"),
    )
    print(f"   ✓ Engine initialized with ${config.backtest.initial_capital} capital")

    # 5. Fetch historical data
    print("[5/8] Fetching SPY historical data...")
    fetcher = YahooFinanceFetcher(cache=True)

    try:
        # Use a shorter period for faster testing
        # For production: use full year or more
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 30)  # 6 months for demo

        market_data = fetcher.fetch_historical_data(
            symbol="SPY",
            start_date=start_date,
            end_date=end_date,
        )

        print(f"   ✓ Fetched {len(market_data)} days of data")
        print(f"   ✓ Period: {start_date.date()} to {end_date.date()}")

    except Exception as e:
        print(f"   ✗ Error fetching data: {e}")
        print("   ! Falling back to synthetic data for demonstration")

        # Create synthetic data for demo
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        prices = 100 * (1 + np.random.normal(0.0005, 0.01, len(dates))).cumprod()

        market_data = pd.DataFrame({
            "timestamp": dates,
            "symbol": "SPY",
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(50000000, 150000000, len(dates)),
        })

        print(f"   ✓ Created {len(market_data)} days of synthetic data")

    # 6. Run backtest
    print("[6/8] Running backtest...")
    print("   (This may take a minute...)")

    try:
        result = engine.run(market_data)

        print(f"   ✓ Backtest completed")
        print(f"   ✓ Executed {len(result.trades)} trades")

    except Exception as e:
        print(f"   ✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. Calculate metrics
    print("[7/8] Calculating performance metrics...")
    try:
        metrics = calculate_comprehensive_metrics(
            equity_curve=result.equity_curve,
            initial_capital=result.initial_capital,
            final_capital=result.final_capital,
            num_trades=len(result.trades),
        )

        print(f"   ✓ Calculated {len(metrics.to_dict())} metrics")

    except Exception as e:
        print(f"   ✗ Metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 8. Save results and generate visualizations
    print("[8/8] Saving results and generating charts...")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save JSON results
    results_data = {
        "backtest_period": {
            "start": result.start_date.isoformat(),
            "end": result.end_date.isoformat(),
            "days": len(result.equity_curve),
        },
        "capital": {
            "initial": float(result.initial_capital),
            "final": float(result.final_capital),
        },
        "performance": {
            "total_return_pct": float(metrics.total_return_pct),
            "annualized_return": float(metrics.annualized_return),
            "sharpe_ratio": float(metrics.sharpe_ratio),
            "sortino_ratio": float(metrics.sortino_ratio),
            "calmar_ratio": float(metrics.calmar_ratio),
            "max_drawdown_pct": float(metrics.max_drawdown_pct),
            "volatility": float(metrics.volatility),
        },
        "trading": {
            "num_trades": metrics.num_trades,
            "win_rate": float(metrics.win_rate),
            "profit_factor": float(metrics.profit_factor),
        },
        "full_metrics": metrics.to_dict(),
    }

    with open(results_dir / "backtest_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"   ✓ Saved results to {results_dir}/backtest_results.json")

    # Generate equity curve chart
    try:
        plot_equity_curve(
            result.equity_curve,
            results_dir / "equity_curve.png",
            title="Volatility Arbitrage Strategy - SPY",
        )
        print(f"   ✓ Saved equity curve to {results_dir}/equity_curve.png")
    except Exception as e:
        print(f"   ! Could not generate equity curve: {e}")

    # Generate Greeks evolution chart
    try:
        if "portfolio_delta" in result.equity_curve.columns:
            plot_greeks_evolution(
                result.equity_curve,
                results_dir / "greeks_evolution.png",
            )
            print(f"   ✓ Saved Greeks evolution to {results_dir}/greeks_evolution.png")
    except Exception as e:
        print(f"   ! Could not generate Greeks chart: {e}")

    # Generate summary table
    try:
        create_summary_table(
            metrics.to_dict(),
            results_dir / "summary_table.png",
        )
        print(f"   ✓ Saved summary table to {results_dir}/summary_table.png")
    except Exception as e:
        print(f"   ! Could not generate summary table: {e}")

    # Print results summary
    print("\n" + "="*70)
    print("  BACKTEST RESULTS SUMMARY")
    print("="*70)
    print(f"\n  Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"  Trading Days: {len(result.equity_curve)}")
    print(f"\n  RETURNS")
    print(f"    Initial Capital:     ${result.initial_capital:>12,.2f}")
    print(f"    Final Capital:       ${result.final_capital:>12,.2f}")
    print(f"    Total Return:        {metrics.total_return_pct:>12.2f}%")
    print(f"    Annualized Return:   {metrics.annualized_return:>12.2f}%")
    print(f"\n  RISK METRICS")
    print(f"    Volatility (ann.):   {metrics.volatility:>12.2f}%")
    print(f"    Max Drawdown:        {metrics.max_drawdown_pct:>12.2f}%")
    print(f"    Sharpe Ratio:        {metrics.sharpe_ratio:>12.2f}")
    print(f"    Sortino Ratio:       {metrics.sortino_ratio:>12.2f}")
    print(f"    Calmar Ratio:        {metrics.calmar_ratio:>12.2f}")
    print(f"\n  TRADING")
    print(f"    Number of Trades:    {metrics.num_trades:>12}")
    print(f"    Win Rate:            {metrics.win_rate:>12.1f}%")
    print(f"    Profit Factor:       {metrics.profit_factor:>12.2f}")
    print("\n" + "="*70)
    print(f"\n  Results saved to: {results_dir.absolute()}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
