#!/usr/bin/env python3
"""
Monte Carlo Simulation for QV Strategy.

Bootstraps trade returns from a backtest to estimate confidence intervals
on Sharpe, returns, and drawdown.

Usage:
    PYTHONPATH=./src:. python scripts/run_monte_carlo.py
    PYTHONPATH=./src:. python scripts/run_monte_carlo.py --simulations 10000
"""

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

from volatility_arbitrage.core.config import load_strategy_config
from volatility_arbitrage.analysis.monte_carlo import (
    extract_trade_returns,
    bootstrap_resample,
    block_bootstrap_resample,
    print_monte_carlo_report,
    plot_monte_carlo_distributions,
)

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load_json_options_data, run_qv_backtest


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation')
    parser.add_argument(
        '--config',
        type=str,
        default='config/volatility_arb_aggressive.yaml',
        help='Config file to use',
    )
    parser.add_argument(
        '--data',
        type=str,
        default='src/volatility_arbitrage/data/SPY_Options_2019_24',
        help='Options data directory',
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital',
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulations',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show distribution plots',
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Path to save plot (e.g., reports/monte_carlo.png)',
    )
    parser.add_argument(
        '--block-bootstrap',
        action='store_true',
        help='Use block bootstrap to preserve serial correlation',
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=3,
        help='Block size for block bootstrap (default 3)',
    )
    parser.add_argument(
        '--winsorize',
        type=float,
        default=None,
        help='Winsorize returns at this percentile (e.g., 95 to cap at 5th/95th)',
    )
    args = parser.parse_args()

    print("="*60)
    print("MONTE CARLO SIMULATION")
    print("="*60)
    print(f"\nConfig: {args.config}")
    print(f"Simulations: {args.simulations:,}")

    # Load data
    print(f"\nLoading data from {args.data}...")
    options_df = load_json_options_data(args.data)
    options_df['date'] = pd.to_datetime(options_df['date'])
    print(f"  Records: {len(options_df):,}")

    # Load config
    config = load_strategy_config(Path(args.config))

    # Run backtest to get trades
    print("\nRunning backtest to collect trades...")
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        results = run_qv_backtest(
            options_df,
            config,
            args.capital,
            config_path=str(args.config)
        )
    finally:
        sys.stdout = old_stdout

    # Extract trade returns
    trades = results.get('trade_log', [])
    trade_returns = extract_trade_returns(trades)

    print(f"  Total trades: {len(trades)}")
    print(f"  Exit trades: {len(trade_returns)}")
    print(f"  Avg return/trade: {trade_returns.mean()*100:.2f}%")
    print(f"  Std dev/trade: {trade_returns.std()*100:.2f}%")

    # Actual metrics for comparison
    actual_metrics = {
        'total_return': results['total_return'],
        'sharpe': results['sharpe'],
        'max_drawdown': results['max_drawdown'],
    }

    print(f"\nActual backtest results:")
    print(f"  Total Return: {actual_metrics['total_return']:.1f}%")
    print(f"  Sharpe Ratio: {actual_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {actual_metrics['max_drawdown']:.1f}%")

    # Run Monte Carlo
    method = "block bootstrap" if args.block_bootstrap else "standard bootstrap"
    winsorize_str = f" (winsorized at {args.winsorize}%)" if args.winsorize else ""
    print(f"\nRunning {args.simulations:,} Monte Carlo simulations ({method}{winsorize_str})...")

    if args.block_bootstrap:
        mc_result = block_bootstrap_resample(
            trade_returns,
            n_simulations=args.simulations,
            block_size=args.block_size,
            initial_capital=args.capital,
            random_seed=args.seed,
            winsorize_pct=args.winsorize,
        )
    else:
        mc_result = bootstrap_resample(
            trade_returns,
            n_simulations=args.simulations,
            initial_capital=args.capital,
            random_seed=args.seed,
            winsorize_pct=args.winsorize,
        )

    # Print report
    print_monte_carlo_report(mc_result, actual_metrics)

    # Plot if requested
    if args.plot or args.save_plot:
        plot_monte_carlo_distributions(
            mc_result,
            actual_metrics,
            save_path=args.save_plot,
        )


if __name__ == "__main__":
    main()
