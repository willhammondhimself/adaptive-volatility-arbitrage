#!/usr/bin/env python3
"""
MCPT Validation for QV 6-Signal Strategy.

Validates that strategy performance isn't just data mining noise using
Monte Carlo Permutation Testing.

This script runs:
1. In-sample MCPT: Tests if full-period performance is significant
2. Walk-forward MCPT (test-only): Tests if OOS performance is luck
3. Walk-forward MCPT (full): Tests if entire strategy edge is noise

Usage:
    python scripts/validate_mcpt.py --data path/to/options
    python scripts/validate_mcpt.py --quick  # 100 permutations for testing
    python scripts/validate_mcpt.py --help

Output:
    Table with p-values and PASS/MARGINAL/FAIL status for each test.
    Distribution plots saved to reports/ directory.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from volatility_arbitrage.testing.mcpt import (
    run_insample_mcpt,
    run_walkforward_mcpt,
    MCPTConfig,
    MCPTResult,
)
from volatility_arbitrage.testing.mcpt.walkforward_test import (
    run_dual_walkforward_mcpt,
    WalkForwardMCPTResult,
)
from volatility_arbitrage.testing.mcpt.utils import (
    plot_mcpt_distribution,
    generate_mcpt_report,
)
from volatility_arbitrage.core.config import load_strategy_config


def load_options_data(data_path: str) -> pd.DataFrame:
    """
    Load options data from JSON files or CSV.

    Supports the same format as run_backtest.py.
    """
    import json
    from pathlib import Path

    data_path = Path(data_path)

    if data_path.is_file() and data_path.suffix == '.csv':
        # Load CSV directly
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'])
        return df

    elif data_path.is_dir():
        # Load JSON files from directory
        all_records = []
        json_files = sorted(data_path.glob("*.json"))

        if len(json_files) == 0:
            raise ValueError(f"No JSON files found in {data_path}")

        print(f"Loading {len(json_files)} JSON files from {data_path}...")

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Handle nested list structure
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    for day_records in data:
                        all_records.extend(day_records)
                else:
                    all_records.extend(data)

        print(f"Loaded {len(all_records):,} records")

        df = pd.DataFrame(all_records)

        # Convert types
        df['date'] = pd.to_datetime(df['date'])
        df['expiration'] = pd.to_datetime(df['expiration'])
        for col in ['strike', 'last', 'bid', 'ask', 'mark', 'volume', 'open_interest',
                    'implied_volatility', 'delta', 'gamma', 'theta', 'vega']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    else:
        raise ValueError(f"Data path not found: {data_path}")


def create_strategy_func(config):
    """
    Create a strategy function wrapper for MCPT.

    This wraps run_qv_backtest to match the expected signature.
    """
    # Import here to avoid circular imports
    from scripts.run_backtest import run_qv_backtest

    def strategy_func(options_df: pd.DataFrame) -> dict:
        """Run QV strategy and return results dict."""
        return run_qv_backtest(options_df, config)

    return strategy_func


def print_results_table(
    insample: MCPTResult,
    wf_test_only: WalkForwardMCPTResult,
    wf_full: WalkForwardMCPTResult,
) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("MCPT VALIDATION RESULTS")
    print("=" * 80)
    print(f"{'Test':<35} | {'Real':>10} | {'Perm Mean':>10} | {'p-value':>8} | {'Status':>10}")
    print("-" * 80)

    results = [insample, wf_test_only, wf_full]

    for result in results:
        status = result.status
        print(
            f"{result.test_name:<35} | "
            f"{result.real_metric:>10.3f} | "
            f"{result.mean_permuted:>10.3f} | "
            f"{result.p_value:>8.4f} | "
            f"{status:>10}"
        )

    print("=" * 80)

    # Summary
    all_pass = all(r.p_value < 0.01 for r in results)
    any_pass = any(r.p_value < 0.05 for r in results)

    print("\nINTERPRETATION:")
    if all_pass:
        print("  ALL TESTS PASS (p < 0.01): STRONG evidence of genuine alpha")
        print("  Strategy performance is very unlikely to be due to random chance.")
    elif any_pass:
        print("  SOME TESTS PASS: Moderate evidence of alpha")
        print("  Consider additional validation and risk management.")
    else:
        print("  ALL TESTS FAIL (p >= 0.05): WEAK or NO evidence of alpha")
        print("  Strategy performance could easily be due to data mining or luck.")


def main():
    parser = argparse.ArgumentParser(
        description='MCPT Validation for Volatility Arbitrage Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full validation (1000 permutations, ~1-2 hours)
    python scripts/validate_mcpt.py --data data/SPY_Options_2019_24

    # Quick test (100 permutations, ~10 minutes)
    python scripts/validate_mcpt.py --data data/SPY_Options_2019_24 --quick

    # Custom settings
    python scripts/validate_mcpt.py --data path/to/options --permutations 500 --jobs 4
        """
    )

    parser.add_argument(
        '--data',
        type=str,
        default='src/volatility_arbitrage/data/SPY_Options_2019_24',
        help='Path to options data directory or CSV file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/volatility_arb.yaml',
        help='Path to strategy config file'
    )
    parser.add_argument(
        '--mcpt-config',
        type=str,
        default=None,
        help='Path to MCPT config file (optional)'
    )
    parser.add_argument(
        '--permutations',
        type=int,
        default=None,
        help='Number of permutations (overrides config)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: 100 permutations for testing'
    )
    parser.add_argument(
        '--tiny',
        action='store_true',
        help='Tiny mode: 20 permutations + 1 year data (~5-10 min)'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=-1,
        help='Number of parallel workers (-1 = all CPUs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--objective',
        type=str,
        default='sharpe',
        choices=['sharpe', 'nw_sharpe', 'profit_factor', 'total_return'],
        help='Objective metric to test'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/mcpt_validation',
        help='Output directory for reports and plots'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating distribution plots'
    )
    parser.add_argument(
        '--insample-only',
        action='store_true',
        help='Only run in-sample MCPT (skip walk-forward)'
    )
    parser.add_argument(
        '--walkforward-only',
        action='store_true',
        help='Only run walk-forward MCPT (skip in-sample)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run sequentially (no parallel, for debugging)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MCPT VALIDATION FOR VOLATILITY ARBITRAGE STRATEGY")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Data: {args.data}")
    print(f"Config: {args.config}")
    print(f"Objective: {args.objective}")

    # Load strategy config
    print(f"\nLoading strategy config from {args.config}...")
    strategy_config = load_strategy_config(Path(args.config))

    # Load or create MCPT config
    if args.mcpt_config:
        print(f"Loading MCPT config from {args.mcpt_config}...")
        mcpt_config = MCPTConfig.from_yaml(Path(args.mcpt_config))
    else:
        mcpt_config = MCPTConfig()

    # Override with command line args
    if args.tiny:
        mcpt_config.n_permutations = 20
        mcpt_config.n_permutations_walkforward = 10
        print("Tiny mode: 20 permutations + 1 year data")
    elif args.quick:
        mcpt_config.n_permutations = 100
        mcpt_config.n_permutations_walkforward = 50
        print("Quick mode: 100 permutations")
    elif args.permutations:
        mcpt_config.n_permutations = args.permutations
        mcpt_config.n_permutations_walkforward = max(50, args.permutations // 5)

    mcpt_config.n_jobs = args.jobs
    mcpt_config.random_seed = args.seed

    print(f"\nMCPT Configuration:")
    print(f"  Permutations (in-sample): {mcpt_config.n_permutations}")
    print(f"  Permutations (walk-forward): {mcpt_config.n_permutations_walkforward}")
    print(f"  Workers: {mcpt_config.n_jobs if mcpt_config.n_jobs != -1 else 'all CPUs'}")
    print(f"  Seed: {mcpt_config.random_seed}")
    print(f"  Warmup days: {mcpt_config.warmup_days}")

    # Load options data
    print(f"\nLoading options data from {args.data}...")
    options_df = load_options_data(args.data)
    print(f"Loaded {len(options_df):,} records")

    # Filter to 1 year for tiny mode
    if args.tiny:
        max_date = options_df['date'].max()
        min_date = max_date - pd.Timedelta(days=365)
        options_df = options_df[options_df['date'] >= min_date].copy()
        print(f"Tiny mode: filtered to last year ({len(options_df):,} records)")

    print(f"Date range: {options_df['date'].min()} to {options_df['date'].max()}")
    print(f"Unique dates: {options_df['date'].nunique()}")

    # Create strategy function
    print("\nCreating strategy function wrapper...")
    strategy_func = create_strategy_func(strategy_config)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Run in-sample MCPT
    if not args.walkforward_only:
        print("\n" + "=" * 70)
        print("RUNNING IN-SAMPLE MCPT")
        print("=" * 70)

        insample_result = run_insample_mcpt(
            options_df=options_df,
            strategy_func=strategy_func,
            config=mcpt_config,
            objective=args.objective,
            use_parallel=not args.sequential,
        )
        results.append(insample_result)

        # Save plot
        if not args.no_plots:
            plot_path = output_dir / "insample_distribution.png"
            plot_mcpt_distribution(
                real_metric=insample_result.real_metric,
                permuted_metrics=insample_result.permuted_metrics,
                metric_name=args.objective.replace('_', ' ').title(),
                p_value=insample_result.p_value,
                save_path=plot_path,
                title=f"In-Sample MCPT: {args.objective}",
            )

    # Run walk-forward MCPT
    if not args.insample_only:
        print("\n" + "=" * 70)
        print("RUNNING WALK-FORWARD MCPT (DUAL MODE)")
        print("=" * 70)

        wf_results = run_dual_walkforward_mcpt(
            options_df=options_df,
            strategy_func=strategy_func,
            config=mcpt_config,
            objective=args.objective,
        )

        results.append(wf_results['test_only'])
        results.append(wf_results['full'])

        # Save plots
        if not args.no_plots:
            for mode, wf_result in wf_results.items():
                plot_path = output_dir / f"walkforward_{mode}_distribution.png"
                plot_mcpt_distribution(
                    real_metric=wf_result.real_test_metric,
                    permuted_metrics=wf_result.permuted_test_metrics,
                    metric_name=args.objective.replace('_', ' ').title(),
                    p_value=wf_result.p_value,
                    save_path=plot_path,
                    title=f"Walk-Forward MCPT ({mode}): {args.objective}",
                )

    # Generate report
    if len(results) == 3:
        print_results_table(results[0], results[1], results[2])
    else:
        # Generate text report
        report = generate_mcpt_report(results)
        print(report)

    # Save report
    report_path = output_dir / "mcpt_report.txt"
    report = generate_mcpt_report(results, output_path=report_path)

    print(f"\nOutput saved to: {output_dir}/")
    print("  - mcpt_report.txt")
    if not args.no_plots:
        print("  - insample_distribution.png")
        print("  - walkforward_test_only_distribution.png")
        print("  - walkforward_full_distribution.png")

    # Return exit code based on results
    all_pass = all(r.p_value < 0.05 for r in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
