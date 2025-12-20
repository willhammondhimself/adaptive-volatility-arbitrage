#!/usr/bin/env python3
"""
Ablation Study for Volatility Arbitrage Strategy Enhancements.

Tests each enhancement INDEPENDENTLY to measure its impact on:
- Sharpe ratio
- Total return
- MCPT p-value (statistical significance)

Usage:
    python scripts/run_ablation_study.py
    python scripts/run_ablation_study.py --permutations 50  # Quick mode
    python scripts/run_ablation_study.py --variants E1,E2,E3  # Test specific variants

Output:
    Comparison table and individual results saved to reports/ablation_study/
"""

import argparse
import json
import sys
import yaml
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from volatility_arbitrage.core.config import load_strategy_config
from volatility_arbitrage.testing.mcpt import run_insample_mcpt, MCPTConfig


def load_ablation_config(path: str = "config/ablation_config.yaml") -> dict:
    """Load ablation study configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_options_data(data_path: str) -> pd.DataFrame:
    """Load options data from JSON files."""
    from pathlib import Path
    import json

    data_path = Path(data_path)
    all_records = []
    json_files = sorted(data_path.glob("*.json"))

    print(f"Loading {len(json_files)} JSON files...")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                for day_records in data:
                    all_records.extend(day_records)
            else:
                all_records.extend(data)

    df = pd.DataFrame(all_records)

    # Convert types
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    for col in ['strike', 'last', 'bid', 'ask', 'mark', 'volume', 'open_interest',
                'implied_volatility', 'delta', 'gamma', 'theta', 'vega']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def apply_config_overrides(base_config_path: str, overrides: dict, baseline_overrides: dict = None) -> dict:
    """
    Apply ablation variant overrides to base config.

    First applies baseline_overrides (disabling all enhancements),
    then applies variant-specific overrides.
    """
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    strategy = config.get('strategy', {})

    # First apply baseline (disable all enhancements)
    if baseline_overrides:
        for key, value in baseline_overrides.items():
            strategy[key] = value

    # Then apply variant-specific overrides
    for key, value in overrides.items():
        strategy[key] = value

    config['strategy'] = strategy
    return config


def create_temp_config(config_dict: dict, variant_name: str) -> Path:
    """Create temporary config file for variant."""
    temp_dir = Path("reports/ablation_study/temp_configs")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize variant name for filename
    safe_name = variant_name.replace(" ", "_").replace(":", "_").replace("/", "_")
    temp_path = temp_dir / f"config_{safe_name}.yaml"

    with open(temp_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    return temp_path


def run_single_variant(
    variant_name: str,
    config_overrides: dict,
    options_df: pd.DataFrame,
    base_config_path: str,
    mcpt_permutations: int,
    run_mcpt: bool = True,
    baseline_overrides: dict = None,
) -> dict:
    """
    Run backtest and MCPT for a single variant.

    Returns dict with metrics.
    """
    from scripts.run_backtest import run_qv_backtest

    print(f"\n{'='*70}")
    print(f"VARIANT: {variant_name}")
    print(f"{'='*70}")
    print(f"Overrides: {config_overrides}")

    # Create config with overrides (baseline first, then variant-specific)
    config_dict = apply_config_overrides(base_config_path, config_overrides, baseline_overrides)
    temp_config_path = create_temp_config(config_dict, variant_name)

    # Load config object
    config = load_strategy_config(temp_config_path)

    # Run backtest with the temp config path so enhancements are loaded correctly
    print(f"\nRunning backtest...")
    results = run_qv_backtest(options_df, config, config_path=str(temp_config_path))

    metrics = {
        'variant': variant_name,
        'sharpe': results['sharpe'],
        'nw_sharpe': results['nw_sharpe'],
        'total_return': results['total_return'],
        'max_drawdown': results['max_drawdown'],
        'trades': results['trades'],
    }

    # Calculate additional metrics from equity curve
    equity_curve = results['equity_curve']
    returns = equity_curve['equity'].pct_change().dropna()

    # Win rate (from trades with position returns)
    equity_curve['daily_return'] = returns
    positive_days = (returns > 0).sum()
    metrics['win_rate'] = positive_days / len(returns) * 100 if len(returns) > 0 else 0

    # Run MCPT if requested
    if run_mcpt:
        print(f"\nRunning MCPT with {mcpt_permutations} permutations...")

        mcpt_config = MCPTConfig()
        mcpt_config.n_permutations = mcpt_permutations

        def strategy_func(df):
            return run_qv_backtest(df, config)

        try:
            mcpt_result = run_insample_mcpt(
                options_df=options_df,
                strategy_func=strategy_func,
                config=mcpt_config,
                objective='sharpe',
                use_parallel=True,
            )
            metrics['mcpt_pvalue'] = mcpt_result.p_value
            metrics['mcpt_status'] = mcpt_result.status
        except Exception as e:
            print(f"MCPT failed: {e}")
            metrics['mcpt_pvalue'] = None
            metrics['mcpt_status'] = 'ERROR'
    else:
        metrics['mcpt_pvalue'] = None
        metrics['mcpt_status'] = 'SKIPPED'

    return metrics


def generate_comparison_table(results: List[dict], baseline_metrics: dict) -> str:
    """Generate formatted comparison table."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("ABLATION STUDY RESULTS")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Variant':<30} | {'Sharpe':>8} | {'dSharpe':>8} | {'Return%':>8} | {'p-value':>8} | {'Status':>10}")
    lines.append("-" * 100)

    baseline_sharpe = baseline_metrics['sharpe']

    # Format p-value helper
    def fmt_pvalue(pval):
        if pval is None:
            return "N/A"
        return f"{pval:.4f}"

    # Add baseline first
    baseline_pval = fmt_pvalue(baseline_metrics.get('mcpt_pvalue'))
    lines.append(
        f"{'Baseline (v1.0)':<30} | "
        f"{baseline_metrics['sharpe']:>8.3f} | "
        f"{'-':>8} | "
        f"{baseline_metrics['total_return']:>8.1f} | "
        f"{baseline_pval:>8} | "
        f"{baseline_metrics.get('mcpt_status', 'N/A'):>10}"
    )
    lines.append("-" * 100)

    # Add variants
    for r in results:
        if r['sharpe'] is None:
            lines.append(f"{r['variant']:<30} | {'ERROR':>8} | {'-':>8} | {'-':>8} | {'N/A':>8} | {'ERROR':>10}")
            continue

        delta_sharpe = r['sharpe'] - baseline_sharpe
        delta_str = f"{delta_sharpe:+.3f}"
        pvalue_str = fmt_pvalue(r.get('mcpt_pvalue'))

        lines.append(
            f"{r['variant']:<30} | "
            f"{r['sharpe']:>8.3f} | "
            f"{delta_str:>8} | "
            f"{r['total_return']:>8.1f} | "
            f"{pvalue_str:>8} | "
            f"{r.get('mcpt_status', 'N/A'):>10}"
        )

    lines.append("=" * 100)
    lines.append("")

    # Summary statistics
    lines.append("SUMMARY")
    lines.append("-" * 50)

    # Find best/worst
    valid_results = [r for r in results if r['sharpe'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['sharpe'])
        worst = min(valid_results, key=lambda x: x['sharpe'])

        lines.append(f"Best Enhancement:  {best['variant']:<25} (Sharpe: {best['sharpe']:.3f}, dSharpe: {best['sharpe'] - baseline_sharpe:+.3f})")
        lines.append(f"Worst Enhancement: {worst['variant']:<25} (Sharpe: {worst['sharpe']:.3f}, dSharpe: {worst['sharpe'] - baseline_sharpe:+.3f})")

        # Count improvements
        improvements = [r for r in valid_results if r['sharpe'] > baseline_sharpe]
        lines.append(f"Enhancements that improved Sharpe: {len(improvements)} / {len(valid_results)}")

        # Statistical significance
        significant = [r for r in valid_results if r.get('mcpt_pvalue') is not None and r['mcpt_pvalue'] < 0.05]
        lines.append(f"Statistically significant (p<0.05): {len(significant)} / {len(valid_results)}")

    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  dSharpe > 0: Enhancement IMPROVES performance")
    lines.append("  dSharpe < 0: Enhancement HURTS performance")
    lines.append("  p-value < 0.05: Strong evidence of alpha")
    lines.append("  p-value < 0.10: Marginal evidence")
    lines.append("  p-value >= 0.10: No evidence (likely noise)")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study for strategy enhancements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--data',
        type=str,
        default='src/volatility_arbitrage/data/SPY_Options_2019_24',
        help='Path to options data directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/volatility_arb.yaml',
        help='Path to base strategy config'
    )
    parser.add_argument(
        '--ablation-config',
        type=str,
        default='config/ablation_config.yaml',
        help='Path to ablation study config'
    )
    parser.add_argument(
        '--permutations',
        type=int,
        default=50,
        help='Number of MCPT permutations per variant'
    )
    parser.add_argument(
        '--variants',
        type=str,
        default=None,
        help='Comma-separated list of variant names to run (e.g., "E1,E2,E3")'
    )
    parser.add_argument(
        '--skip-mcpt',
        action='store_true',
        help='Skip MCPT (just run backtests for quick comparison). By default, MCPT runs with 1000 permutations.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/ablation_study',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ABLATION STUDY FOR VOLATILITY ARBITRAGE ENHANCEMENTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Data: {args.data}")
    print(f"Config: {args.config}")
    print(f"MCPT Permutations: {args.permutations}")
    print(f"Skip MCPT: {args.skip_mcpt}")

    # Load ablation config
    ablation_config = load_ablation_config(args.ablation_config)
    # Default to 1000 permutations for statistically reliable MCPT results
    mcpt_perms = args.permutations or ablation_config['ablation'].get('mcpt_permutations', 1000)

    # Load options data
    print(f"\nLoading options data...")
    options_df = load_options_data(args.data)
    print(f"Loaded {len(options_df):,} records")
    print(f"Date range: {options_df['date'].min()} to {options_df['date'].max()}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get variants to run
    all_variants = ablation_config['ablation']['variants']
    if args.variants:
        variant_filter = set(args.variants.split(','))
        variants = [v for v in all_variants if any(vf in v['name'] for vf in variant_filter)]
        print(f"\nFiltered to {len(variants)} variants: {[v['name'] for v in variants]}")
    else:
        variants = all_variants

    # Run baseline first
    print("\n" + "=" * 70)
    print("RUNNING BASELINE")
    print("=" * 70)

    baseline_overrides = ablation_config['ablation']['baseline']['config_overrides']
    baseline_metrics = run_single_variant(
        variant_name="Baseline (v1.0)",
        config_overrides=baseline_overrides,
        options_df=options_df,
        base_config_path=args.config,
        mcpt_permutations=mcpt_perms,
        run_mcpt=not args.skip_mcpt,
    )

    # Run each variant
    # Pass baseline_overrides so each variant starts from clean baseline
    results = []
    for i, variant in enumerate(variants, 1):
        print(f"\n[{i}/{len(variants)}] Running {variant['name']}...")

        try:
            metrics = run_single_variant(
                variant_name=variant['name'],
                config_overrides=variant['config_overrides'],
                options_df=options_df,
                base_config_path=args.config,
                mcpt_permutations=mcpt_perms,
                run_mcpt=not args.skip_mcpt,
                baseline_overrides=baseline_overrides,  # Apply baseline first, then variant override
            )
            results.append(metrics)
        except Exception as e:
            print(f"ERROR running {variant['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'variant': variant['name'],
                'sharpe': None,
                'total_return': None,
                'mcpt_pvalue': None,
                'mcpt_status': 'ERROR',
                'error': str(e),
            })

    # Generate comparison table
    table = generate_comparison_table(results, baseline_metrics)
    print(table)

    # Save results
    results_path = output_dir / "ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_metrics,
            'variants': results,
            'config': {
                'mcpt_permutations': mcpt_perms,
                'data_path': args.data,
            }
        }, f, indent=2, default=str)

    # Save table as text
    table_path = output_dir / "ablation_summary.txt"
    with open(table_path, 'w') as f:
        f.write(table)

    # Save as CSV for easy analysis
    csv_path = output_dir / "ablation_results.csv"
    all_results = [baseline_metrics] + results
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to:")
    print(f"  - {results_path}")
    print(f"  - {table_path}")
    print(f"  - {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
