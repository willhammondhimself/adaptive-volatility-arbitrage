#!/usr/bin/env python3
"""
Sharpe Ratio Optimization via Grid Search.

Searches parameter space to find optimal configuration for Sharpe ratio.
Only uses training data (2019-2022) to avoid overfitting.
"""

import argparse
import itertools
import json
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import yaml

from volatility_arbitrage.core.config import load_strategy_config

# Import from run_backtest
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load_json_options_data, run_qv_backtest


def create_config_variant(base_config_path: str, overrides: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Create a temporary config with parameter overrides.

    Returns (temp_config_path, config_object)
    """
    with open(base_config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Apply overrides to strategy section
    strategy = config_dict.get('strategy', {})
    for key, value in overrides.items():
        if '.' in key:
            # Handle nested keys like 'asymmetric_targets.short_vol_profit_target'
            parts = key.split('.')
            target = strategy
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            strategy[key] = value
    config_dict['strategy'] = strategy

    # Write to temp file
    temp_path = Path('/tmp/volatility_arb_grid_search.yaml')
    with open(temp_path, 'w') as f:
        yaml.dump(config_dict, f)

    # Load as config object
    config = load_strategy_config(temp_path)

    return str(temp_path), config


def run_single_backtest(
    options_df: pd.DataFrame,
    config,
    config_path: str,
    initial_capital: float = 100000,
    verbose: bool = False
) -> Dict[str, float]:
    """Run a single backtest and return key metrics."""
    import io
    import sys

    # Capture stdout to suppress output unless verbose
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    try:
        results = run_qv_backtest(
            options_df,
            config,
            initial_capital,
            config_path=config_path
        )
        return {
            'sharpe': results['sharpe'],
            'nw_sharpe': results['nw_sharpe'],
            'total_return': results['total_return'],
            'max_drawdown': results['max_drawdown'],
            'trades': results['trades'],
        }
    except Exception as e:
        return {
            'sharpe': -999,
            'nw_sharpe': -999,
            'total_return': -999,
            'max_drawdown': -999,
            'trades': 0,
            'error': str(e)
        }
    finally:
        if not verbose:
            sys.stdout = old_stdout


def grid_search(
    options_df: pd.DataFrame,
    base_config_path: str,
    param_grid: Dict[str, List],
    initial_capital: float = 100000,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run grid search over parameter combinations.

    Returns DataFrame with all results.
    """
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"\nGrid Search: {len(combinations)} combinations")
    print(f"Parameters: {keys}")
    print("-" * 60)

    results = []
    best_sharpe = -999
    best_params = None

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # Create config variant
        try:
            temp_path, config = create_config_variant(base_config_path, params)
        except Exception as e:
            print(f"[{i+1}/{len(combinations)}] Config error: {e}")
            continue

        # Run backtest
        metrics = run_single_backtest(
            options_df,
            config,
            temp_path,
            initial_capital,
            verbose=verbose
        )

        # Store results
        result = {**params, **metrics}
        results.append(result)

        # Update best
        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_params = params

        # Progress update
        sharpe = metrics['sharpe']
        ret = metrics['total_return']
        dd = metrics['max_drawdown']
        trades = metrics['trades']

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(combinations)}] Sharpe={sharpe:.2f} Ret={ret:.1f}% DD={dd:.1f}% Trades={trades}")

    print("-" * 60)
    print(f"\nBest Sharpe: {best_sharpe:.3f}")
    print(f"Best Params: {best_params}")

    return pd.DataFrame(results)


def coarse_then_fine_search(
    options_df: pd.DataFrame,
    base_config_path: str,
    initial_capital: float = 100000,
) -> pd.DataFrame:
    """
    Two-stage optimization: coarse grid then fine-tune around best.
    """
    print("\n" + "="*60)
    print("PHASE 1: COARSE GRID SEARCH")
    print("="*60)

    # Load suggested weights from signal analysis
    suggested_weights_path = Path('reports/suggested_weights.json')
    if suggested_weights_path.exists():
        with open(suggested_weights_path) as f:
            suggested_weights = json.load(f)
        print(f"Loaded suggested weights from signal analysis")
    else:
        suggested_weights = None
        print("No suggested weights found, using defaults")

    # Coarse grid
    coarse_grid = {
        # Signal weights (from analysis)
        'weight_iv_premium': [0.20, 0.25, 0.30] if not suggested_weights else [
            round(suggested_weights['iv_premium'] - 0.05, 2),
            round(suggested_weights['iv_premium'], 2),
            round(suggested_weights['iv_premium'] + 0.05, 2)
        ],

        # Entry threshold
        'consensus_threshold': [0.12, 0.15, 0.18],

        # Regime scalars
        'regime_extreme_low_scalar': [1.5, 1.8, 2.0],
        'regime_crisis_scalar': [0.3, 0.5],

        # Enhancements
        'use_asymmetric_targets': [True, False],
    }

    coarse_results = grid_search(
        options_df,
        base_config_path,
        coarse_grid,
        initial_capital
    )

    # Find best from coarse
    best_idx = coarse_results['sharpe'].idxmax()
    best_coarse = coarse_results.loc[best_idx]
    print(f"\nBest coarse config: Sharpe={best_coarse['sharpe']:.3f}")

    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNE SEARCH")
    print("="*60)

    # Fine grid around best coarse parameters
    best_iv_weight = best_coarse['weight_iv_premium']
    best_threshold = best_coarse['consensus_threshold']
    best_regime_low = best_coarse['regime_extreme_low_scalar']

    fine_grid = {
        'weight_iv_premium': [
            max(0.15, best_iv_weight - 0.03),
            best_iv_weight,
            min(0.35, best_iv_weight + 0.03)
        ],
        'consensus_threshold': [
            max(0.10, best_threshold - 0.02),
            best_threshold,
            min(0.22, best_threshold + 0.02)
        ],
        'regime_extreme_low_scalar': [
            max(1.3, best_regime_low - 0.2),
            best_regime_low,
            min(2.2, best_regime_low + 0.2)
        ],
        'use_asymmetric_targets': [best_coarse['use_asymmetric_targets']],
        'regime_crisis_scalar': [best_coarse['regime_crisis_scalar']],

        # Add profit targets if asymmetric is on
        'asymmetric_targets.short_vol_profit_target': [0.03, 0.04, 0.05] if best_coarse['use_asymmetric_targets'] else [0.03],
        'asymmetric_targets.short_vol_stop_loss': [-0.04, -0.06] if best_coarse['use_asymmetric_targets'] else [-0.06],
    }

    fine_results = grid_search(
        options_df,
        base_config_path,
        fine_grid,
        initial_capital
    )

    # Combine results
    all_results = pd.concat([coarse_results, fine_results], ignore_index=True)
    all_results = all_results.sort_values('sharpe', ascending=False)

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Optimize Sharpe ratio via grid search')
    parser.add_argument(
        '--config',
        type=str,
        default='config/volatility_arb.yaml',
        help='Base config file',
    )
    parser.add_argument(
        '--data',
        type=str,
        default='src/volatility_arbitrage/data/SPY_Options_2019_24',
        help='Options data directory',
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2022-12-31',
        help='End date for training (to avoid OOS data)',
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/optimization_results.csv',
        help='Output CSV path',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with minimal grid',
    )
    args = parser.parse_args()

    print("="*60)
    print("SHARPE RATIO OPTIMIZATION")
    print("="*60)

    # Load options data
    print(f"\nLoading options data from {args.data}...")
    options_df = load_json_options_data(args.data)

    # Filter to training period
    if args.end_date:
        end_dt = pd.to_datetime(args.end_date)
        options_df = options_df[options_df['date'] <= end_dt]
        print(f"Filtered to training period ending {args.end_date}")

    print(f"  Records: {len(options_df):,}")
    print(f"  Date range: {options_df['date'].min()} to {options_df['date'].max()}")

    if args.quick:
        # Quick test grid
        print("\n[QUICK MODE] Running minimal grid for testing...")
        quick_grid = {
            'consensus_threshold': [0.15, 0.18],
            'regime_extreme_low_scalar': [1.5, 2.0],
        }
        results = grid_search(
            options_df,
            args.config,
            quick_grid,
            args.capital
        )
    else:
        # Full coarse-then-fine search
        results = coarse_then_fine_search(
            options_df,
            args.config,
            args.capital
        )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Show top 10
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS")
    print("="*60)
    top_10 = results.nlargest(10, 'sharpe')
    for i, (_, row) in enumerate(top_10.iterrows()):
        print(f"\n#{i+1}: Sharpe={row['sharpe']:.3f} Return={row['total_return']:.1f}% DD={row['max_drawdown']:.1f}%")
        # Print key params
        for col in results.columns:
            if col not in ['sharpe', 'nw_sharpe', 'total_return', 'max_drawdown', 'trades', 'error']:
                print(f"  {col}: {row[col]}")

    # Save best config
    best = results.iloc[0]
    best_params = {col: best[col] for col in results.columns
                   if col not in ['sharpe', 'nw_sharpe', 'total_return', 'max_drawdown', 'trades', 'error']}

    print("\n" + "="*60)
    print("CREATING OPTIMIZED CONFIG")
    print("="*60)

    # Create optimized config file
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    strategy = config_dict.get('strategy', {})
    for key, value in best_params.items():
        if pd.notna(value):  # Skip NaN values
            if '.' in key:
                parts = key.split('.')
                target = strategy
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value
            else:
                strategy[key] = value

    config_dict['strategy'] = strategy

    optimized_path = Path('config/volatility_arb_optimized.yaml')
    with open(optimized_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Optimized config saved to: {optimized_path}")

    print(f"\nBest Training Sharpe: {best['sharpe']:.3f}")
    print(f"Best Training Return: {best['total_return']:.1f}%")
    print(f"Best Training Max DD: {best['max_drawdown']:.1f}%")


if __name__ == "__main__":
    main()
