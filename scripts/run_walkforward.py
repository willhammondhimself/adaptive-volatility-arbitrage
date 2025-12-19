#!/usr/bin/env python3
"""
Walk-Forward Validation for QV Strategy.

Tests parameter stability across expanding windows:
  Fold 1: Train 2019 → Test 2020
  Fold 2: Train 2019-2020 → Test 2021
  Fold 3: Train 2019-2021 → Test 2022
  Fold 4: Train 2019-2022 → Test 2023
  Fold 5: Train 2019-2023 → Test 2024

Two modes:
  --fixed: Use aggressive config params on all folds (test stability)
  --reoptimize: Re-run grid search on each train window (test adaptability)

Usage:
    PYTHONPATH=./src:. python scripts/run_walkforward.py --fixed
    PYTHONPATH=./src:. python scripts/run_walkforward.py --reoptimize
"""

import argparse
import io
import itertools
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from volatility_arbitrage.core.config import load_strategy_config

sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load_json_options_data, run_qv_backtest


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float
    train_return: float
    train_dd: float
    test_sharpe: float
    test_return: float
    test_dd: float
    test_win_rate: float
    test_trades: int
    best_params: Optional[Dict[str, Any]] = None


def run_backtest_silent(
    options_df: pd.DataFrame,
    config,
    config_path: str,
    initial_capital: float = 100000,
) -> Dict[str, Any]:
    """Run backtest with suppressed output."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        results = run_qv_backtest(
            options_df,
            config,
            initial_capital,
            config_path=config_path
        )
        return results
    finally:
        sys.stdout = old_stdout


def create_config_variant(base_config_path: str, overrides: Dict[str, Any]) -> Tuple[str, Any]:
    """Create a temporary config with parameter overrides."""
    with open(base_config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    strategy = config_dict.get('strategy', {})
    for key, value in overrides.items():
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

    temp_path = Path('/tmp/walkforward_temp.yaml')
    with open(temp_path, 'w') as f:
        yaml.dump(config_dict, f)

    config = load_strategy_config(temp_path)
    return str(temp_path), config


def grid_search_fold(
    train_df: pd.DataFrame,
    base_config_path: str,
    param_grid: Dict[str, List],
    initial_capital: float = 100000,
) -> Tuple[Dict[str, Any], float]:
    """Run grid search on training data, return best params and sharpe."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    best_sharpe = -999
    best_params = None

    for combo in combinations:
        params = dict(zip(keys, combo))

        try:
            temp_path, config = create_config_variant(base_config_path, params)
            results = run_backtest_silent(train_df, config, temp_path, initial_capital)

            if results['sharpe'] > best_sharpe:
                best_sharpe = results['sharpe']
                best_params = params
        except Exception:
            continue

    return best_params, best_sharpe


def run_walkforward_fixed(
    options_df: pd.DataFrame,
    config_path: str,
    folds: List[Dict],
    initial_capital: float = 100000,
) -> List[FoldResult]:
    """Run walk-forward with fixed parameters (from config)."""
    results = []
    config = load_strategy_config(Path(config_path))

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"FOLD {fold['num']}: Train {fold['train_start']}-{fold['train_end']} → Test {fold['test_start']}-{fold['test_end']}")
        print('='*60)

        # Filter data
        train_df = options_df[
            (options_df['date'] >= fold['train_start']) &
            (options_df['date'] <= fold['train_end'])
        ]
        test_df = options_df[
            (options_df['date'] >= fold['test_start']) &
            (options_df['date'] <= fold['test_end'])
        ]

        print(f"Train records: {len(train_df):,}, Test records: {len(test_df):,}")

        # Run on train
        train_results = run_backtest_silent(train_df, config, config_path, initial_capital)

        # Run on test
        test_results = run_backtest_silent(test_df, config, config_path, initial_capital)

        # Extract metrics
        result = FoldResult(
            fold_num=fold['num'],
            train_start=fold['train_start'],
            train_end=fold['train_end'],
            test_start=fold['test_start'],
            test_end=fold['test_end'],
            train_sharpe=train_results['sharpe'],
            train_return=train_results['total_return'],
            train_dd=train_results['max_drawdown'],
            test_sharpe=test_results['sharpe'],
            test_return=test_results['total_return'],
            test_dd=test_results['max_drawdown'],
            test_win_rate=0,  # Calculate from trades
            test_trades=test_results['trades'],
        )

        # Calculate win rate from trades
        trades = test_results.get('trade_log', [])
        exit_trades = [t for t in trades if t.get('action') == 'EXIT']
        if exit_trades:
            winners = sum(1 for t in exit_trades if t.get('position_return', 0) > 0)
            result.test_win_rate = winners / len(exit_trades) * 100

        results.append(result)

        print(f"\nTrain: Sharpe={result.train_sharpe:.2f}, Return={result.train_return:.1f}%, DD={result.train_dd:.1f}%")
        print(f"Test:  Sharpe={result.test_sharpe:.2f}, Return={result.test_return:.1f}%, DD={result.test_dd:.1f}%")

    return results


def run_walkforward_reoptimize(
    options_df: pd.DataFrame,
    base_config_path: str,
    folds: List[Dict],
    initial_capital: float = 100000,
) -> List[FoldResult]:
    """Run walk-forward with re-optimization on each fold."""
    results = []

    # Full grid for re-optimization
    param_grid = {
        'kelly_fraction': [0.25, 0.40, 0.50],
        'consensus_threshold': [0.10, 0.15, 0.20],
        'asymmetric_targets.short_vol_profit_target': [0.03, 0.04, 0.05],
        'asymmetric_targets.short_vol_stop_loss': [-0.03, -0.04, -0.05],
        'short_vol_leverage': [1.0, 1.15, 1.3],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"FOLD {fold['num']}: Train {fold['train_start']}-{fold['train_end']} → Test {fold['test_start']}-{fold['test_end']}")
        print('='*60)

        # Filter data
        train_df = options_df[
            (options_df['date'] >= fold['train_start']) &
            (options_df['date'] <= fold['train_end'])
        ]
        test_df = options_df[
            (options_df['date'] >= fold['test_start']) &
            (options_df['date'] <= fold['test_end'])
        ]

        print(f"Train records: {len(train_df):,}, Test records: {len(test_df):,}")
        print(f"Running grid search ({total_combos} combinations)...")

        # Optimize on train
        best_params, train_sharpe = grid_search_fold(
            train_df, base_config_path, param_grid, initial_capital
        )

        if best_params is None:
            print("  ERROR: No valid config found")
            continue

        print(f"  Best train Sharpe: {train_sharpe:.2f}")
        print(f"  Best params: {best_params}")

        # Create optimized config and run on train (for full metrics)
        temp_path, config = create_config_variant(base_config_path, best_params)
        train_results = run_backtest_silent(train_df, config, temp_path, initial_capital)

        # Run on test with optimized params
        test_results = run_backtest_silent(test_df, config, temp_path, initial_capital)

        result = FoldResult(
            fold_num=fold['num'],
            train_start=fold['train_start'],
            train_end=fold['train_end'],
            test_start=fold['test_start'],
            test_end=fold['test_end'],
            train_sharpe=train_results['sharpe'],
            train_return=train_results['total_return'],
            train_dd=train_results['max_drawdown'],
            test_sharpe=test_results['sharpe'],
            test_return=test_results['total_return'],
            test_dd=test_results['max_drawdown'],
            test_win_rate=0,
            test_trades=test_results['trades'],
            best_params=best_params,
        )

        # Calculate win rate
        trades = test_results.get('trade_log', [])
        exit_trades = [t for t in trades if t.get('action') == 'EXIT']
        if exit_trades:
            winners = sum(1 for t in exit_trades if t.get('position_return', 0) > 0)
            result.test_win_rate = winners / len(exit_trades) * 100

        results.append(result)

        print(f"\nTrain: Sharpe={result.train_sharpe:.2f}, Return={result.train_return:.1f}%, DD={result.train_dd:.1f}%")
        print(f"Test:  Sharpe={result.test_sharpe:.2f}, Return={result.test_return:.1f}%, DD={result.test_dd:.1f}%")

    return results


def print_summary(results: List[FoldResult], mode: str):
    """Print walk-forward summary statistics."""
    print("\n" + "="*70)
    print(f"WALK-FORWARD SUMMARY ({mode.upper()} MODE)")
    print("="*70)

    # Per-fold table
    print("\n{:<6} {:<12} {:<12} {:<10} {:<10} {:<10} {:<8}".format(
        "Fold", "Test Period", "Train Sharpe", "Test Sharpe", "Test Ret%", "Test DD%", "Trades"
    ))
    print("-"*70)

    for r in results:
        test_period = f"{r.test_start[:4]}".replace("20", "")
        print("{:<6} {:<12} {:<12.2f} {:<10.2f} {:<10.1f} {:<10.1f} {:<8}".format(
            r.fold_num,
            f"{r.test_start[:7]} to {r.test_end[:7]}",
            r.train_sharpe,
            r.test_sharpe,
            r.test_return,
            r.test_dd,
            r.test_trades
        ))

    # Aggregate statistics
    test_sharpes = [r.test_sharpe for r in results]
    test_returns = [r.test_return for r in results]
    test_dds = [r.test_dd for r in results]
    train_sharpes = [r.train_sharpe for r in results]

    print("-"*70)
    print("\nAGGREGATE STATISTICS:")
    print(f"  Test Sharpe:  Mean={np.mean(test_sharpes):.2f}, Std={np.std(test_sharpes):.2f}")
    print(f"  Test Return:  Mean={np.mean(test_returns):.1f}%, Std={np.std(test_returns):.1f}%")
    print(f"  Test Max DD:  Mean={np.mean(test_dds):.1f}%, Worst={min(test_dds):.1f}%")
    print(f"  Train→Test Degradation: {np.mean(train_sharpes):.2f} → {np.mean(test_sharpes):.2f}")

    # Stability score
    stability = 1 - (np.std(test_sharpes) / np.mean(test_sharpes)) if np.mean(test_sharpes) > 0 else 0
    print(f"\n  Stability Score: {stability:.2f} (higher is better, want >0.7)")

    # Pass/Fail criteria
    print("\n" + "-"*70)
    print("VALIDATION CRITERIA:")

    all_profitable = all(r.test_return > 0 for r in results)
    avg_sharpe_ok = np.mean(test_sharpes) >= 0.8
    no_extreme_dd = all(r.test_dd > -25 for r in results)

    print(f"  ✅ All folds profitable: {'PASS' if all_profitable else 'FAIL'}")
    print(f"  {'✅' if avg_sharpe_ok else '❌'} Avg OOS Sharpe ≥ 0.8: {'PASS' if avg_sharpe_ok else 'FAIL'} ({np.mean(test_sharpes):.2f})")
    print(f"  {'✅' if no_extreme_dd else '❌'} No fold DD > 25%: {'PASS' if no_extreme_dd else 'FAIL'}")

    overall_pass = all_profitable and avg_sharpe_ok and no_extreme_dd
    print(f"\n  OVERALL: {'✅ PASS' if overall_pass else '❌ FAIL'}")

    # Parameter stability (if reoptimized)
    if results[0].best_params is not None:
        print("\n" + "-"*70)
        print("PARAMETER STABILITY ACROSS FOLDS:")
        all_keys = set()
        for r in results:
            if r.best_params:
                all_keys.update(r.best_params.keys())

        for key in sorted(all_keys):
            values = [r.best_params.get(key) for r in results if r.best_params]
            unique = set(values)
            if len(unique) == 1:
                print(f"  {key}: {values[0]} (stable)")
            else:
                print(f"  {key}: {values} (varies)")


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Validation')
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
        '--fixed',
        action='store_true',
        help='Run with fixed parameters from config',
    )
    parser.add_argument(
        '--reoptimize',
        action='store_true',
        help='Re-optimize parameters on each fold',
    )
    args = parser.parse_args()

    if not args.fixed and not args.reoptimize:
        print("ERROR: Must specify --fixed or --reoptimize")
        return

    print("="*60)
    print("WALK-FORWARD VALIDATION")
    print("="*60)

    # Define folds (expanding windows)
    folds = [
        {'num': 1, 'train_start': '2019-01-01', 'train_end': '2019-12-31',
         'test_start': '2020-01-01', 'test_end': '2020-12-31'},
        {'num': 2, 'train_start': '2019-01-01', 'train_end': '2020-12-31',
         'test_start': '2021-01-01', 'test_end': '2021-12-31'},
        {'num': 3, 'train_start': '2019-01-01', 'train_end': '2021-12-31',
         'test_start': '2022-01-01', 'test_end': '2022-12-31'},
        {'num': 4, 'train_start': '2019-01-01', 'train_end': '2022-12-31',
         'test_start': '2023-01-01', 'test_end': '2023-12-31'},
        {'num': 5, 'train_start': '2019-01-01', 'train_end': '2023-12-31',
         'test_start': '2024-01-01', 'test_end': '2024-12-31'},
    ]

    print(f"\nConfig: {args.config}")
    print(f"Mode: {'Fixed Parameters' if args.fixed else 'Re-Optimize Each Fold'}")
    print(f"Folds: {len(folds)}")

    # Load data
    print(f"\nLoading data from {args.data}...")
    options_df = load_json_options_data(args.data)
    options_df['date'] = pd.to_datetime(options_df['date'])
    print(f"  Records: {len(options_df):,}")
    print(f"  Date range: {options_df['date'].min().date()} to {options_df['date'].max().date()}")

    # Run walk-forward
    if args.fixed:
        results = run_walkforward_fixed(
            options_df, args.config, folds, args.capital
        )
        print_summary(results, "fixed")
    else:
        results = run_walkforward_reoptimize(
            options_df, args.config, folds, args.capital
        )
        print_summary(results, "reoptimize")


if __name__ == "__main__":
    main()
