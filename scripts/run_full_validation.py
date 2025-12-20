#!/usr/bin/env python3
"""
Full Validation Suite: Comprehensive Strategy Testing.

Runs all statistical validation tests in a unified workflow:
1. Standard backtest (with NW-adjusted Sharpe)
2. Monte Carlo Permutation Test (MCPT)
3. Walk-forward validation
4. Block bootstrap

Provides clear pass/fail assessment and recommendations.

Usage:
    PYTHONPATH=./src:. python scripts/run_full_validation.py \
        --config config/volatility_arb.yaml \
        --data src/volatility_arbitrage/data/SPY_Options_2019_24 \
        --capital 100000 \
        --output reports/validation_results.json
"""

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load_json_options_data, run_qv_backtest
from run_walkforward import run_walkforward_fixed, run_backtest_silent, FoldResult

from volatility_arbitrage.core.config import load_strategy_config
from volatility_arbitrage.testing.validation_result import ValidationResult
from volatility_arbitrage.testing.validation_decision import (
    assess_mcpt_result,
    assess_walkforward_result,
    assess_bootstrap_result,
    determine_overall_status,
    calculate_hac_adjustment_estimate,
)
from volatility_arbitrage.testing.mcpt.config import MCPTConfig
from volatility_arbitrage.testing.mcpt.insample_test import run_insample_mcpt
from volatility_arbitrage.analysis.monte_carlo import (
    extract_trade_returns,
    block_bootstrap_resample,
    MonteCarloResult,
)
from volatility_arbitrage.backtest.metrics import calculate_sharpe_ratio


def run_mcpt_validation(
    options_df: pd.DataFrame,
    config,
    config_path: str,
    initial_capital: float,
    n_permutations: int = 1000,
    use_parallel: bool = True,
) -> Tuple[float, str]:
    """
    Run MCPT in-sample test.

    Returns:
        (p_value, status)
    """
    print("\n" + "=" * 60)
    print("RUNNING MCPT VALIDATION")
    print("=" * 60)
    print(f"Permutations: {n_permutations}")
    print(f"Parallel: {use_parallel}")

    mcpt_config = MCPTConfig()
    mcpt_config.n_permutations = n_permutations

    # Use partial to avoid pickle issues with nested functions
    strategy_func = partial(run_qv_backtest, config=config, initial_capital=initial_capital, config_path=config_path)

    try:
        result = run_insample_mcpt(
            options_df=options_df,
            strategy_func=strategy_func,
            config=mcpt_config,
            objective='sharpe',
            test_name='QV Strategy In-Sample',
            use_parallel=use_parallel,
        )

        print(f"\nReal Sharpe:      {result.real_metric:.3f}")
        print(f"Permuted Mean:    {result.mean_permuted:.3f}")
        print(f"Permuted Std:     {result.std_permuted:.3f}")
        print(f"p-value:          {result.p_value:.4f}")
        print(f"Z-score:          {result.z_score:.2f}")
        print(f"Status:           {result.status}")

        return result.p_value, result.status

    except Exception as e:
        print(f"\n❌ MCPT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, 'ERROR'


def run_walkforward_validation(
    options_df: pd.DataFrame,
    config_path: str,
    initial_capital: float,
) -> Tuple[float, float, float, int, int, str]:
    """
    Run walk-forward validation (fixed parameters).

    Returns:
        (avg_train_sharpe, avg_test_sharpe, efficiency, folds_passed, total_folds, status)
    """
    print("\n" + "=" * 60)
    print("RUNNING WALK-FORWARD VALIDATION")
    print("=" * 60)
    print("Mode: Fixed parameters (testing stability)")

    # Get available years from actual data
    available_years = sorted(options_df['date'].dt.year.unique())
    print(f"Available data years: {available_years}")

    # Create folds from consecutive available years
    # For [2019, 2020, 2021, 2024]:
    # Fold 1: Train 2019 → Test 2020
    # Fold 2: Train 2019-2020 → Test 2021
    # Fold 3: Train 2020-2021 → Test 2024

    folds = []
    for i in range(len(available_years) - 1):
        train_start_year = available_years[max(0, i - 1)] if i > 0 else available_years[i]
        train_end_year = available_years[i]
        test_year = available_years[i + 1]

        folds.append({
            'num': i + 1,
            'train_start': f'{train_start_year}-01-01',
            'train_end': f'{train_end_year}-12-31',
            'test_start': f'{test_year}-01-01',
            'test_end': f'{test_year}-12-31',
        })

    print(f"Generated {len(folds)} walk-forward folds")

    try:
        results = run_walkforward_fixed(
            options_df=options_df,
            config_path=config_path,
            folds=folds,
            initial_capital=initial_capital,
        )

        train_sharpes = [r.train_sharpe for r in results]
        test_sharpes = [r.test_sharpe for r in results]

        avg_train = np.mean(train_sharpes)
        avg_test = np.mean(test_sharpes)
        efficiency = avg_test / avg_train if avg_train > 0 else 0

        folds_passed = sum(1 for r in results if r.test_return > 0)
        total_folds = len(results)

        print(f"\nAvg Train Sharpe: {avg_train:.3f}")
        print(f"Avg Test Sharpe:  {avg_test:.3f}")
        print(f"Efficiency Ratio: {efficiency:.3f}")
        print(f"Profitable Folds: {folds_passed}/{total_folds}")

        status = assess_walkforward_result(efficiency, folds_passed, total_folds)
        print(f"Status:           {status}")

        return avg_train, avg_test, efficiency, folds_passed, total_folds, status

    except Exception as e:
        print(f"\n❌ WALK-FORWARD FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0, 5, 'ERROR'


def run_bootstrap_validation(
    backtest_results: Dict[str, Any],
    n_simulations: int = 10000,
    block_size: int = 3,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], str]:
    """
    Run block bootstrap validation.

    Returns:
        (sharpe_ci, nw_sharpe_ci, return_ci, status)
    """
    print("\n" + "=" * 60)
    print("RUNNING BLOCK BOOTSTRAP VALIDATION")
    print("=" * 60)
    print(f"Simulations: {n_simulations}")
    print(f"Block Size:  {block_size}")

    try:
        trade_log = backtest_results.get('trade_log', [])
        if not trade_log:
            print("\n❌ No trades found for bootstrap")
            return (0, 0), (0, 0), (0, 0), 'ERROR'

        trade_returns = extract_trade_returns(trade_log)
        print(f"Number of trades: {len(trade_returns)}")

        result = block_bootstrap_resample(
            trade_returns=trade_returns,
            n_simulations=n_simulations,
            block_size=block_size,
            initial_capital=100000,
            random_seed=42,
        )

        # Also calculate NW Sharpe for each bootstrap sample
        # This is computationally expensive, so we'll approximate
        # by scaling the standard Sharpe CIs
        nw_adjustment = calculate_hac_adjustment_estimate(
            float(backtest_results['nw_sharpe']),
            float(backtest_results['sharpe'])
        )

        nw_sharpe_ci_lower = result.sharpe_ci_lower / nw_adjustment
        nw_sharpe_ci_upper = result.sharpe_ci_upper / nw_adjustment

        sharpe_ci = (result.sharpe_ci_lower, result.sharpe_ci_upper)
        nw_sharpe_ci = (nw_sharpe_ci_lower, nw_sharpe_ci_upper)
        return_ci = (result.return_ci_lower, result.return_ci_upper)

        print(f"\nSharpe 95% CI:    [{sharpe_ci[0]:.3f}, {sharpe_ci[1]:.3f}]")
        print(f"NW Sharpe 95% CI: [{nw_sharpe_ci[0]:.3f}, {nw_sharpe_ci[1]:.3f}]")
        print(f"Return 95% CI:    [{return_ci[0]:.1f}%, {return_ci[1]:.1f}%]")

        status = assess_bootstrap_result(sharpe_ci, nw_sharpe_ci)
        print(f"Status:           {status}")

        return sharpe_ci, nw_sharpe_ci, return_ci, status

    except Exception as e:
        print(f"\n❌ BOOTSTRAP FAILED: {e}")
        import traceback
        traceback.print_exc()
        return (0, 0), (0, 0), (0, 0), 'ERROR'


def print_comprehensive_report(result: ValidationResult):
    """Print formatted comprehensive validation report."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE VALIDATION REPORT")
    print("=" * 60)

    # Backtest results
    print("\nBACKTEST RESULTS")
    print("-" * 60)
    print(f"  Standard Sharpe:   {result.backtest_sharpe:>10.2f}")
    print(f"  Newey-West Sharpe: {result.backtest_nw_sharpe:>10.2f}", end="")
    if result.backtest_nw_sharpe < 0:
        print("  ⚠️  HIGH AUTOCORRELATION")
    else:
        print()
    print(f"  Total Return:      {result.total_return:>10.1f}%")
    print(f"  Max Drawdown:      {result.max_drawdown:>10.1f}%")
    print(f"  Number of Trades:  {result.num_trades:>10}")

    # MCPT results
    print("\nMCPT VALIDATION")
    print("-" * 60)
    if result.mcpt_status not in ['SKIPPED', 'ERROR'] and result.mcpt_pvalue is not None:
        print(f"  p-value:           {result.mcpt_pvalue:>10.4f}")
        print(f"  Status:            {result.mcpt_status:>10}", end="")
        if result.mcpt_status == 'PASS':
            print("  ✓")
        elif result.mcpt_status == 'MARGINAL':
            print("  ⚠️")
        else:
            print("  ❌")
    else:
        print(f"  Status:            {result.mcpt_status:>10}")

    # Walk-forward results
    print("\nWALK-FORWARD VALIDATION")
    print("-" * 60)
    if result.wf_status != 'SKIPPED':
        print(f"  Avg Train Sharpe:  {result.wf_avg_train_sharpe:>10.2f}")
        print(f"  Avg Test Sharpe:   {result.wf_avg_test_sharpe:>10.2f}")
        print(f"  Efficiency Ratio:  {result.wf_avg_efficiency:>10.2f}")
        print(f"  Folds Passed:      {result.wf_folds_passed:>5}/{result.wf_total_folds:<5}")
        print(f"  Status:            {result.wf_status:>10}", end="")
        if result.wf_status == 'PASS':
            print("  ✓")
        elif result.wf_status == 'MARGINAL':
            print("  ⚠️")
        else:
            print("  ❌")
    else:
        print(f"  Status:            {result.wf_status:>10}")

    # Bootstrap results
    print("\nBLOCK BOOTSTRAP")
    print("-" * 60)
    if result.bootstrap_status != 'SKIPPED':
        sl, su = result.bootstrap_sharpe_ci
        nsl, nsu = result.bootstrap_nw_sharpe_ci
        rl, ru = result.bootstrap_return_ci
        print(f"  Sharpe 95% CI:     [{sl:>6.2f}, {su:>6.2f}]")
        print(f"  NW Sharpe 95% CI:  [{nsl:>6.2f}, {nsu:>6.2f}]", end="")
        if nsl < 0:
            print("  ⚠️  Includes negative")
        else:
            print()
        print(f"  Return 95% CI:     [{rl:>6.1f}%, {ru:>6.1f}%]")
        print(f"  Status:            {result.bootstrap_status:>10}", end="")
        if result.bootstrap_status == 'PASS':
            print("  ✓")
        elif result.bootstrap_status == 'MARGINAL':
            print("  ⚠️")
        else:
            print("  ❌")
    else:
        print(f"  Status:            {result.bootstrap_status:>10}")

    # Overall assessment
    print("\n" + "-" * 60)
    print(f"OVERALL ASSESSMENT: {result.overall_status}")
    print("-" * 60)

    # Recommendation
    print("\n" + result.get_recommendation())
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Full Validation Suite')
    parser.add_argument(
        '--config',
        type=str,
        default='config/volatility_arb.yaml',
        help='Strategy configuration file',
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
        '--output',
        type=str,
        default='reports/validation_results.json',
        help='Output JSON file for results',
    )
    parser.add_argument(
        '--skip-mcpt',
        action='store_true',
        help='Skip MCPT (faster)',
    )
    parser.add_argument(
        '--skip-walkforward',
        action='store_true',
        help='Skip walk-forward validation',
    )
    parser.add_argument(
        '--skip-bootstrap',
        action='store_true',
        help='Skip block bootstrap',
    )
    parser.add_argument(
        '--mcpt-permutations',
        type=int,
        default=1000,
        help='Number of MCPT permutations',
    )
    parser.add_argument(
        '--bootstrap-samples',
        type=int,
        default=10000,
        help='Number of bootstrap samples',
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FULL VALIDATION SUITE")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading options data...")
    options_df = load_json_options_data(args.data)

    # Load config
    config = load_strategy_config(Path(args.config))

    # Step 1: Run standard backtest
    print("\n" + "=" * 60)
    print("STEP 1: RUNNING STANDARD BACKTEST")
    print("=" * 60)
    backtest_results = run_qv_backtest(options_df, config, args.capital, args.config)

    backtest_sharpe = backtest_results['sharpe']
    backtest_nw_sharpe = backtest_results['nw_sharpe']
    total_return = backtest_results['total_return']
    max_drawdown = backtest_results['max_drawdown']
    num_trades = backtest_results['trades']

    # Step 2: Run MCPT
    if not args.skip_mcpt:
        mcpt_pvalue, mcpt_status = run_mcpt_validation(
            options_df, config, args.config, args.capital,
            n_permutations=args.mcpt_permutations,
        )
    else:
        print("\n⏭️  Skipping MCPT")
        mcpt_pvalue, mcpt_status = None, 'SKIPPED'

    # Step 3: Run walk-forward
    if not args.skip_walkforward:
        wf_train, wf_test, wf_eff, wf_passed, wf_total, wf_status = run_walkforward_validation(
            options_df, args.config, args.capital
        )
    else:
        print("\n⏭️  Skipping walk-forward")
        wf_train, wf_test, wf_eff, wf_passed, wf_total, wf_status = 0, 0, 0, 0, 5, 'SKIPPED'

    # Step 4: Run bootstrap
    if not args.skip_bootstrap:
        sharpe_ci, nw_sharpe_ci, return_ci, bootstrap_status = run_bootstrap_validation(
            backtest_results,
            n_simulations=args.bootstrap_samples,
        )
    else:
        print("\n⏭️  Skipping bootstrap")
        sharpe_ci, nw_sharpe_ci, return_ci, bootstrap_status = (0, 0), (0, 0), (0, 0), 'SKIPPED'

    # Determine overall status
    overall_status = determine_overall_status(mcpt_status, wf_status, bootstrap_status)

    # Create validation result
    validation_result = ValidationResult(
        backtest_sharpe=Decimal(str(backtest_sharpe)),
        backtest_nw_sharpe=Decimal(str(backtest_nw_sharpe)),
        total_return=Decimal(str(total_return)),
        max_drawdown=Decimal(str(max_drawdown)),
        num_trades=num_trades,
        mcpt_pvalue=mcpt_pvalue,
        mcpt_status=mcpt_status,
        wf_avg_train_sharpe=wf_train,
        wf_avg_test_sharpe=wf_test,
        wf_avg_efficiency=wf_eff,
        wf_folds_passed=wf_passed,
        wf_total_folds=wf_total,
        wf_status=wf_status,
        bootstrap_sharpe_ci=sharpe_ci,
        bootstrap_nw_sharpe_ci=nw_sharpe_ci,
        bootstrap_return_ci=return_ci,
        bootstrap_status=bootstrap_status,
        overall_status=overall_status,
    )

    # Print comprehensive report
    print_comprehensive_report(validation_result)

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = {
        'timestamp': datetime.now().isoformat(),
        'config': args.config,
        'data_path': args.data,
        'initial_capital': args.capital,
        'validation_result': validation_result.get_summary_dict(),
    }

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
