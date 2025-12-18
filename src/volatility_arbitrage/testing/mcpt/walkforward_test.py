"""
Walk-forward Monte Carlo Permutation Test.

Implements MCPT for walk-forward analysis, with two modes:
1. Test-only permutation: Permute only test data (keeps training real)
2. Full permutation: Permute both training and test data
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Literal
import numpy as np
import pandas as pd

from volatility_arbitrage.testing.mcpt.config import MCPTConfig
from volatility_arbitrage.testing.mcpt.permutation import bar_permute
from volatility_arbitrage.testing.mcpt.utils import calculate_objective, compute_pvalue


@dataclass
class WalkForwardFold:
    """Single fold in walk-forward analysis."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    real_test_metric: float
    permuted_test_metrics: List[float]
    p_value: float

    @property
    def status(self) -> str:
        if self.p_value < 0.01:
            return "PASS"
        elif self.p_value < 0.05:
            return "MARGINAL"
        return "FAIL"


@dataclass
class WalkForwardMCPTResult:
    """
    Results from walk-forward Monte Carlo Permutation Test.

    Contains aggregate metrics across all folds and per-fold breakdown.
    """
    test_name: str
    permutation_type: Literal['test_only', 'full']
    real_test_metric: float  # Aggregate across folds
    metric_name: str
    permuted_test_metrics: List[float]  # Aggregate across all permutations
    p_value: float  # Aggregate p-value
    mean_permuted: float
    std_permuted: float
    fold_results: List[WalkForwardFold]
    n_permutations: int
    n_folds: int

    @property
    def is_significant(self) -> bool:
        """Check if aggregate result is significant (p < 0.01)."""
        return self.p_value < 0.01

    @property
    def is_marginal(self) -> bool:
        """Check if aggregate result is marginal (0.01 <= p < 0.05)."""
        return 0.01 <= self.p_value < 0.05

    @property
    def status(self) -> str:
        """Get status string."""
        if self.is_significant:
            return "PASS"
        elif self.is_marginal:
            return "MARGINAL"
        return "FAIL"

    @property
    def fold_pass_rate(self) -> float:
        """Percentage of folds that passed (p < 0.05)."""
        passed = sum(1 for f in self.fold_results if f.p_value < 0.05)
        return passed / len(self.fold_results) * 100 if self.fold_results else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'permutation_type': self.permutation_type,
            'real_test_metric': self.real_test_metric,
            'metric_name': self.metric_name,
            'p_value': self.p_value,
            'mean_permuted': self.mean_permuted,
            'std_permuted': self.std_permuted,
            'n_permutations': self.n_permutations,
            'n_folds': self.n_folds,
            'status': self.status,
            'fold_pass_rate': self.fold_pass_rate,
        }

    def __repr__(self) -> str:
        return (
            f"WalkForwardMCPTResult(test_name='{self.test_name}', "
            f"type={self.permutation_type}, real={self.real_test_metric:.3f}, "
            f"p={self.p_value:.4f}, status={self.status})"
        )


def generate_walkforward_folds(
    options_df: pd.DataFrame,
    train_window_days: int,
    test_window_days: int,
    step_days: Optional[int] = None,
    date_column: str = 'date',
) -> List[Dict[str, Any]]:
    """
    Generate walk-forward folds from options data.

    Args:
        options_df: Options data with date column
        train_window_days: Training window size in trading days
        test_window_days: Test window size in trading days
        step_days: Step size between folds (default = test_window_days)
        date_column: Name of date column

    Returns:
        List of fold dictionaries with train/test date ranges
    """
    if step_days is None:
        step_days = test_window_days

    # Get unique sorted dates
    dates = sorted(options_df[date_column].unique())
    n_dates = len(dates)

    folds = []
    fold_id = 0

    # Start from position where we have enough training data
    start_idx = 0

    while True:
        train_start_idx = start_idx
        train_end_idx = train_start_idx + train_window_days - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_window_days - 1

        # Check if we have enough data for this fold
        if test_end_idx >= n_dates:
            break

        folds.append({
            'fold_id': fold_id,
            'train_start': pd.Timestamp(dates[train_start_idx]),
            'train_end': pd.Timestamp(dates[train_end_idx]),
            'test_start': pd.Timestamp(dates[test_start_idx]),
            'test_end': pd.Timestamp(dates[test_end_idx]),
        })

        fold_id += 1
        start_idx += step_days

    return folds


def run_walkforward_mcpt(
    options_df: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], Dict[str, Any]],
    config: MCPTConfig,
    permutation_type: Literal['test_only', 'full'] = 'test_only',
    objective: str = 'sharpe',
    test_name: Optional[str] = None,
    folds: Optional[List[Dict[str, Any]]] = None,
) -> WalkForwardMCPTResult:
    """
    Run walk-forward Monte Carlo Permutation Test.

    Two modes:

    1. permutation_type='test_only' (default):
       - Train on REAL data, test on PERMUTED data
       - Tests: "Is my OOS performance real or luck?"
       - Null hypothesis: OOS performance is due to random chance
       - More lenient (easier to pass) because training is on real patterns

    2. permutation_type='full':
       - Permute BOTH training and test data
       - Tests: "Is the entire strategy edge noise?"
       - Null hypothesis: Entire strategy edge is due to data structure
       - More conservative (harder to pass) because even training is randomized

    Process per fold:
    1. Split data into train/test periods
    2. For real data: run strategy on test period, record metric
    3. Loop n_permutations:
       a. Permute according to permutation_type
       b. Run strategy on (possibly permuted) test data
       c. Record test metric
    4. Compute p-value for this fold
    5. Aggregate across all folds

    Args:
        options_df: Options data
        strategy_func: Function that runs backtest
        config: MCPT configuration
        permutation_type: 'test_only' or 'full'
        objective: Metric to optimize
        test_name: Name for this test
        folds: Pre-computed folds (optional)

    Returns:
        WalkForwardMCPTResult with aggregate and per-fold results
    """
    if test_name is None:
        dates = pd.to_datetime(options_df['date'])
        start_year = dates.min().year
        end_year = dates.max().year
        test_name = f"Walk-Forward ({start_year}-{end_year}) [{permutation_type}]"

    print(f"\n{'='*60}")
    print(f"RUNNING WALK-FORWARD MCPT: {test_name}")
    print(f"{'='*60}")
    print(f"Permutation type: {permutation_type}")
    print(f"Objective: {objective}")
    print(f"Permutations per fold: {config.n_permutations_walkforward}")
    print(f"Train window: {config.train_window_days} days")
    print(f"Test window: {config.test_window_days} days")

    # Generate folds if not provided
    if folds is None:
        folds = generate_walkforward_folds(
            options_df,
            train_window_days=config.train_window_days,
            test_window_days=config.test_window_days,
            step_days=config.retrain_days,
        )

    print(f"Number of folds: {len(folds)}")

    if len(folds) == 0:
        raise ValueError(
            f"Not enough data for walk-forward analysis. "
            f"Need at least {config.train_window_days + config.test_window_days} trading days."
        )

    # Process each fold
    fold_results = []
    all_real_metrics = []
    all_permuted_metrics = []

    for fold in folds:
        print(f"\n--- Fold {fold['fold_id']} ---")
        print(f"Train: {fold['train_start'].date()} to {fold['train_end'].date()}")
        print(f"Test:  {fold['test_start'].date()} to {fold['test_end'].date()}")

        # Split data
        train_data = options_df[
            (options_df['date'] >= fold['train_start']) &
            (options_df['date'] <= fold['train_end'])
        ].copy()

        test_data = options_df[
            (options_df['date'] >= fold['test_start']) &
            (options_df['date'] <= fold['test_end'])
        ].copy()

        # Combine for full dataset (strategy may need context)
        fold_data = pd.concat([train_data, test_data], ignore_index=True)

        # Run on real data
        real_results = strategy_func(fold_data)
        real_metric = calculate_objective(real_results, objective)
        all_real_metrics.append(real_metric)
        print(f"Real test {objective}: {real_metric:.4f}")

        # Run permutations
        permuted_metrics = []
        for i in range(config.n_permutations_walkforward):
            seed = config.random_seed + fold['fold_id'] * 10000 + i

            if permutation_type == 'test_only':
                # Permute only test data
                permuted_test = bar_permute(
                    test_data,
                    seed=seed,
                    preserve_warmup=0,  # No warmup needed for test-only
                )
                permuted_fold = pd.concat([train_data, permuted_test], ignore_index=True)
            else:  # 'full'
                # Permute entire dataset
                permuted_fold = bar_permute(
                    fold_data,
                    seed=seed,
                    preserve_warmup=config.warmup_days,
                )

            # Run strategy on permuted data
            try:
                perm_results = strategy_func(permuted_fold)
                perm_metric = calculate_objective(perm_results, objective)
                permuted_metrics.append(perm_metric)
            except Exception as e:
                print(f"  Permutation {i} failed: {e}")
                continue

        all_permuted_metrics.extend(permuted_metrics)

        # Compute fold p-value
        fold_p_value = compute_pvalue(real_metric, permuted_metrics)

        fold_result = WalkForwardFold(
            fold_id=fold['fold_id'],
            train_start=fold['train_start'],
            train_end=fold['train_end'],
            test_start=fold['test_start'],
            test_end=fold['test_end'],
            real_test_metric=real_metric,
            permuted_test_metrics=permuted_metrics,
            p_value=fold_p_value,
        )
        fold_results.append(fold_result)

        print(f"Fold p-value: {fold_p_value:.4f} ({fold_result.status})")

    # Aggregate results
    aggregate_real = np.mean(all_real_metrics)
    aggregate_p_value = compute_pvalue(aggregate_real, all_permuted_metrics)
    mean_permuted = float(np.mean(all_permuted_metrics))
    std_permuted = float(np.std(all_permuted_metrics))

    result = WalkForwardMCPTResult(
        test_name=test_name,
        permutation_type=permutation_type,
        real_test_metric=aggregate_real,
        metric_name=objective,
        permuted_test_metrics=all_permuted_metrics,
        p_value=aggregate_p_value,
        mean_permuted=mean_permuted,
        std_permuted=std_permuted,
        fold_results=fold_results,
        n_permutations=len(all_permuted_metrics),
        n_folds=len(fold_results),
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULT: {result.status}")
    print(f"{'='*60}")
    print(f"Real mean test {objective}: {aggregate_real:.4f}")
    print(f"Permuted mean:          {mean_permuted:.4f}")
    print(f"Permuted std:           {std_permuted:.4f}")
    print(f"Aggregate p-value:      {aggregate_p_value:.4f}")
    print(f"Fold pass rate:         {result.fold_pass_rate:.1f}%")

    if result.is_significant:
        print("\nINTERPRETATION: Strong evidence of genuine OOS alpha (p < 0.01)")
    elif result.is_marginal:
        print("\nINTERPRETATION: Moderate evidence of OOS alpha (p < 0.05)")
    else:
        print("\nINTERPRETATION: No evidence of OOS alpha (p >= 0.05)")
        print("Walk-forward performance could be due to random chance.")

    return result


def run_dual_walkforward_mcpt(
    options_df: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], Dict[str, Any]],
    config: MCPTConfig,
    objective: str = 'sharpe',
) -> Dict[str, WalkForwardMCPTResult]:
    """
    Run both test-only and full walk-forward MCPT.

    This provides two perspectives on strategy validity:
    1. test_only: Is OOS performance real? (more lenient)
    2. full: Is the entire edge real? (more conservative)

    A strategy that passes both has very strong evidence of genuine alpha.

    Args:
        options_df: Options data
        strategy_func: Strategy function
        config: MCPT configuration
        objective: Metric to optimize

    Returns:
        Dict with 'test_only' and 'full' results
    """
    print("\n" + "="*70)
    print("DUAL WALK-FORWARD MCPT")
    print("="*70)
    print("Running both test-only and full permutation modes...")

    # Generate folds once
    folds = generate_walkforward_folds(
        options_df,
        train_window_days=config.train_window_days,
        test_window_days=config.test_window_days,
        step_days=config.retrain_days,
    )

    results = {}

    # Test-only mode
    print("\n" + "-"*50)
    print("MODE 1: Test-Only Permutation")
    print("-"*50)
    results['test_only'] = run_walkforward_mcpt(
        options_df=options_df,
        strategy_func=strategy_func,
        config=config,
        permutation_type='test_only',
        objective=objective,
        folds=folds,
    )

    # Full mode
    print("\n" + "-"*50)
    print("MODE 2: Full Permutation")
    print("-"*50)
    results['full'] = run_walkforward_mcpt(
        options_df=options_df,
        strategy_func=strategy_func,
        config=config,
        permutation_type='full',
        objective=objective,
        folds=folds,
    )

    # Summary
    print("\n" + "="*70)
    print("DUAL MCPT SUMMARY")
    print("="*70)
    print(f"{'Mode':<20} | {'Real':>10} | {'Perm Mean':>10} | {'p-value':>8} | {'Status':>10}")
    print("-"*70)

    for mode, result in results.items():
        print(
            f"{mode:<20} | {result.real_test_metric:>10.3f} | "
            f"{result.mean_permuted:>10.3f} | {result.p_value:>8.4f} | {result.status:>10}"
        )

    print("="*70)

    # Interpretation
    if results['test_only'].is_significant and results['full'].is_significant:
        print("\nBOTH MODES PASS: Very strong evidence of genuine alpha")
    elif results['test_only'].is_significant:
        print("\nTEST-ONLY PASSES: OOS performance is likely real")
        print("(Full mode failure suggests strategy depends on time-series structure)")
    elif results['full'].is_significant:
        print("\nFULL MODE PASSES: Unusual - investigate further")
    else:
        print("\nBOTH MODES FAIL: Weak or no evidence of alpha")

    return results
