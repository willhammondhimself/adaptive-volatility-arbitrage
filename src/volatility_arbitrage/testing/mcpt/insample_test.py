"""
In-sample Monte Carlo Permutation Test.

Implements the core MCPT algorithm for validating that strategy performance
is statistically significant and not due to data mining or random chance.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional
import numpy as np
import pandas as pd

from volatility_arbitrage.testing.mcpt.config import MCPTConfig
from volatility_arbitrage.testing.mcpt.permutation import bar_permute
from volatility_arbitrage.testing.mcpt.utils import calculate_objective, compute_pvalue
from volatility_arbitrage.testing.mcpt.parallel import (
    run_parallel_permutations,
    run_sequential_permutations,
)


@dataclass
class MCPTResult:
    """
    Results from Monte Carlo Permutation Test.

    Contains the real metric, permuted distribution, p-value, and
    significance assessment.
    """
    test_name: str
    real_metric: float
    metric_name: str
    permuted_metrics: List[float]
    p_value: float
    mean_permuted: float
    std_permuted: float
    n_permutations: int

    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant (p < 0.01)."""
        return self.p_value < 0.01

    @property
    def is_marginal(self) -> bool:
        """Check if result is marginally significant (0.01 <= p < 0.05)."""
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
    def z_score(self) -> float:
        """Calculate z-score of real metric vs permuted distribution."""
        if self.std_permuted == 0:
            return 0.0
        return (self.real_metric - self.mean_permuted) / self.std_permuted

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'real_metric': self.real_metric,
            'metric_name': self.metric_name,
            'p_value': self.p_value,
            'mean_permuted': self.mean_permuted,
            'std_permuted': self.std_permuted,
            'n_permutations': self.n_permutations,
            'z_score': self.z_score,
            'status': self.status,
            'is_significant': self.is_significant,
            'is_marginal': self.is_marginal,
        }

    def __repr__(self) -> str:
        return (
            f"MCPTResult(test_name='{self.test_name}', "
            f"real={self.real_metric:.3f}, mean_perm={self.mean_permuted:.3f}, "
            f"p={self.p_value:.4f}, status={self.status})"
        )


def run_insample_mcpt(
    options_df: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], Dict[str, Any]],
    config: MCPTConfig,
    objective: str = 'sharpe',
    test_name: Optional[str] = None,
    use_parallel: bool = True,
) -> MCPTResult:
    """
    Run in-sample Monte Carlo Permutation Test.

    This is the core MCPT algorithm:
    1. Run strategy on REAL data -> record objective (e.g., Sharpe)
    2. Loop n_permutations times:
       a. Permute bar order (preserving warmup period)
       b. Run strategy on permuted data
       c. Record objective
    3. Compute p-value: p = (count(permuted >= real) + 1) / (n_permutations + 1)

    The p-value tells us the probability of achieving the observed performance
    (or better) under the null hypothesis that the time-series structure
    contains no exploitable information.

    Interpretation:
    - p < 0.01: Strong evidence of genuine alpha
    - p < 0.05: Moderate evidence (marginal)
    - p >= 0.05: No evidence - likely noise or overfit

    Args:
        options_df: DataFrame with options data. Must have 'date' column.
        strategy_func: Function that takes options_df and returns results dict
            with at least the objective metric (e.g., {'sharpe': 1.5, ...})
        config: MCPTConfig with test parameters
        objective: Name of objective to use ('sharpe', 'nw_sharpe', etc.)
        test_name: Name for this test (for reporting)
        use_parallel: If True, use multiprocessing for permutations

    Returns:
        MCPTResult with real metric, permuted distribution, and p-value

    Example:
        >>> config = MCPTConfig(n_permutations=1000)
        >>> def my_strategy(df):
        ...     return run_qv_backtest(df, strategy_config)
        >>> result = run_insample_mcpt(options_df, my_strategy, config)
        >>> print(f"p-value: {result.p_value:.4f}, status: {result.status}")
    """
    if test_name is None:
        # Generate test name from date range
        dates = pd.to_datetime(options_df['date'])
        start_year = dates.min().year
        end_year = dates.max().year
        test_name = f"In-Sample ({start_year}-{end_year})"

    print(f"\n{'='*60}")
    print(f"RUNNING IN-SAMPLE MCPT: {test_name}")
    print(f"{'='*60}")
    print(f"Objective: {objective}")
    print(f"Permutations: {config.n_permutations}")
    print(f"Warmup days: {config.warmup_days}")
    print(f"Parallel: {use_parallel}")

    # Step 1: Run strategy on real data
    print("\nStep 1: Running strategy on real data...")
    real_results = strategy_func(options_df)
    real_metric = calculate_objective(real_results, objective)
    print(f"Real {objective}: {real_metric:.4f}")

    # Step 2: Run permutations
    print(f"\nStep 2: Running {config.n_permutations} permutations...")

    if use_parallel:
        permuted_metrics = run_parallel_permutations(
            options_df=options_df,
            strategy_func=strategy_func,
            n_permutations=config.n_permutations,
            n_jobs=config.n_jobs,
            warmup_days=config.warmup_days,
            base_seed=config.random_seed,
            objective=objective,
            show_progress=True,
        )
    else:
        permuted_metrics = run_sequential_permutations(
            options_df=options_df,
            strategy_func=strategy_func,
            n_permutations=config.n_permutations,
            warmup_days=config.warmup_days,
            base_seed=config.random_seed,
            objective=objective,
            show_progress=True,
        )

    # Step 3: Compute statistics
    print("\nStep 3: Computing statistics...")
    p_value = compute_pvalue(real_metric, permuted_metrics)
    mean_permuted = float(np.mean(permuted_metrics))
    std_permuted = float(np.std(permuted_metrics))

    # Create result
    result = MCPTResult(
        test_name=test_name,
        real_metric=real_metric,
        metric_name=objective,
        permuted_metrics=permuted_metrics,
        p_value=p_value,
        mean_permuted=mean_permuted,
        std_permuted=std_permuted,
        n_permutations=len(permuted_metrics),
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULT: {result.status}")
    print(f"{'='*60}")
    print(f"Real {objective}:     {real_metric:.4f}")
    print(f"Permuted mean:   {mean_permuted:.4f}")
    print(f"Permuted std:    {std_permuted:.4f}")
    print(f"Z-score:         {result.z_score:.2f}")
    print(f"p-value:         {p_value:.4f}")

    if result.is_significant:
        print("\nINTERPRETATION: Strong evidence of genuine alpha (p < 0.01)")
        print("The strategy's performance is very unlikely to be due to random chance.")
    elif result.is_marginal:
        print("\nINTERPRETATION: Moderate evidence of alpha (p < 0.05)")
        print("Consider additional validation. Walk-forward MCPT recommended.")
    else:
        print("\nINTERPRETATION: No evidence of alpha (p >= 0.05)")
        print("The strategy's performance could easily be due to random chance.")

    return result


def run_insample_mcpt_multi_objective(
    options_df: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], Dict[str, Any]],
    config: MCPTConfig,
    objectives: Optional[List[str]] = None,
    test_name_prefix: str = "In-Sample",
    use_parallel: bool = True,
) -> Dict[str, MCPTResult]:
    """
    Run in-sample MCPT for multiple objectives.

    This is more efficient than running run_insample_mcpt multiple times
    because we only run each permutation once and extract all objectives.

    Args:
        options_df: Options data
        strategy_func: Strategy function
        config: MCPT configuration
        objectives: List of objectives to test (default from config)
        test_name_prefix: Prefix for test names
        use_parallel: Whether to use parallel execution

    Returns:
        Dict mapping objective name to MCPTResult
    """
    if objectives is None:
        objectives = config.objectives

    # Get date range for test name
    dates = pd.to_datetime(options_df['date'])
    start_year = dates.min().year
    end_year = dates.max().year

    print(f"\n{'='*60}")
    print(f"RUNNING MULTI-OBJECTIVE IN-SAMPLE MCPT")
    print(f"{'='*60}")
    print(f"Objectives: {objectives}")
    print(f"Permutations: {config.n_permutations}")

    # Run strategy on real data
    print("\nRunning strategy on real data...")
    real_results = strategy_func(options_df)

    # Run permutations (once, extracting all metrics)
    print(f"\nRunning {config.n_permutations} permutations...")

    # For multi-objective, we need to collect all results
    all_permuted_results = []

    def _run_and_collect(df):
        return strategy_func(df)

    if use_parallel:
        # Run parallel but collect full results
        # This is a bit wasteful but ensures consistency
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        n_jobs = config.n_jobs if config.n_jobs != -1 else mp.cpu_count()
        seeds = [config.random_seed + i for i in range(config.n_permutations)]

        def run_one(seed):
            permuted = bar_permute(options_df, seed=seed, preserve_warmup=config.warmup_days)
            return strategy_func(permuted)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            all_permuted_results = list(executor.map(run_one, seeds))
    else:
        for i in range(config.n_permutations):
            seed = config.random_seed + i
            permuted = bar_permute(options_df, seed=seed, preserve_warmup=config.warmup_days)
            all_permuted_results.append(strategy_func(permuted))

    # Extract results for each objective
    results = {}
    for obj in objectives:
        real_metric = calculate_objective(real_results, obj)
        permuted_metrics = [
            calculate_objective(r, obj) for r in all_permuted_results
        ]
        p_value = compute_pvalue(real_metric, permuted_metrics)

        test_name = f"{test_name_prefix} ({start_year}-{end_year}) - {obj}"

        results[obj] = MCPTResult(
            test_name=test_name,
            real_metric=real_metric,
            metric_name=obj,
            permuted_metrics=permuted_metrics,
            p_value=p_value,
            mean_permuted=float(np.mean(permuted_metrics)),
            std_permuted=float(np.std(permuted_metrics)),
            n_permutations=len(permuted_metrics),
        )

        print(f"\n{obj}: real={real_metric:.3f}, p={p_value:.4f}, status={results[obj].status}")

    return results
