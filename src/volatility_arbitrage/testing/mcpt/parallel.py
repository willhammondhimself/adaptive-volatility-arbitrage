"""
Parallel execution utilities for MCPT.

Provides multiprocessing support for running many permutation backtests
in parallel, dramatically reducing total execution time.
"""

from typing import Callable, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import pandas as pd

from volatility_arbitrage.testing.mcpt.permutation import bar_permute
from volatility_arbitrage.testing.mcpt.utils import calculate_objective


def _run_single_permutation(
    options_df: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], dict],
    seed: int,
    warmup_days: int,
    objective: str,
) -> float:
    """
    Worker function for single permutation run.

    This function is called in a separate process for each permutation.

    Args:
        options_df: Original options data
        strategy_func: Function that runs backtest and returns results dict
        seed: Random seed for this permutation
        warmup_days: Days to preserve at start
        objective: Objective metric to extract

    Returns:
        Float value of the objective metric for this permutation
    """
    # Permute the data
    permuted_df = bar_permute(options_df, seed=seed, preserve_warmup=warmup_days)

    # Run the strategy
    results = strategy_func(permuted_df)

    # Extract the objective
    return calculate_objective(results, objective)


def _run_single_permutation_wrapper(args: Tuple) -> float:
    """
    Wrapper for _run_single_permutation that unpacks tuple args.

    Required for ProcessPoolExecutor.map() which can only pass single argument.
    """
    return _run_single_permutation(*args)


def run_parallel_permutations(
    options_df: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], dict],
    n_permutations: int,
    n_jobs: int = -1,
    warmup_days: int = 80,
    base_seed: int = 42,
    objective: str = 'sharpe',
    show_progress: bool = True,
) -> List[float]:
    """
    Run N permutation backtests in parallel.

    Uses ProcessPoolExecutor for true parallelism (bypasses GIL).
    Each worker gets a unique seed for reproducibility.

    Args:
        options_df: Original options data
        strategy_func: Function that runs backtest and returns results dict
        n_permutations: Number of permutations to run
        n_jobs: Number of parallel workers (-1 = all CPUs)
        warmup_days: Days to preserve at start of each permutation
        base_seed: Base random seed (each worker gets base_seed + i)
        objective: Objective metric to extract from each run
        show_progress: Whether to print progress updates

    Returns:
        List of objective values from each permutation
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # Limit workers to available CPUs
    n_jobs = min(n_jobs, mp.cpu_count())

    # Generate unique seeds for each permutation
    seeds = [base_seed + i for i in range(n_permutations)]

    # Prepare work items as tuples
    work_items = [
        (options_df, strategy_func, seed, warmup_days, objective)
        for seed in seeds
    ]

    results = []

    if show_progress:
        print(f"Running {n_permutations} permutations with {n_jobs} workers...")

    try:
        # Try using tqdm for progress bar if available
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Run in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        if use_tqdm and show_progress:
            # Use tqdm for nice progress bar
            futures = [
                executor.submit(_run_single_permutation_wrapper, item)
                for item in work_items
            ]
            for future in tqdm(as_completed(futures), total=n_permutations, desc="Permutations"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Permutation failed: {e}")
                    results.append(np.nan)
        else:
            # Fall back to simple map
            results = list(executor.map(_run_single_permutation_wrapper, work_items))

            if show_progress:
                print(f"Completed {len(results)} permutations")

    # Filter out any failed runs (NaN values)
    valid_results = [r for r in results if not np.isnan(r)]

    if len(valid_results) < len(results):
        print(f"Warning: {len(results) - len(valid_results)} permutations failed")

    return valid_results


def run_sequential_permutations(
    options_df: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], dict],
    n_permutations: int,
    warmup_days: int = 80,
    base_seed: int = 42,
    objective: str = 'sharpe',
    show_progress: bool = True,
) -> List[float]:
    """
    Run permutations sequentially (for debugging or when parallelization fails).

    This is slower but useful for:
    - Debugging issues with permutation logic
    - When multiprocessing doesn't work (e.g., in some notebook environments)
    - When strategy_func uses non-picklable objects

    Args:
        Same as run_parallel_permutations

    Returns:
        List of objective values from each permutation
    """
    results = []

    if show_progress:
        print(f"Running {n_permutations} permutations sequentially...")

    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_permutations), desc="Permutations")
    except ImportError:
        iterator = range(n_permutations)

    for i in iterator:
        seed = base_seed + i
        try:
            result = _run_single_permutation(
                options_df, strategy_func, seed, warmup_days, objective
            )
            results.append(result)
        except Exception as e:
            print(f"Permutation {i} failed: {e}")
            results.append(np.nan)

    if show_progress:
        print(f"Completed {len(results)} permutations")

    return [r for r in results if not np.isnan(r)]


def estimate_runtime(
    options_df: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame], dict],
    n_permutations: int,
    n_jobs: int = -1,
    warmup_days: int = 80,
    base_seed: int = 42,
    objective: str = 'sharpe',
    n_sample: int = 3,
) -> dict:
    """
    Estimate total runtime by running a few sample permutations.

    Args:
        Same as run_parallel_permutations, plus:
        n_sample: Number of sample runs for timing estimate

    Returns:
        Dict with timing estimates
    """
    import time

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # Run a few permutations to estimate time
    times = []
    for i in range(n_sample):
        start = time.time()
        _run_single_permutation(
            options_df, strategy_func, base_seed + i, warmup_days, objective
        )
        times.append(time.time() - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    # Estimate parallel runtime
    batches = n_permutations / n_jobs
    estimated_parallel = avg_time * batches

    # Estimate sequential runtime
    estimated_sequential = avg_time * n_permutations

    return {
        'avg_single_run_seconds': avg_time,
        'std_single_run_seconds': std_time,
        'n_workers': n_jobs,
        'estimated_parallel_minutes': estimated_parallel / 60,
        'estimated_sequential_minutes': estimated_sequential / 60,
        'speedup_factor': n_jobs,
    }
