"""
Monte Carlo Trade Resampling for Strategy Validation.

Bootstraps trade-level returns to estimate confidence intervals on:
- Total return
- Sharpe ratio
- Maximum drawdown

This helps answer: "What's the range of outcomes I could expect?"
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    n_trades: int

    # Total Return
    return_mean: float
    return_std: float
    return_ci_lower: float
    return_ci_upper: float
    return_median: float

    # Sharpe Ratio
    sharpe_mean: float
    sharpe_std: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    sharpe_median: float

    # Max Drawdown (negative values)
    dd_mean: float
    dd_std: float
    dd_ci_lower: float  # Worst case (more negative)
    dd_ci_upper: float  # Best case (less negative)
    dd_median: float

    # Win Rate
    win_rate_mean: float
    win_rate_std: float

    # Raw distributions for plotting
    return_distribution: np.ndarray
    sharpe_distribution: np.ndarray
    dd_distribution: np.ndarray


def extract_trade_returns(trades: List[Dict]) -> np.ndarray:
    """
    Extract position returns from trade log.

    Args:
        trades: List of trade dictionaries from backtest

    Returns:
        Array of trade returns as decimals (not percentages)
    """
    returns = []
    for t in trades:
        if t.get('action') == 'EXIT':
            # position_return is stored as percentage, convert to decimal
            ret = t.get('position_return', 0) / 100.0
            returns.append(ret)
    return np.array(returns)


def calculate_equity_curve(returns: np.ndarray, initial_capital: float = 100000) -> np.ndarray:
    """Calculate equity curve from sequence of trade returns."""
    equity = initial_capital
    curve = [equity]
    for ret in returns:
        equity *= (1 + ret)
        curve.append(equity)
    return np.array(curve)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown from equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)  # Returns negative value


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.05) -> float:
    """
    Calculate annualized Sharpe ratio from trade returns.

    Assumes average holding period of ~20 trading days.
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0

    # Annualization factor based on avg trades per year
    # With ~10 trades/year, each trade spans ~25 trading days
    trades_per_year = 252 / 25  # ~10 trades/year

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # Annualized values
    annual_return = mean_return * trades_per_year
    annual_std = std_return * np.sqrt(trades_per_year)

    sharpe = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0
    return sharpe


def winsorize_returns(returns: np.ndarray, percentile: float = 95) -> np.ndarray:
    """
    Winsorize returns to cap outliers at specified percentile.

    Args:
        returns: Array of trade returns
        percentile: Cap outliers at this percentile (default 95th)

    Returns:
        Winsorized returns array
    """
    lower = np.percentile(returns, 100 - percentile)
    upper = np.percentile(returns, percentile)
    return np.clip(returns, lower, upper)


def block_bootstrap_resample(
    trade_returns: np.ndarray,
    n_simulations: int = 10000,
    block_size: int = 3,
    initial_capital: float = 100000,
    random_seed: Optional[int] = None,
    winsorize_pct: Optional[float] = None,
) -> MonteCarloResult:
    """
    Block bootstrap resample to preserve serial correlation in trades.

    Instead of resampling individual trades, resample in blocks to maintain
    any time-series structure (e.g., consecutive winning/losing streaks).

    Args:
        trade_returns: Array of trade returns as decimals
        n_simulations: Number of bootstrap samples
        block_size: Number of consecutive trades per block (default 3)
        initial_capital: Starting capital for equity calculations
        random_seed: Optional seed for reproducibility
        winsorize_pct: If set, winsorize returns at this percentile to cap outliers

    Returns:
        MonteCarloResult with distributions and confidence intervals
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Optionally winsorize to reduce outlier impact
    if winsorize_pct is not None:
        trade_returns = winsorize_returns(trade_returns, winsorize_pct)

    n_trades = len(trade_returns)
    if n_trades < 5:
        raise ValueError(f"Need at least 5 trades for Monte Carlo, got {n_trades}")

    # Adjust block size if needed
    effective_block_size = min(block_size, n_trades // 2)
    n_blocks = n_trades - effective_block_size + 1  # Number of possible blocks
    blocks_needed = int(np.ceil(n_trades / effective_block_size))

    # Storage for simulation results
    total_returns = np.zeros(n_simulations)
    sharpes = np.zeros(n_simulations)
    max_dds = np.zeros(n_simulations)
    win_rates = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Sample block starting indices
        block_starts = np.random.choice(n_blocks, size=blocks_needed, replace=True)

        # Build resampled sequence from blocks
        sample_returns = []
        for start in block_starts:
            end = min(start + effective_block_size, n_trades)
            sample_returns.extend(trade_returns[start:end])

        # Trim to original length
        sample_returns = np.array(sample_returns[:n_trades])

        # Calculate metrics for this sample
        equity = calculate_equity_curve(sample_returns, initial_capital)
        total_returns[i] = (equity[-1] - initial_capital) / initial_capital * 100
        sharpes[i] = calculate_sharpe(sample_returns)
        max_dds[i] = calculate_max_drawdown(equity) * 100
        win_rates[i] = np.sum(sample_returns > 0) / len(sample_returns) * 100

    # Calculate statistics
    return MonteCarloResult(
        n_simulations=n_simulations,
        n_trades=n_trades,

        return_mean=np.mean(total_returns),
        return_std=np.std(total_returns),
        return_ci_lower=np.percentile(total_returns, 2.5),
        return_ci_upper=np.percentile(total_returns, 97.5),
        return_median=np.median(total_returns),

        sharpe_mean=np.mean(sharpes),
        sharpe_std=np.std(sharpes),
        sharpe_ci_lower=np.percentile(sharpes, 2.5),
        sharpe_ci_upper=np.percentile(sharpes, 97.5),
        sharpe_median=np.median(sharpes),

        dd_mean=np.mean(max_dds),
        dd_std=np.std(max_dds),
        dd_ci_lower=np.percentile(max_dds, 2.5),
        dd_ci_upper=np.percentile(max_dds, 97.5),
        dd_median=np.median(max_dds),

        win_rate_mean=np.mean(win_rates),
        win_rate_std=np.std(win_rates),

        return_distribution=total_returns,
        sharpe_distribution=sharpes,
        dd_distribution=max_dds,
    )


def bootstrap_resample(
    trade_returns: np.ndarray,
    n_simulations: int = 10000,
    initial_capital: float = 100000,
    random_seed: Optional[int] = None,
    winsorize_pct: Optional[float] = None,
) -> MonteCarloResult:
    """
    Bootstrap resample trade returns to estimate confidence intervals.

    Args:
        trade_returns: Array of trade returns as decimals
        n_simulations: Number of bootstrap samples
        initial_capital: Starting capital for equity calculations
        random_seed: Optional seed for reproducibility
        winsorize_pct: If set, winsorize returns at this percentile to cap outliers

    Returns:
        MonteCarloResult with distributions and confidence intervals
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Optionally winsorize to reduce outlier impact
    if winsorize_pct is not None:
        trade_returns = winsorize_returns(trade_returns, winsorize_pct)

    n_trades = len(trade_returns)
    if n_trades < 5:
        raise ValueError(f"Need at least 5 trades for Monte Carlo, got {n_trades}")

    # Storage for simulation results
    total_returns = np.zeros(n_simulations)
    sharpes = np.zeros(n_simulations)
    max_dds = np.zeros(n_simulations)
    win_rates = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Resample with replacement (same number of trades)
        sample_idx = np.random.choice(n_trades, size=n_trades, replace=True)
        sample_returns = trade_returns[sample_idx]

        # Calculate metrics for this sample
        equity = calculate_equity_curve(sample_returns, initial_capital)
        total_returns[i] = (equity[-1] - initial_capital) / initial_capital * 100
        sharpes[i] = calculate_sharpe(sample_returns)
        max_dds[i] = calculate_max_drawdown(equity) * 100
        win_rates[i] = np.sum(sample_returns > 0) / len(sample_returns) * 100

    # Calculate statistics
    return MonteCarloResult(
        n_simulations=n_simulations,
        n_trades=n_trades,

        return_mean=np.mean(total_returns),
        return_std=np.std(total_returns),
        return_ci_lower=np.percentile(total_returns, 2.5),
        return_ci_upper=np.percentile(total_returns, 97.5),
        return_median=np.median(total_returns),

        sharpe_mean=np.mean(sharpes),
        sharpe_std=np.std(sharpes),
        sharpe_ci_lower=np.percentile(sharpes, 2.5),
        sharpe_ci_upper=np.percentile(sharpes, 97.5),
        sharpe_median=np.median(sharpes),

        dd_mean=np.mean(max_dds),
        dd_std=np.std(max_dds),
        dd_ci_lower=np.percentile(max_dds, 2.5),  # Worst case
        dd_ci_upper=np.percentile(max_dds, 97.5),  # Best case
        dd_median=np.median(max_dds),

        win_rate_mean=np.mean(win_rates),
        win_rate_std=np.std(win_rates),

        return_distribution=total_returns,
        sharpe_distribution=sharpes,
        dd_distribution=max_dds,
    )


def print_monte_carlo_report(result: MonteCarloResult, actual_metrics: Optional[Dict] = None):
    """Print formatted Monte Carlo results."""
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION RESULTS")
    print("="*60)
    print(f"\nSimulations: {result.n_simulations:,}")
    print(f"Trades resampled: {result.n_trades}")

    print("\n" + "-"*60)
    print("TOTAL RETURN")
    print("-"*60)
    print(f"  Mean:     {result.return_mean:>8.1f}%")
    print(f"  Median:   {result.return_median:>8.1f}%")
    print(f"  Std Dev:  {result.return_std:>8.1f}%")
    print(f"  95% CI:   [{result.return_ci_lower:>6.1f}%, {result.return_ci_upper:>6.1f}%]")
    if actual_metrics:
        actual = actual_metrics.get('total_return', 0)
        percentile = np.sum(result.return_distribution <= actual) / result.n_simulations * 100
        print(f"  Actual:   {actual:>8.1f}% (percentile: {percentile:.0f}%)")

    print("\n" + "-"*60)
    print("SHARPE RATIO")
    print("-"*60)
    print(f"  Mean:     {result.sharpe_mean:>8.2f}")
    print(f"  Median:   {result.sharpe_median:>8.2f}")
    print(f"  Std Dev:  {result.sharpe_std:>8.2f}")
    print(f"  95% CI:   [{result.sharpe_ci_lower:>6.2f}, {result.sharpe_ci_upper:>6.2f}]")
    if actual_metrics:
        actual = actual_metrics.get('sharpe', 0)
        percentile = np.sum(result.sharpe_distribution <= actual) / result.n_simulations * 100
        print(f"  Actual:   {actual:>8.2f} (percentile: {percentile:.0f}%)")

    print("\n" + "-"*60)
    print("MAXIMUM DRAWDOWN")
    print("-"*60)
    print(f"  Mean:     {result.dd_mean:>8.1f}%")
    print(f"  Median:   {result.dd_median:>8.1f}%")
    print(f"  Std Dev:  {result.dd_std:>8.1f}%")
    print(f"  95% CI:   [{result.dd_ci_lower:>6.1f}%, {result.dd_ci_upper:>6.1f}%]")
    if actual_metrics:
        actual = actual_metrics.get('max_drawdown', 0)
        print(f"  Actual:   {actual:>8.1f}%")

    print("\n" + "-"*60)
    print("WIN RATE")
    print("-"*60)
    print(f"  Mean:     {result.win_rate_mean:>8.1f}%")
    print(f"  Std Dev:  {result.win_rate_std:>8.1f}%")

    # Risk assessment
    print("\n" + "-"*60)
    print("RISK ASSESSMENT")
    print("-"*60)

    # Probability of loss
    prob_loss = np.sum(result.return_distribution < 0) / result.n_simulations * 100
    print(f"  P(Total Loss): {prob_loss:.1f}%")

    # Probability of Sharpe < 0.5
    prob_low_sharpe = np.sum(result.sharpe_distribution < 0.5) / result.n_simulations * 100
    print(f"  P(Sharpe < 0.5): {prob_low_sharpe:.1f}%")

    # Probability of DD worse than 20%
    prob_bad_dd = np.sum(result.dd_distribution < -20) / result.n_simulations * 100
    print(f"  P(DD > 20%): {prob_bad_dd:.1f}%")

    # Overall confidence
    print("\n" + "-"*60)
    print("VALIDATION")
    print("-"*60)

    if result.sharpe_ci_lower > 0.5:
        print("  ✅ 95% CI for Sharpe > 0.5 - Strategy is robust")
    elif result.sharpe_ci_lower > 0:
        print("  ⚠️  95% CI for Sharpe > 0 but < 0.5 - Marginal")
    else:
        print("  ❌ 95% CI for Sharpe includes negative - Not robust")

    if prob_loss < 5:
        print("  ✅ <5% probability of total loss")
    else:
        print(f"  ⚠️  {prob_loss:.1f}% probability of total loss")


def plot_monte_carlo_distributions(
    result: MonteCarloResult,
    actual_metrics: Optional[Dict] = None,
    save_path: Optional[str] = None,
):
    """
    Plot Monte Carlo distributions for returns, Sharpe, and drawdown.

    Args:
        result: MonteCarloResult from bootstrap_resample
        actual_metrics: Dict with 'total_return', 'sharpe', 'max_drawdown'
        save_path: Optional path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Total Return Distribution
    ax1 = axes[0]
    ax1.hist(result.return_distribution, bins=50, density=True, alpha=0.7,
             color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.axvline(result.return_ci_lower, color='red', linestyle='--',
                label=f'95% CI: [{result.return_ci_lower:.0f}%, {result.return_ci_upper:.0f}%]')
    ax1.axvline(result.return_ci_upper, color='red', linestyle='--')
    ax1.axvline(result.return_mean, color='darkblue', linestyle='-', linewidth=2,
                label=f'Mean: {result.return_mean:.0f}%')
    if actual_metrics:
        actual = actual_metrics.get('total_return', 0)
        ax1.axvline(actual, color='green', linestyle='-', linewidth=2,
                    label=f'Actual: {actual:.0f}%')
    ax1.set_xlabel('Total Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Total Return Distribution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Sharpe Ratio Distribution
    ax2 = axes[1]
    ax2.hist(result.sharpe_distribution, bins=50, density=True, alpha=0.7,
             color='darkgreen', edgecolor='black', linewidth=0.5)
    ax2.axvline(result.sharpe_ci_lower, color='red', linestyle='--',
                label=f'95% CI: [{result.sharpe_ci_lower:.2f}, {result.sharpe_ci_upper:.2f}]')
    ax2.axvline(result.sharpe_ci_upper, color='red', linestyle='--')
    ax2.axvline(result.sharpe_mean, color='darkgreen', linestyle='-', linewidth=2,
                label=f'Mean: {result.sharpe_mean:.2f}')
    ax2.axvline(0.5, color='orange', linestyle=':', linewidth=2,
                label='Threshold: 0.5')
    if actual_metrics:
        actual = actual_metrics.get('sharpe', 0)
        ax2.axvline(actual, color='blue', linestyle='-', linewidth=2,
                    label=f'Actual: {actual:.2f}')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_ylabel('Density')
    ax2.set_title('Sharpe Ratio Distribution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Max Drawdown Distribution
    ax3 = axes[2]
    ax3.hist(result.dd_distribution, bins=50, density=True, alpha=0.7,
             color='darkred', edgecolor='black', linewidth=0.5)
    ax3.axvline(result.dd_ci_lower, color='red', linestyle='--',
                label=f'95% CI: [{result.dd_ci_lower:.1f}%, {result.dd_ci_upper:.1f}%]')
    ax3.axvline(result.dd_ci_upper, color='red', linestyle='--')
    ax3.axvline(result.dd_mean, color='darkred', linestyle='-', linewidth=2,
                label=f'Mean: {result.dd_mean:.1f}%')
    ax3.axvline(-20, color='orange', linestyle=':', linewidth=2,
                label='Warning: -20%')
    if actual_metrics:
        actual = actual_metrics.get('max_drawdown', 0)
        ax3.axvline(actual, color='blue', linestyle='-', linewidth=2,
                    label=f'Actual: {actual:.1f}%')
    ax3.set_xlabel('Maximum Drawdown (%)')
    ax3.set_ylabel('Density')
    ax3.set_title('Max Drawdown Distribution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f'Monte Carlo Simulation ({result.n_simulations:,} samples, {result.n_trades} trades)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()
