"""
Utility functions for MCPT.

Includes objective function extraction, plotting, and report generation.
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def calculate_objective(results: Dict[str, Any], objective: str) -> float:
    """
    Extract objective metric from backtest results.

    Args:
        results: Dictionary returned by run_qv_backtest() or similar
        objective: Name of objective ('sharpe', 'nw_sharpe', 'profit_factor', etc.)

    Returns:
        Float value of the objective metric

    Raises:
        ValueError: If objective is not recognized
    """
    if objective == 'sharpe':
        return float(results.get('sharpe', 0.0))
    elif objective == 'nw_sharpe':
        return float(results.get('nw_sharpe', results.get('sharpe', 0.0)))
    elif objective == 'profit_factor':
        return float(results.get('profit_factor', 1.0))
    elif objective == 'win_rate':
        return float(results.get('win_rate', 0.5))
    elif objective == 'total_return':
        return float(results.get('total_return', 0.0))
    elif objective == 'max_drawdown':
        # Note: max_drawdown is typically negative, but we want higher = better
        # So we return the absolute value for comparison purposes
        return -abs(float(results.get('max_drawdown', 0.0)))
    elif objective == 'calmar':
        return float(results.get('calmar', 0.0))
    elif objective == 'sortino':
        return float(results.get('sortino', 0.0))
    else:
        # Try to get the objective directly from results
        if objective in results:
            return float(results[objective])
        raise ValueError(f"Unknown objective: {objective}")


def compute_pvalue(
    real_metric: float,
    permuted_metrics: List[float],
    higher_is_better: bool = True,
) -> float:
    """
    Compute p-value from real metric and permuted distribution.

    Uses the standard permutation test p-value formula with continuity correction:
    p = (count(permuted >= real) + 1) / (n_permutations + 1)

    The "+1" in both numerator and denominator is the continuity correction
    that includes the real value itself in the null distribution.

    Args:
        real_metric: The metric from the real (non-permuted) data
        permuted_metrics: List of metrics from permuted data runs
        higher_is_better: If True, count permuted >= real; else count permuted <= real

    Returns:
        P-value in range [1/(n+1), 1.0]
    """
    n = len(permuted_metrics)
    if n == 0:
        return 1.0

    permuted_array = np.array(permuted_metrics)

    if higher_is_better:
        # Count how many permuted values are >= real value
        count_extreme = np.sum(permuted_array >= real_metric)
    else:
        # Count how many permuted values are <= real value
        count_extreme = np.sum(permuted_array <= real_metric)

    # Continuity correction: add 1 to both numerator and denominator
    p_value = (count_extreme + 1) / (n + 1)

    return p_value


def plot_mcpt_distribution(
    real_metric: float,
    permuted_metrics: List[float],
    metric_name: str = "Sharpe Ratio",
    p_value: Optional[float] = None,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Any:
    """
    Plot permuted distribution with real value marked.

    Creates a histogram of permuted metrics with a vertical line showing
    where the real metric falls in the distribution.

    Args:
        real_metric: The metric from real (non-permuted) data
        permuted_metrics: List of metrics from permuted data
        metric_name: Name of the metric for labeling
        p_value: Pre-computed p-value to display (optional)
        save_path: Path to save figure (optional)
        title: Custom title (optional)

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Cannot generate plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute statistics
    permuted_array = np.array(permuted_metrics)
    mean_permuted = np.mean(permuted_array)
    std_permuted = np.std(permuted_array)

    # Histogram of permuted metrics
    ax.hist(
        permuted_array,
        bins=50,
        alpha=0.7,
        color='steelblue',
        edgecolor='white',
        label=f'Permuted Distribution (n={len(permuted_metrics)})'
    )

    # Vertical line for real metric
    ax.axvline(
        real_metric,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Real {metric_name}: {real_metric:.3f}'
    )

    # Vertical line for mean
    ax.axvline(
        mean_permuted,
        color='gray',
        linestyle=':',
        linewidth=1.5,
        label=f'Permuted Mean: {mean_permuted:.3f}'
    )

    # Add shaded region for ±1 std
    ax.axvspan(
        mean_permuted - std_permuted,
        mean_permuted + std_permuted,
        alpha=0.2,
        color='gray',
        label=f'±1 Std: {std_permuted:.3f}'
    )

    # Compute p-value if not provided
    if p_value is None:
        p_value = compute_pvalue(real_metric, permuted_metrics)

    # Title with p-value
    if title is None:
        title = f"MCPT Distribution: {metric_name}"
    ax.set_title(f"{title}\np-value: {p_value:.4f}", fontsize=12)

    ax.set_xlabel(metric_name, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)

    # Add interpretation text
    if p_value < 0.01:
        interpretation = "STRONG EVIDENCE of alpha (p < 0.01)"
        color = 'green'
    elif p_value < 0.05:
        interpretation = "MARGINAL evidence (p < 0.05)"
        color = 'orange'
    else:
        interpretation = "NO EVIDENCE of alpha (p >= 0.05)"
        color = 'red'

    ax.text(
        0.02, 0.98, interpretation,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        color=color,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def generate_mcpt_report(
    results: List[Any],
    output_path: Optional[Path] = None,
    format: str = 'text',
) -> str:
    """
    Generate comprehensive MCPT validation report.

    Args:
        results: List of MCPTResult and/or WalkForwardMCPTResult objects
        output_path: Path to save report (optional)
        format: 'text' or 'html' (default 'text')

    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MCPT VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Header
    lines.append(f"{'Test':<35} | {'Real':>10} | {'Perm Mean':>10} | {'p-value':>8} | {'Status':>10}")
    lines.append("-" * 80)

    all_pass = True
    any_marginal = False

    for result in results:
        status = result.status
        if status == "FAIL":
            all_pass = False
        elif status == "MARGINAL":
            any_marginal = True

        lines.append(
            f"{result.test_name:<35} | "
            f"{result.real_metric:>10.3f} | "
            f"{result.mean_permuted:>10.3f} | "
            f"{result.p_value:>8.4f} | "
            f"{status:>10}"
        )

    lines.append("=" * 80)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    if all_pass and not any_marginal:
        lines.append("ALL TESTS PASSED - Strong evidence of genuine alpha")
    elif all_pass:
        lines.append("TESTS PASSED (some marginal) - Moderate evidence of alpha")
    else:
        lines.append("SOME TESTS FAILED - Evidence of alpha is weak or absent")

    lines.append("")
    lines.append("INTERPRETATION GUIDE:")
    lines.append("  p < 0.01  : Strong evidence (PASS)")
    lines.append("  p < 0.05  : Moderate evidence (MARGINAL)")
    lines.append("  p >= 0.05 : No evidence (FAIL)")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report


@dataclass
class MCPTSummary:
    """Summary statistics for MCPT results."""
    test_name: str
    n_permutations: int
    real_metric: float
    mean_permuted: float
    std_permuted: float
    min_permuted: float
    max_permuted: float
    percentile_real: float  # Where real falls in distribution
    p_value: float
    status: str

    @classmethod
    def from_results(
        cls,
        test_name: str,
        real_metric: float,
        permuted_metrics: List[float],
    ) -> "MCPTSummary":
        """Create summary from raw results."""
        permuted_array = np.array(permuted_metrics)
        p_value = compute_pvalue(real_metric, permuted_metrics)

        # Compute percentile of real metric
        percentile = np.sum(permuted_array <= real_metric) / len(permuted_array) * 100

        # Determine status
        if p_value < 0.01:
            status = "PASS"
        elif p_value < 0.05:
            status = "MARGINAL"
        else:
            status = "FAIL"

        return cls(
            test_name=test_name,
            n_permutations=len(permuted_metrics),
            real_metric=real_metric,
            mean_permuted=float(np.mean(permuted_array)),
            std_permuted=float(np.std(permuted_array)),
            min_permuted=float(np.min(permuted_array)),
            max_permuted=float(np.max(permuted_array)),
            percentile_real=percentile,
            p_value=p_value,
            status=status,
        )
