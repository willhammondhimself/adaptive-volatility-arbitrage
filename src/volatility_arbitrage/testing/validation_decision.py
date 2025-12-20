"""
Decision tree logic for validation test interpretation.

Provides functions to assess MCPT, walk-forward, and bootstrap results
and determine overall pass/fail status with specific thresholds.
"""

from typing import Tuple


def assess_mcpt_result(pvalue: float, significance_level: float = 0.01) -> str:
    """
    Assess MCPT p-value and return status.

    Args:
        pvalue: MCPT p-value from permutation test
        significance_level: Threshold for strong evidence (default 0.01)

    Returns:
        Status string: PASS, MARGINAL, or FAIL
    """
    if pvalue < significance_level:
        return 'PASS'
    elif pvalue < 0.05:
        return 'MARGINAL'
    else:
        return 'FAIL'


def assess_walkforward_result(
    efficiency_ratio: float,
    folds_passed: int,
    total_folds: int,
    efficiency_threshold: float = 0.5,
) -> str:
    """
    Assess walk-forward validation results.

    Args:
        efficiency_ratio: Average test_sharpe / train_sharpe
        folds_passed: Number of folds with profitable test period
        total_folds: Total number of folds
        efficiency_threshold: Minimum acceptable efficiency

    Returns:
        Status string: PASS, MARGINAL, or FAIL
    """
    fold_pass_ratio = folds_passed / total_folds if total_folds > 0 else 0

    # PASS: High efficiency and most folds profitable
    if efficiency_ratio >= efficiency_threshold and fold_pass_ratio >= 0.6:
        return 'PASS'
    # MARGINAL: Moderate efficiency or some folds fail
    elif efficiency_ratio >= 0.3 or fold_pass_ratio >= 0.4:
        return 'MARGINAL'
    # FAIL: Low efficiency or most folds fail
    else:
        return 'FAIL'


def assess_bootstrap_result(
    sharpe_ci: Tuple[float, float],
    nw_sharpe_ci: Tuple[float, float],
) -> str:
    """
    Assess block bootstrap confidence intervals.

    Args:
        sharpe_ci: 95% CI for standard Sharpe ratio (lower, upper)
        nw_sharpe_ci: 95% CI for Newey-West Sharpe ratio (lower, upper)

    Returns:
        Status string: PASS, MARGINAL, or FAIL
    """
    sharpe_lower, sharpe_upper = sharpe_ci
    nw_sharpe_lower, nw_sharpe_upper = nw_sharpe_ci

    # PASS: Both standard and NW Sharpe CIs are positive
    if sharpe_lower > 0 and nw_sharpe_lower > 0:
        return 'PASS'
    # MARGINAL: Standard Sharpe positive but NW includes zero
    elif sharpe_lower > 0:
        return 'MARGINAL'
    # FAIL: Even standard Sharpe CI includes zero or negative
    else:
        return 'FAIL'


def determine_overall_status(
    mcpt_status: str,
    wf_status: str,
    bootstrap_status: str,
) -> str:
    """
    Determine overall validation status from individual test results.

    Args:
        mcpt_status: MCPT test status
        wf_status: Walk-forward test status
        bootstrap_status: Bootstrap test status

    Returns:
        Overall status: STRONG, MARGINAL, or FAIL
    """
    statuses = [mcpt_status, wf_status, bootstrap_status]

    # STRONG: All tests PASS
    if all(s == 'PASS' for s in statuses):
        return 'STRONG'

    # FAIL: Any test fails completely
    if any(s == 'FAIL' for s in statuses):
        # But if only one fails and others pass, it's MARGINAL
        fail_count = sum(1 for s in statuses if s == 'FAIL')
        pass_count = sum(1 for s in statuses if s == 'PASS')
        if fail_count == 1 and pass_count >= 1:
            return 'MARGINAL'
        return 'FAIL'

    # MARGINAL: Mix of PASS and MARGINAL
    return 'MARGINAL'


def calculate_hac_adjustment_estimate(nw_sharpe: float, standard_sharpe: float) -> float:
    """
    Estimate HAC adjustment factor from Sharpe ratios.

    The ratio of standard to NW Sharpe approximates the underestimation
    of volatility due to autocorrelation.

    Args:
        nw_sharpe: Newey-West adjusted Sharpe ratio
        standard_sharpe: Standard Sharpe ratio

    Returns:
        Estimated HAC volatility adjustment factor (typically 1.0-2.5)
    """
    if nw_sharpe <= 0 or standard_sharpe <= 0:
        # If either is non-positive, can't reliably estimate
        return 1.5  # Conservative default

    # HAC adjustment ≈ (σ_standard / σ_NW) ≈ (Sharpe_standard / Sharpe_NW)
    # Because Sharpe = μ/σ, so σ_NW/σ_standard = Sharpe_standard/Sharpe_NW
    adjustment = standard_sharpe / nw_sharpe

    # Bound to reasonable range [1.0, 3.0]
    adjustment = max(1.0, min(adjustment, 3.0))

    return adjustment
