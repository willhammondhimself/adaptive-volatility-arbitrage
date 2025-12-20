"""
Validation result dataclass for comprehensive strategy testing.

Aggregates results from MCPT, walk-forward validation, and block bootstrap
to provide unified pass/fail assessment and recommendations.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Tuple


@dataclass
class ValidationResult:
    """
    Comprehensive validation results from multiple statistical tests.

    Aggregates backtest metrics, MCPT p-values, walk-forward efficiency,
    and bootstrap confidence intervals into a unified assessment.
    """

    # Backtest metrics
    backtest_sharpe: Decimal
    backtest_nw_sharpe: Decimal
    total_return: Decimal
    max_drawdown: Decimal
    num_trades: int

    # MCPT results
    mcpt_pvalue: Optional[float]
    mcpt_status: str  # PASS/MARGINAL/FAIL/SKIPPED

    # Walk-forward results
    wf_avg_train_sharpe: float
    wf_avg_test_sharpe: float
    wf_avg_efficiency: float
    wf_folds_passed: int
    wf_total_folds: int
    wf_status: str  # PASS/MARGINAL/FAIL/SKIPPED

    # Bootstrap results
    bootstrap_sharpe_ci: Tuple[float, float]
    bootstrap_nw_sharpe_ci: Tuple[float, float]
    bootstrap_return_ci: Tuple[float, float]
    bootstrap_status: str  # PASS/MARGINAL/FAIL/SKIPPED

    # Overall assessment
    overall_status: str  # STRONG/MARGINAL/FAIL

    def is_pass(self) -> bool:
        """
        Returns True if alpha appears statistically significant.

        Criteria:
        - MCPT p-value < 0.05 (statistically significant)
        - Walk-forward efficiency > 0.5 (limited overfitting)
        - Bootstrap Sharpe 95% CI lower bound > 0 (positive with confidence)
        """
        mcpt_pass = self.mcpt_pvalue is not None and self.mcpt_pvalue < 0.05
        wf_pass = self.wf_avg_efficiency > 0.5
        bootstrap_pass = self.bootstrap_sharpe_ci[0] > 0

        return mcpt_pass and wf_pass and bootstrap_pass

    def is_strong(self) -> bool:
        """
        Returns True if alpha is strongly supported by all tests.

        Stricter criteria:
        - MCPT p-value < 0.01 (highly significant)
        - Walk-forward efficiency > 0.5
        - Bootstrap Sharpe 95% CI lower bound > 0
        - All tests must be run (not SKIPPED)
        """
        if self.mcpt_status == 'SKIPPED' or self.wf_status == 'SKIPPED' or self.bootstrap_status == 'SKIPPED':
            return False

        mcpt_strong = self.mcpt_pvalue is not None and self.mcpt_pvalue < 0.01
        wf_strong = self.wf_avg_efficiency > 0.5
        bootstrap_strong = self.bootstrap_sharpe_ci[0] > 0

        return mcpt_strong and wf_strong and bootstrap_strong

    def is_marginal(self) -> bool:
        """
        Returns True if alpha exists but is weak.

        Criteria:
        - At least one test passes
        - Not all tests pass (otherwise would be strong/pass)
        """
        passes = 0
        if self.mcpt_status == 'PASS':
            passes += 1
        if self.wf_status == 'PASS':
            passes += 1
        if self.bootstrap_status == 'PASS':
            passes += 1

        return passes > 0 and not self.is_pass()

    def get_recommendation(self) -> str:
        """
        Returns specific next steps based on validation results.

        Returns:
            Multi-line recommendation string with actionable steps
        """
        if self.is_strong():
            return """Alpha appears REAL and STATISTICALLY SIGNIFICANT.

RECOMMENDATION:
- Alpha is well-supported by all validation tests
- Negative NW-Sharpe indicates high autocorrelation in returns
- This is likely inherent to volatility arbitrage, not spurious alpha

NEXT STEPS:
1. Implement HAC-adjusted position sizing (30-50% size reduction)
   - Use Newey-West volatility in Kelly criterion
   - Tighten drawdown thresholds by HAC adjustment factor
2. Monitor autocorrelation in live trading
3. Use conservative position sizing (lower confidence bound)

RISK WARNING:
- True volatility is ~{hac_factor:.1%} higher than naive estimate
- Position sizes should reflect this higher risk
- Expected: Standard Sharpe ~0.8-0.9, NW Sharpe ~0.4-0.6
"""

        elif self.is_marginal():
            return """Alpha appears REAL but WEAK.

RECOMMENDATION:
- Some validation tests pass, others marginal or fail
- Statistical significance is borderline
- Proceed with EXTREME CAUTION

NEXT STEPS:
1. Reduce position sizing by 50% immediately
2. Increase entry thresholds (consensus_threshold 0.15 → 0.25)
3. Add regime-based filters (avoid crisis/elevated regimes)
4. Test on additional out-of-sample data (2024+)
5. Consider strategy modifications to reduce autocorrelation

RISK WARNING:
- Alpha may be partially spurious
- Out-of-sample performance likely to degrade
- Do not increase position sizes without further validation
"""

        else:
            return """Alpha appears SPURIOUS or STATISTICALLY INSIGNIFICANT.

RECOMMENDATION:
- Validation tests FAIL or show no statistical significance
- DO NOT TRADE this strategy in its current form

NEXT STEPS:
1. DO NOT DEPLOY to live trading
2. Investigate potential overfitting:
   - Check for look-ahead bias in signals
   - Too many parameters optimized in-sample?
   - Regime-specific (only works in specific market conditions)?
3. Perform deeper analysis:
   - Examine individual trades for anomalies
   - Check data quality and preprocessing
   - Verify MCPT permutation logic is correct
4. Consider complete strategy redesign with stricter validation

CRITICAL WARNING:
- This strategy does not pass statistical validation
- Backtested performance is likely due to overfitting or luck
- Trading this strategy risks real capital on false alpha
"""

    def get_summary_dict(self) -> dict:
        """
        Returns dictionary representation suitable for JSON serialization.
        """
        return {
            'backtest': {
                'sharpe': float(self.backtest_sharpe),
                'nw_sharpe': float(self.backtest_nw_sharpe),
                'total_return': float(self.total_return),
                'max_drawdown': float(self.max_drawdown),
                'num_trades': self.num_trades,
            },
            'mcpt': {
                'pvalue': self.mcpt_pvalue,
                'status': self.mcpt_status,
            },
            'walk_forward': {
                'avg_train_sharpe': self.wf_avg_train_sharpe,
                'avg_test_sharpe': self.wf_avg_test_sharpe,
                'efficiency': self.wf_avg_efficiency,
                'folds_passed': f"{self.wf_folds_passed}/{self.wf_total_folds}",
                'status': self.wf_status,
            },
            'bootstrap': {
                'sharpe_ci': list(self.bootstrap_sharpe_ci),
                'nw_sharpe_ci': list(self.bootstrap_nw_sharpe_ci),
                'return_ci': list(self.bootstrap_return_ci),
                'status': self.bootstrap_status,
            },
            'overall': {
                'status': self.overall_status,
                'is_pass': self.is_pass(),
                'is_strong': self.is_strong(),
                'is_marginal': self.is_marginal(),
            },
            'recommendation': self.get_recommendation(),
        }
