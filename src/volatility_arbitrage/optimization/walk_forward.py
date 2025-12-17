"""
Walk-Forward Optimization for volatility arbitrage strategies.

Implements time-series cross-validation to detect overfitting and validate
strategy robustness. Uses rolling windows: train on N months, test on M months,
then shift forward.

Typical Configuration:
- Training window: 6 months
- Testing window: 3 months
- Step size: 3 months (no overlap in test sets)

Overfitting Detection:
- Efficiency Ratio: test_sharpe / train_sharpe
- Threshold: >0.5 = acceptable, <0.3 = likely overfit
- Consistency: Check if test performance is stable across folds
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable, Dict, Any, List, Optional

import pandas as pd
import numpy as np

from volatility_arbitrage.backtest.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_returns,
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """
    Single walk-forward fold result.

    Contains training and testing metrics for one time period.
    """

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Training metrics
    train_sharpe: Decimal
    train_return: Decimal
    train_max_dd: Decimal
    train_num_trades: int

    # Testing metrics
    test_sharpe: Decimal
    test_return: Decimal
    test_max_dd: Decimal
    test_num_trades: int

    # Overfitting indicators
    efficiency_ratio: Decimal
    is_overfit: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert fold to dictionary."""
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.strftime("%Y-%m-%d"),
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "test_start": self.test_start.strftime("%Y-%m-%d"),
            "test_end": self.test_end.strftime("%Y-%m-%d"),
            "train_sharpe": float(self.train_sharpe),
            "train_return": float(self.train_return),
            "train_max_dd": float(self.train_max_dd),
            "train_num_trades": self.train_num_trades,
            "test_sharpe": float(self.test_sharpe),
            "test_return": float(self.test_return),
            "test_max_dd": float(self.test_max_dd),
            "test_num_trades": self.test_num_trades,
            "efficiency_ratio": float(self.efficiency_ratio),
            "is_overfit": self.is_overfit,
        }


@dataclass
class WalkForwardResult:
    """
    Aggregate walk-forward optimization results.

    Summarizes performance across all folds with overfitting assessment.
    """

    folds: List[WalkForwardFold]

    # Aggregate metrics
    avg_train_sharpe: Decimal
    avg_test_sharpe: Decimal
    avg_efficiency_ratio: Decimal

    # Consistency metrics
    test_sharpe_std: Decimal
    test_return_std: Decimal

    # Overfitting flags
    num_overfit_folds: int
    is_likely_overfit: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "num_folds": len(self.folds),
            "avg_train_sharpe": float(self.avg_train_sharpe),
            "avg_test_sharpe": float(self.avg_test_sharpe),
            "avg_efficiency_ratio": float(self.avg_efficiency_ratio),
            "test_sharpe_std": float(self.test_sharpe_std),
            "test_return_std": float(self.test_return_std),
            "num_overfit_folds": self.num_overfit_folds,
            "is_likely_overfit": self.is_likely_overfit,
            "folds": [fold.to_dict() for fold in self.folds],
        }


class WalkForwardOptimizer:
    """
    Walk-forward optimization engine.

    Performs time-series cross-validation by splitting data into rolling
    train/test windows and evaluating strategy performance.
    """

    def __init__(
        self,
        train_months: int = 6,
        test_months: int = 3,
        step_months: Optional[int] = None,
        efficiency_threshold: float = 0.5,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            train_months: Training window size in months
            test_months: Testing window size in months
            step_months: Step size in months (default: test_months)
            efficiency_threshold: Threshold for overfitting detection (default: 0.5)
        """
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months or test_months
        self.efficiency_threshold = efficiency_threshold

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        risk_free_rate: Decimal = Decimal("0.05"),
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            data: Market data DataFrame with datetime index
            strategy_func: Strategy function that takes (data, params) and returns equity curve
            param_grid: Parameter grid for optimization (if None, uses default params)
            risk_free_rate: Annual risk-free rate

        Returns:
            WalkForwardResult with all folds and aggregate metrics
        """
        logger.info(
            f"Starting walk-forward optimization: "
            f"train={self.train_months}m, test={self.test_months}m, step={self.step_months}m"
        )

        # Generate fold dates
        folds_dates = self._generate_folds(data.index)
        logger.info(f"Generated {len(folds_dates)} folds")

        folds_results = []

        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds_dates):
            logger.info(
                f"Fold {fold_id + 1}/{len(folds_dates)}: "
                f"train {train_start.date()}-{train_end.date()}, "
                f"test {test_start.date()}-{test_end.date()}"
            )

            # Split data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]

            # Optimize on training set (if param_grid provided)
            if param_grid:
                best_params = self._optimize_params(train_data, strategy_func, param_grid, risk_free_rate)
            else:
                best_params = {}

            # Run backtest on training set
            train_equity = strategy_func(train_data, best_params)
            train_metrics = self._calculate_metrics(train_equity, risk_free_rate)

            # Run backtest on testing set with same params
            test_equity = strategy_func(test_data, best_params)
            test_metrics = self._calculate_metrics(test_equity, risk_free_rate)

            # Calculate efficiency ratio
            efficiency_ratio = (
                test_metrics["sharpe"] / train_metrics["sharpe"]
                if train_metrics["sharpe"] > 0
                else Decimal("0")
            )
            is_overfit = efficiency_ratio < Decimal(str(self.efficiency_threshold))

            fold_result = WalkForwardFold(
                fold_id=fold_id + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_sharpe=train_metrics["sharpe"],
                train_return=train_metrics["return"],
                train_max_dd=train_metrics["max_dd"],
                train_num_trades=train_metrics["num_trades"],
                test_sharpe=test_metrics["sharpe"],
                test_return=test_metrics["return"],
                test_max_dd=test_metrics["max_dd"],
                test_num_trades=test_metrics["num_trades"],
                efficiency_ratio=efficiency_ratio,
                is_overfit=is_overfit,
            )

            folds_results.append(fold_result)

            logger.info(
                f"  Train Sharpe: {train_metrics['sharpe']:.2f}, "
                f"Test Sharpe: {test_metrics['sharpe']:.2f}, "
                f"Efficiency: {efficiency_ratio:.2f} "
                f"({'OVERFIT' if is_overfit else 'OK'})"
            )

        # Aggregate results
        return self._aggregate_results(folds_results)

    def _generate_folds(self, dates: pd.DatetimeIndex) -> List[tuple]:
        """
        Generate train/test fold dates.

        Args:
            dates: Full date range

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        folds = []
        start_date = dates[0]
        end_date = dates[-1]

        current_date = start_date

        while True:
            # Calculate train window
            train_start = current_date
            train_end = train_start + timedelta(days=30 * self.train_months)

            # Calculate test window
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=30 * self.test_months)

            # Check if we have enough data
            if test_end > end_date:
                break

            folds.append((train_start, train_end, test_start, test_end))

            # Step forward
            current_date += timedelta(days=30 * self.step_months)

        return folds

    def _optimize_params(
        self,
        data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
        param_grid: Dict[str, List[Any]],
        risk_free_rate: Decimal,
    ) -> Dict[str, Any]:
        """
        Optimize parameters on training data using grid search.

        Args:
            data: Training data
            strategy_func: Strategy function
            param_grid: Parameter grid
            risk_free_rate: Risk-free rate

        Returns:
            Best parameters dict
        """
        # Generate parameter combinations
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        best_sharpe = Decimal("-inf")
        best_params = {}

        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))

            try:
                equity = strategy_func(data, params)
                metrics = self._calculate_metrics(equity, risk_free_rate)

                if metrics["sharpe"] > best_sharpe:
                    best_sharpe = metrics["sharpe"]
                    best_params = params
            except Exception as e:
                logger.warning(f"Failed to evaluate params {params}: {e}")
                continue

        logger.debug(f"Best params: {best_params} (Sharpe: {best_sharpe:.2f})")
        return best_params

    def _calculate_metrics(
        self, equity_curve: pd.DataFrame, risk_free_rate: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate performance metrics from equity curve.

        Args:
            equity_curve: DataFrame with 'total_equity' column
            risk_free_rate: Annual risk-free rate

        Returns:
            Dict with sharpe, return, max_dd, num_trades
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return {
                "sharpe": Decimal("0"),
                "return": Decimal("0"),
                "max_dd": Decimal("0"),
                "num_trades": 0,
            }

        # Calculate returns
        equity_series = equity_curve["total_equity"]
        returns = calculate_returns(equity_series)

        # Sharpe ratio
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate)

        # Total return
        total_return = Decimal(str((equity_series.iloc[-1] / equity_series.iloc[0]) - 1)) * Decimal("100")

        # Max drawdown
        _, max_dd_pct = calculate_max_drawdown(equity_series)

        # Number of trades (if available)
        num_trades = len(equity_curve.get("trades", []))

        return {
            "sharpe": sharpe,
            "return": total_return,
            "max_dd": max_dd_pct,
            "num_trades": num_trades,
        }

    def _aggregate_results(self, folds: List[WalkForwardFold]) -> WalkForwardResult:
        """
        Aggregate fold results into summary statistics.

        Args:
            folds: List of fold results

        Returns:
            Aggregate WalkForwardResult
        """
        if not folds:
            return WalkForwardResult(
                folds=[],
                avg_train_sharpe=Decimal("0"),
                avg_test_sharpe=Decimal("0"),
                avg_efficiency_ratio=Decimal("0"),
                test_sharpe_std=Decimal("0"),
                test_return_std=Decimal("0"),
                num_overfit_folds=0,
                is_likely_overfit=True,
            )

        # Calculate averages
        avg_train_sharpe = Decimal(str(np.mean([float(f.train_sharpe) for f in folds])))
        avg_test_sharpe = Decimal(str(np.mean([float(f.test_sharpe) for f in folds])))
        avg_efficiency = Decimal(str(np.mean([float(f.efficiency_ratio) for f in folds])))

        # Calculate consistency
        test_sharpes = [float(f.test_sharpe) for f in folds]
        test_returns = [float(f.test_return) for f in folds]

        test_sharpe_std = Decimal(str(np.std(test_sharpes))) if len(test_sharpes) > 1 else Decimal("0")
        test_return_std = Decimal(str(np.std(test_returns))) if len(test_returns) > 1 else Decimal("0")

        # Overfitting assessment
        num_overfit = sum(1 for f in folds if f.is_overfit)
        is_likely_overfit = num_overfit > len(folds) / 2  # More than half overfit

        return WalkForwardResult(
            folds=folds,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            avg_efficiency_ratio=avg_efficiency,
            test_sharpe_std=test_sharpe_std,
            test_return_std=test_return_std,
            num_overfit_folds=num_overfit,
            is_likely_overfit=is_likely_overfit,
        )
