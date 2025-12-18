"""
Configuration for Monte Carlo Permutation Testing (MCPT).

Defines parameters for permutation tests, significance thresholds,
and walk-forward settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path

import yaml


@dataclass
class MCPTConfig:
    """
    Configuration for MCPT validation.

    Attributes:
        n_permutations: Number of permutations for in-sample test (default 1000)
        n_permutations_walkforward: Fewer permutations for walk-forward (more expensive)
        significance_level: p-value threshold for "pass" (default 0.01)
        marginal_significance: p-value threshold for "marginal" (default 0.05)
        n_jobs: Number of parallel workers (-1 = all CPUs)
        random_seed: Base random seed for reproducibility
        warmup_days: Days to preserve at start for indicator warmup

        Walk-forward settings:
        train_window_days: Training window in trading days (4 years default)
        test_window_days: Test window in trading days (1 year default)
        retrain_days: Retraining frequency in trading days (monthly)

        objectives: List of objective functions to compute
    """
    # Permutation settings
    n_permutations: int = 1000
    n_permutations_walkforward: int = 200
    significance_level: float = 0.01
    marginal_significance: float = 0.05

    # Parallelization
    n_jobs: int = -1  # -1 = all CPUs
    random_seed: int = 42

    # Warmup
    warmup_days: int = 80  # Days to preserve for indicator calculation

    # Walk-forward settings
    train_window_days: int = 252 * 4  # 4 years training
    test_window_days: int = 252       # 1 year testing
    retrain_days: int = 30            # Monthly retrain

    # Objectives to compute
    objectives: List[str] = field(default_factory=lambda: [
        'sharpe', 'nw_sharpe', 'profit_factor', 'win_rate'
    ])

    def is_significant(self, p_value: float) -> bool:
        """Check if p-value indicates statistical significance."""
        return p_value < self.significance_level

    def is_marginal(self, p_value: float) -> bool:
        """Check if p-value is marginally significant."""
        return self.significance_level <= p_value < self.marginal_significance

    def get_status(self, p_value: float) -> str:
        """Get status string for p-value."""
        if self.is_significant(p_value):
            return "PASS"
        elif self.is_marginal(p_value):
            return "MARGINAL"
        return "FAIL"

    @classmethod
    def from_yaml(cls, path: Path) -> "MCPTConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        mcpt_data = data.get('mcpt', data)
        return cls(
            n_permutations=mcpt_data.get('n_permutations', 1000),
            n_permutations_walkforward=mcpt_data.get('n_permutations_walkforward', 200),
            significance_level=mcpt_data.get('significance_level', 0.01),
            marginal_significance=mcpt_data.get('marginal_significance', 0.05),
            n_jobs=mcpt_data.get('n_jobs', -1),
            random_seed=mcpt_data.get('random_seed', 42),
            warmup_days=mcpt_data.get('warmup_days', 80),
            train_window_days=mcpt_data.get('train_window_days', 252 * 4),
            test_window_days=mcpt_data.get('test_window_days', 252),
            retrain_days=mcpt_data.get('retrain_days', 30),
            objectives=mcpt_data.get('objectives', ['sharpe', 'nw_sharpe', 'profit_factor']),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            'mcpt': {
                'n_permutations': self.n_permutations,
                'n_permutations_walkforward': self.n_permutations_walkforward,
                'significance_level': self.significance_level,
                'marginal_significance': self.marginal_significance,
                'n_jobs': self.n_jobs,
                'random_seed': self.random_seed,
                'warmup_days': self.warmup_days,
                'train_window_days': self.train_window_days,
                'test_window_days': self.test_window_days,
                'retrain_days': self.retrain_days,
                'objectives': self.objectives,
            }
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
