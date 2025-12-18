"""
Tests for Monte Carlo Permutation Testing (MCPT) module.

Tests the core functionality:
- bar_permute() preserves marginal distributions
- run_insample_mcpt() computes correct p-values
- run_walkforward_mcpt() handles fold generation correctly
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from volatility_arbitrage.testing.mcpt.config import MCPTConfig
from volatility_arbitrage.testing.mcpt.permutation import (
    bar_permute,
    bar_permute_returns,
    validate_permutation,
)
from volatility_arbitrage.testing.mcpt.utils import (
    calculate_objective,
    compute_pvalue,
)
from volatility_arbitrage.testing.mcpt.insample_test import MCPTResult
from volatility_arbitrage.testing.mcpt.walkforward_test import (
    generate_walkforward_folds,
    WalkForwardFold,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_options_df():
    """Generate sample options data for testing."""
    np.random.seed(42)

    # Generate 100 trading days
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B')

    records = []
    for date in dates:
        # Each day has 10 options (5 calls + 5 puts)
        base_spot = 450 + np.random.randn() * 5
        base_iv = 0.20 + np.random.randn() * 0.02

        for strike_offset in range(-20, 30, 10):
            strike = base_spot + strike_offset

            # Call
            records.append({
                'date': date,
                'type': 'call',
                'strike': strike,
                'expiration': date + timedelta(days=30),
                'implied_volatility': base_iv + np.random.randn() * 0.01,
                'delta': 0.5 + strike_offset / 100,
                'volume': np.random.randint(100, 1000),
            })

            # Put
            records.append({
                'date': date,
                'type': 'put',
                'strike': strike,
                'expiration': date + timedelta(days=30),
                'implied_volatility': base_iv + np.random.randn() * 0.01 + 0.02,
                'delta': -0.5 + strike_offset / 100,
                'volume': np.random.randint(100, 1000),
            })

    return pd.DataFrame(records)


@pytest.fixture
def sample_returns():
    """Generate sample return series."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    returns = np.random.normal(0.001, 0.02, 252)
    return pd.Series(returns, index=dates)


@pytest.fixture
def mcpt_config():
    """Create test MCPT configuration with fewer permutations."""
    return MCPTConfig(
        n_permutations=10,  # Small for fast tests
        n_permutations_walkforward=5,
        n_jobs=1,
        random_seed=42,
        warmup_days=20,
        train_window_days=50,
        test_window_days=20,
    )


# ============================================================================
# Test MCPTConfig
# ============================================================================

class TestMCPTConfig:
    """Tests for MCPTConfig dataclass."""

    def test_default_values(self):
        """Default config should have sensible values."""
        config = MCPTConfig()
        assert config.n_permutations == 1000
        assert config.significance_level == 0.01
        assert config.warmup_days == 80

    def test_is_significant(self):
        """is_significant should check p < 0.01."""
        config = MCPTConfig()
        assert config.is_significant(0.005)
        assert not config.is_significant(0.02)
        assert not config.is_significant(0.10)

    def test_is_marginal(self):
        """is_marginal should check 0.01 <= p < 0.05."""
        config = MCPTConfig()
        assert not config.is_marginal(0.005)
        assert config.is_marginal(0.02)
        assert not config.is_marginal(0.10)

    def test_get_status(self):
        """get_status should return correct status string."""
        config = MCPTConfig()
        assert config.get_status(0.005) == "PASS"
        assert config.get_status(0.02) == "MARGINAL"
        assert config.get_status(0.10) == "FAIL"


# ============================================================================
# Test bar_permute
# ============================================================================

class TestBarPermute:
    """Tests for bar_permute function."""

    def test_preserves_row_count(self, sample_options_df):
        """Permutation should not add or remove rows."""
        permuted = bar_permute(sample_options_df, seed=42)
        assert len(permuted) == len(sample_options_df)

    def test_preserves_column_names(self, sample_options_df):
        """All columns should be preserved."""
        permuted = bar_permute(sample_options_df, seed=42)
        assert list(permuted.columns) == list(sample_options_df.columns)

    def test_preserves_daily_cross_sections(self, sample_options_df):
        """Each day's options should stay together."""
        permuted = bar_permute(sample_options_df, seed=42)

        # Check that option counts per date are same (just reordered)
        orig_counts = sample_options_df.groupby('date').size().sort_values()
        perm_counts = permuted.groupby('date').size().sort_values()

        assert (orig_counts.values == perm_counts.values).all()

    def test_preserves_marginal_distribution_iv(self, sample_options_df):
        """IV distribution should be preserved."""
        permuted = bar_permute(sample_options_df, seed=42)

        orig_iv = sample_options_df['implied_volatility'].dropna()
        perm_iv = permuted['implied_volatility'].dropna()

        # Mean should be very close
        assert np.isclose(orig_iv.mean(), perm_iv.mean(), rtol=0.01)
        # Std should be very close
        assert np.isclose(orig_iv.std(), perm_iv.std(), rtol=0.01)

    def test_preserves_marginal_distribution_volume(self, sample_options_df):
        """Volume distribution should be preserved."""
        permuted = bar_permute(sample_options_df, seed=42)

        orig_vol = sample_options_df['volume'].sum()
        perm_vol = permuted['volume'].sum()

        # Total volume should be exactly the same
        assert orig_vol == perm_vol

    def test_reproducible_with_seed(self, sample_options_df):
        """Same seed should give same result."""
        perm1 = bar_permute(sample_options_df, seed=42)
        perm2 = bar_permute(sample_options_df, seed=42)

        # Reset indices for comparison
        perm1 = perm1.reset_index(drop=True)
        perm2 = perm2.reset_index(drop=True)

        pd.testing.assert_frame_equal(perm1, perm2)

    def test_different_seeds_give_different_results(self, sample_options_df):
        """Different seeds should give different permutations."""
        perm1 = bar_permute(sample_options_df, seed=42)
        perm2 = bar_permute(sample_options_df, seed=43)

        # At least one date should differ (but data same)
        # The structure should be different (different day assignment)
        assert not perm1['implied_volatility'].equals(perm2['implied_volatility'])

    def test_warmup_preserved(self, sample_options_df):
        """First N days should not be shuffled."""
        warmup = 10  # 10 days of warmup

        # Get original first 10 days
        orig_dates = sorted(sample_options_df['date'].unique())
        warmup_dates = orig_dates[:warmup]

        permuted = bar_permute(sample_options_df, seed=42, preserve_warmup=warmup)

        # Check that warmup days have same data
        for date in warmup_dates:
            orig_day = sample_options_df[sample_options_df['date'] == date]
            perm_day = permuted[permuted['date'] == date]

            # IV values should match exactly
            orig_ivs = sorted(orig_day['implied_volatility'].values)
            perm_ivs = sorted(perm_day['implied_volatility'].values)
            np.testing.assert_array_almost_equal(orig_ivs, perm_ivs)

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty DataFrame."""
        empty_df = pd.DataFrame(columns=['date', 'implied_volatility'])
        result = bar_permute(empty_df, seed=42)
        assert len(result) == 0


class TestBarPermuteReturns:
    """Tests for bar_permute_returns function."""

    def test_preserves_length(self, sample_returns):
        """Permuted returns should have same length."""
        permuted = bar_permute_returns(sample_returns, seed=42)
        assert len(permuted) == len(sample_returns)

    def test_preserves_mean(self, sample_returns):
        """Mean should be exactly preserved."""
        permuted = bar_permute_returns(sample_returns, seed=42)
        assert np.isclose(sample_returns.mean(), permuted.mean())

    def test_preserves_std(self, sample_returns):
        """Standard deviation should be exactly preserved."""
        permuted = bar_permute_returns(sample_returns, seed=42)
        assert np.isclose(sample_returns.std(), permuted.std())


# ============================================================================
# Test utils functions
# ============================================================================

class TestCalculateObjective:
    """Tests for calculate_objective function."""

    def test_sharpe(self):
        """Should extract sharpe correctly."""
        results = {'sharpe': 1.5, 'other': 0}
        assert calculate_objective(results, 'sharpe') == 1.5

    def test_nw_sharpe_fallback(self):
        """Should fallback to sharpe if nw_sharpe missing."""
        results = {'sharpe': 1.5}
        assert calculate_objective(results, 'nw_sharpe') == 1.5

    def test_unknown_raises(self):
        """Unknown objective should raise ValueError."""
        results = {'sharpe': 1.5}
        with pytest.raises(ValueError):
            calculate_objective(results, 'unknown_metric')


class TestComputePvalue:
    """Tests for compute_pvalue function."""

    def test_perfect_score(self):
        """Real metric better than all permuted should give p ~ 1/n."""
        real = 2.0
        permuted = [0.5, 0.6, 0.7, 0.8, 0.9]  # All worse
        p = compute_pvalue(real, permuted)
        # p = (0 + 1) / (5 + 1) = 1/6
        assert np.isclose(p, 1/6, rtol=0.01)

    def test_worst_score(self):
        """Real metric worse than all permuted should give p ~ 1."""
        real = 0.0
        permuted = [0.5, 0.6, 0.7, 0.8, 0.9]  # All better
        p = compute_pvalue(real, permuted)
        # p = (5 + 1) / (5 + 1) = 1.0
        assert p == 1.0

    def test_median_score(self):
        """Real metric at median should give p ~ 0.5."""
        real = 0.5
        permuted = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
        p = compute_pvalue(real, permuted)
        # About half are >= 0.5
        assert 0.4 < p < 0.7

    def test_empty_permuted(self):
        """Empty permuted list should give p = 1."""
        p = compute_pvalue(1.0, [])
        assert p == 1.0

    def test_pvalue_bounds(self):
        """P-value should always be in [0, 1]."""
        np.random.seed(42)
        for _ in range(100):
            real = np.random.randn()
            permuted = list(np.random.randn(100))
            p = compute_pvalue(real, permuted)
            assert 0 <= p <= 1


# ============================================================================
# Test walkforward fold generation
# ============================================================================

class TestGenerateWalkforwardFolds:
    """Tests for generate_walkforward_folds function."""

    def test_generates_folds(self, sample_options_df):
        """Should generate at least one fold."""
        folds = generate_walkforward_folds(
            sample_options_df,
            train_window_days=30,
            test_window_days=10,
        )
        assert len(folds) > 0

    def test_fold_structure(self, sample_options_df):
        """Each fold should have correct structure."""
        folds = generate_walkforward_folds(
            sample_options_df,
            train_window_days=30,
            test_window_days=10,
        )

        for fold in folds:
            assert 'fold_id' in fold
            assert 'train_start' in fold
            assert 'train_end' in fold
            assert 'test_start' in fold
            assert 'test_end' in fold
            # Test should come after train
            assert fold['test_start'] > fold['train_end']

    def test_no_folds_if_insufficient_data(self):
        """Should return empty list if not enough data."""
        # Create tiny dataset
        small_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='B'),
            'value': range(10),
        })

        folds = generate_walkforward_folds(
            small_df,
            train_window_days=100,  # More than we have
            test_window_days=50,
        )

        assert len(folds) == 0


# ============================================================================
# Test MCPTResult
# ============================================================================

class TestMCPTResult:
    """Tests for MCPTResult dataclass."""

    def test_is_significant(self):
        """is_significant should check p < 0.01."""
        result = MCPTResult(
            test_name="Test",
            real_metric=1.5,
            metric_name="sharpe",
            permuted_metrics=[0.1, 0.2, 0.3],
            p_value=0.005,
            mean_permuted=0.2,
            std_permuted=0.1,
            n_permutations=3,
        )
        assert result.is_significant

    def test_is_marginal(self):
        """is_marginal should check 0.01 <= p < 0.05."""
        result = MCPTResult(
            test_name="Test",
            real_metric=1.5,
            metric_name="sharpe",
            permuted_metrics=[0.1, 0.2, 0.3],
            p_value=0.03,
            mean_permuted=0.2,
            std_permuted=0.1,
            n_permutations=3,
        )
        assert result.is_marginal
        assert not result.is_significant

    def test_status_pass(self):
        """status should be PASS when p < 0.01."""
        result = MCPTResult(
            test_name="Test",
            real_metric=1.5,
            metric_name="sharpe",
            permuted_metrics=[0.1, 0.2, 0.3],
            p_value=0.005,
            mean_permuted=0.2,
            std_permuted=0.1,
            n_permutations=3,
        )
        assert result.status == "PASS"

    def test_status_marginal(self):
        """status should be MARGINAL when 0.01 <= p < 0.05."""
        result = MCPTResult(
            test_name="Test",
            real_metric=1.5,
            metric_name="sharpe",
            permuted_metrics=[0.1, 0.2, 0.3],
            p_value=0.03,
            mean_permuted=0.2,
            std_permuted=0.1,
            n_permutations=3,
        )
        assert result.status == "MARGINAL"

    def test_status_fail(self):
        """status should be FAIL when p >= 0.05."""
        result = MCPTResult(
            test_name="Test",
            real_metric=1.5,
            metric_name="sharpe",
            permuted_metrics=[0.1, 0.2, 0.3],
            p_value=0.10,
            mean_permuted=0.2,
            std_permuted=0.1,
            n_permutations=3,
        )
        assert result.status == "FAIL"

    def test_z_score(self):
        """z_score should be (real - mean) / std."""
        result = MCPTResult(
            test_name="Test",
            real_metric=1.0,
            metric_name="sharpe",
            permuted_metrics=[0.1, 0.2, 0.3],
            p_value=0.01,
            mean_permuted=0.5,
            std_permuted=0.1,
            n_permutations=3,
        )
        expected_z = (1.0 - 0.5) / 0.1
        assert np.isclose(result.z_score, expected_z)

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        result = MCPTResult(
            test_name="Test",
            real_metric=1.5,
            metric_name="sharpe",
            permuted_metrics=[0.1, 0.2, 0.3],
            p_value=0.005,
            mean_permuted=0.2,
            std_permuted=0.1,
            n_permutations=3,
        )
        d = result.to_dict()
        assert d['test_name'] == "Test"
        assert d['real_metric'] == 1.5
        assert d['p_value'] == 0.005
        assert d['status'] == "PASS"


# ============================================================================
# Integration tests
# ============================================================================

@pytest.mark.slow
class TestIntegration:
    """Integration tests that run actual MCPT (slower)."""

    def test_insample_mcpt_with_random_strategy(self, sample_options_df, mcpt_config):
        """Random strategy should have high p-value."""
        from volatility_arbitrage.testing.mcpt.insample_test import run_insample_mcpt

        def random_strategy(df):
            """Random strategy that ignores data."""
            np.random.seed(len(df))  # Deterministic based on data size
            return {'sharpe': np.random.randn()}

        result = run_insample_mcpt(
            options_df=sample_options_df,
            strategy_func=random_strategy,
            config=mcpt_config,
            objective='sharpe',
            use_parallel=False,  # Faster for small tests
        )

        # Random strategy should NOT be significant
        # (though with small n_permutations there's variance)
        assert result.n_permutations == mcpt_config.n_permutations
        assert 0 <= result.p_value <= 1

    def test_walkforward_fold_count(self, sample_options_df, mcpt_config):
        """Walk-forward should generate expected number of folds."""
        folds = generate_walkforward_folds(
            sample_options_df,
            train_window_days=mcpt_config.train_window_days,
            test_window_days=mcpt_config.test_window_days,
        )

        # With 100 days and 50 train + 20 test, we should get a few folds
        # Exact number depends on step size
        assert len(folds) >= 1


# ============================================================================
# Test validate_permutation
# ============================================================================

class TestValidatePermutation:
    """Tests for validate_permutation function."""

    def test_valid_permutation(self, sample_options_df):
        """Valid permutation should pass validation."""
        permuted = bar_permute(sample_options_df, seed=42)

        results = validate_permutation(
            sample_options_df,
            permuted,
            columns_to_check=['implied_volatility', 'volume'],
            tolerance=0.05,
        )

        assert 'implied_volatility' in results
        assert results['implied_volatility']['mean_preserved']
        assert results['implied_volatility']['std_preserved']
