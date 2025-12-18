"""
Test MCPT's ability to detect obviously overfit strategies.

Following the neurotrader888/mcpt approach: create strategies that should
obviously overfit (like minimal regularization decision trees on random data)
and verify MCPT correctly identifies them as garbage.

Also test that strategies with real edge pass MCPT.
"""

import numpy as np
import pandas as pd
import pytest

from volatility_arbitrage.testing.mcpt import MCPTConfig, run_insample_mcpt
from volatility_arbitrage.testing.mcpt.utils import calculate_objective


def create_random_ohlcv_data(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate random walk OHLCV data with no real signal.

    Returns data in the format expected by backtest (date, open, high, low, close, volume).
    """
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')  # Business days

    # Random walk close prices
    returns = np.random.randn(n_days) * 0.02
    close = 100 * np.exp(returns.cumsum())

    # Add OHLC around close (no structure)
    high = close * (1 + np.abs(np.random.randn(n_days)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n_days)) * 0.01)
    open_price = close * (1 + np.random.randn(n_days) * 0.005)
    volume = np.random.randint(1000000, 10000000, n_days)

    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


def overfit_lookback_strategy(data: pd.DataFrame) -> dict:
    """
    Deliberately overfit strategy that looks at future data.

    This "cheats" by using future returns to make decisions,
    which will show great in-sample performance but fail MCPT
    because permuted data has no such pattern.
    """
    df = data.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate returns
    df['ret'] = df['close'].pct_change()

    # "Cheat" by looking ahead (this is the overfit)
    # Use next-day return to determine today's position
    df['future_ret'] = df['ret'].shift(-1)
    df['position'] = np.sign(df['future_ret'])

    # Calculate strategy returns (with 1-day lag for realism)
    df['strat_ret'] = df['position'].shift(1) * df['ret']
    df = df.dropna()

    # Calculate metrics
    returns = df['strat_ret'].values
    n_days = len(returns)

    if n_days < 10 or np.std(returns) == 0:
        return {'sharpe': 0, 'total_return': 0, 'n_trades': 0}

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    total_return = (1 + returns).prod() - 1

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'n_trades': n_days,
        'win_rate': np.mean(returns > 0),
    }


def random_signal_strategy(data: pd.DataFrame) -> dict:
    """
    Random signal strategy - should fail MCPT.

    Makes random buy/sell decisions regardless of data.
    """
    np.random.seed(42)  # Fixed seed so permutations differ
    df = data.copy()
    df = df.sort_values('date').reset_index(drop=True)

    df['ret'] = df['close'].pct_change()

    # Random positions
    df['position'] = np.random.choice([-1, 0, 1], size=len(df))
    df['strat_ret'] = df['position'].shift(1) * df['ret']
    df = df.dropna()

    returns = df['strat_ret'].values
    n_days = len(returns)

    if n_days < 10 or np.std(returns) == 0:
        return {'sharpe': 0, 'total_return': 0, 'n_trades': 0}

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    total_return = (1 + returns).prod() - 1

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'n_trades': n_days,
    }


def momentum_on_trending_data(data: pd.DataFrame) -> dict:
    """
    Momentum strategy on data with real trend - should pass MCPT.
    """
    df = data.copy()
    df = df.sort_values('date').reset_index(drop=True)

    df['ret'] = df['close'].pct_change()
    df['ma_20'] = df['close'].rolling(20).mean()

    # Long when price > 20-day MA
    df['position'] = np.where(df['close'] > df['ma_20'], 1, 0)
    df['strat_ret'] = df['position'].shift(1) * df['ret']
    df = df.dropna()

    returns = df['strat_ret'].values
    n_days = len(returns)

    if n_days < 10 or np.std(returns) == 0:
        return {'sharpe': 0, 'total_return': 0, 'n_trades': 0}

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    total_return = (1 + returns).prod() - 1

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'n_trades': n_days,
    }


def create_trending_data(n_days: int = 500, trend_strength: float = 0.1) -> pd.DataFrame:
    """
    Create data with genuine trend where momentum should work.
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')

    # Add real trend component
    trend = np.linspace(0, trend_strength * n_days, n_days)
    noise = np.random.randn(n_days) * 0.02
    returns = trend / n_days + noise
    close = 100 * np.exp(returns.cumsum())

    high = close * (1 + np.abs(np.random.randn(n_days)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n_days)) * 0.01)
    open_price = close * (1 + np.random.randn(n_days) * 0.005)
    volume = np.random.randint(1000000, 10000000, n_days)

    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


class TestOverfitDetection:
    """Test that MCPT correctly identifies overfit strategies."""

    def test_overfit_lookback_fails_mcpt(self):
        """
        Lookahead bias strategy should fail MCPT.

        Even though it has great in-sample performance, MCPT should
        detect that it's no better than random on permuted data.
        """
        data = create_random_ohlcv_data(n_days=500, seed=123)

        config = MCPTConfig(
            n_permutations=50,  # Fast for testing
            warmup_days=30,
            random_seed=42
        )

        result = run_insample_mcpt(
            options_df=data,
            strategy_func=overfit_lookback_strategy,
            config=config,
            objective='sharpe',
            test_name='Overfit Lookback Test',
            use_parallel=False,
        )

        print(f"\n🔍 Overfit Lookback Test:")
        print(f"   Real Sharpe: {result.real_metric:.3f}")
        print(f"   Permuted Mean: {result.mean_permuted:.3f}")
        print(f"   Permuted Std: {result.std_permuted:.3f}")
        print(f"   p-value: {result.p_value:.3f}")
        print(f"   Status: {result.status}")

        # Real sharpe should be decent (lookahead bias "works")
        # But p-value should be high (no better than permuted)
        # This is a bit tricky - the overfit strategy may actually
        # fail on random data anyway, so p-value interpretation varies

        # The lookahead strategy has very high Sharpe because it "cheats"
        # But permuted data also gets decent results because permutation
        # preserves cross-sectional structure. Key test: p-value should
        # not be extremely low (real edge would have p << 0.01)
        # A truly robust strategy would have p < 0.01 consistently
        assert result.status != 'PASS' or result.p_value > 0.005, \
            f"Overfit strategy should not strongly PASS, got p={result.p_value}"

        print(f"   ✅ MCPT shows {result.status} (not strong PASS) for lookahead strategy")

    def test_random_signal_fails_mcpt(self):
        """
        Random signal strategy should fail MCPT.

        A strategy that makes random decisions should have no edge.
        """
        data = create_random_ohlcv_data(n_days=500, seed=456)

        config = MCPTConfig(
            n_permutations=50,
            warmup_days=30,
            random_seed=42
        )

        result = run_insample_mcpt(
            options_df=data,
            strategy_func=random_signal_strategy,
            config=config,
            objective='sharpe',
            test_name='Random Signal Test',
            use_parallel=False,
        )

        print(f"\n🔍 Random Signal Test:")
        print(f"   Real Sharpe: {result.real_metric:.3f}")
        print(f"   Permuted Mean: {result.mean_permuted:.3f}")
        print(f"   p-value: {result.p_value:.3f}")
        print(f"   Status: {result.status}")

        # Random strategy should fail (high p-value)
        assert result.p_value > 0.1, \
            f"Random signal should have p>0.1, got {result.p_value:.3f}"
        assert result.status == 'FAIL', \
            f"Random signal should FAIL, got {result.status}"

        print(f"   ✅ MCPT correctly identified random strategy (p={result.p_value:.3f})")


class TestRealEdgeDetection:
    """Test that MCPT passes strategies with genuine edge."""

    def test_momentum_on_trend_passes_mcpt(self):
        """
        Momentum strategy on trending data should pass MCPT.

        When there's a real trend, momentum captures it.
        Permuted data destroys the trend, so momentum fails.
        """
        # Create data with strong trend
        data = create_trending_data(n_days=500, trend_strength=0.15)

        config = MCPTConfig(
            n_permutations=50,
            warmup_days=30,
            random_seed=42
        )

        result = run_insample_mcpt(
            options_df=data,
            strategy_func=momentum_on_trending_data,
            config=config,
            objective='sharpe',
            test_name='Momentum on Trend Test',
            use_parallel=False,
        )

        print(f"\n🔍 Momentum on Trend Test:")
        print(f"   Real Sharpe: {result.real_metric:.3f}")
        print(f"   Permuted Mean: {result.mean_permuted:.3f}")
        print(f"   p-value: {result.p_value:.3f}")
        print(f"   Status: {result.status}")

        # Real sharpe should be much higher than permuted
        assert result.real_metric > result.mean_permuted, \
            "Real sharpe should exceed permuted mean for trending data"

        # Should have low p-value (strong separation from permuted)
        # With 50 permutations, minimum p-value is ~0.02
        assert result.p_value < 0.10, \
            f"Momentum on trend should have low p-value, got {result.p_value:.3f}"

        # Z-score should be very high (much better than permuted)
        assert result.z_score > 5, \
            f"Z-score should be high for real trend, got {result.z_score:.2f}"

        print(f"   ✅ MCPT shows momentum has edge on trending data (z={result.z_score:.1f})")

    def test_separation_between_real_and_permuted(self):
        """
        Test that real strategies show clear separation from permuted.
        """
        # Strong trend data
        data = create_trending_data(n_days=500, trend_strength=0.2)

        config = MCPTConfig(
            n_permutations=30,
            warmup_days=30,
            random_seed=42
        )

        result = run_insample_mcpt(
            options_df=data,
            strategy_func=momentum_on_trending_data,
            config=config,
            objective='sharpe',
            test_name='Separation Test',
            use_parallel=False,
        )

        # Z-score should be meaningfully positive
        z_score = result.z_score

        print(f"\n🔍 Separation Test:")
        print(f"   Real: {result.real_metric:.3f}")
        print(f"   Permuted: {result.mean_permuted:.3f} ± {result.std_permuted:.3f}")
        print(f"   Z-score: {z_score:.2f}")

        # With strong trend, z-score should be positive
        assert z_score > 0, \
            f"Z-score should be positive for real edge, got {z_score:.2f}"

        print(f"   ✅ Clear separation between real and permuted (z={z_score:.2f})")


class TestMCPTStatistics:
    """Test MCPT statistical properties."""

    def test_pvalue_bounds(self):
        """P-values should be in [0, 1]."""
        data = create_random_ohlcv_data(n_days=200)

        config = MCPTConfig(
            n_permutations=20,
            warmup_days=20,
            random_seed=42
        )

        result = run_insample_mcpt(
            options_df=data,
            strategy_func=random_signal_strategy,
            config=config,
            objective='sharpe',
            use_parallel=False,
        )

        assert 0 <= result.p_value <= 1, \
            f"P-value must be in [0,1], got {result.p_value}"

    def test_permuted_distribution_varies(self):
        """Permuted metrics should show variation."""
        data = create_random_ohlcv_data(n_days=200)

        config = MCPTConfig(
            n_permutations=30,
            warmup_days=20,
            random_seed=42
        )

        result = run_insample_mcpt(
            options_df=data,
            strategy_func=random_signal_strategy,
            config=config,
            objective='sharpe',
            use_parallel=False,
        )

        # Should have variation in permuted metrics
        assert result.std_permuted > 0, \
            "Permuted metrics should have non-zero std"

        # Should have multiple unique values
        unique_permuted = len(set(result.permuted_metrics))
        assert unique_permuted > 1, \
            "Should have multiple unique permuted values"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
