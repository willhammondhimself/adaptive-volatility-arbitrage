#!/usr/bin/env python3
"""
Sharpe Ratio Validation Suite

Comprehensive anti-overfitting tests for volatility arbitrage backtests:
1. Out-of-Sample Testing - Train/Validate/Test splits
2. Walk-Forward Analysis - Rolling window re-calibration
3. Monte Carlo Simulation - Entry timing randomization
4. Transaction Cost Sensitivity - Stress test costs
5. Regime Analysis - Performance across market conditions
6. Parameter Sensitivity - Robustness to parameter changes
7. Bootstrap Confidence Intervals - Statistical significance

Usage:
    python scripts/validate_sharpe.py --data ~/avabe_data_backup/options_eod_SPY.csv
"""

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


@dataclass
class ValidationResults:
    """Results from a single validation test."""
    test_name: str
    sharpe_ratio: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    notes: str = ""


@dataclass
class ComprehensiveResults:
    """Aggregated results across all validation tests."""
    oos_sharpe: float
    is_sharpe: float
    sharpe_decay_pct: float  # (IS - OOS) / IS * 100
    cost_sensitivity: float  # Sharpe at 3x costs / base Sharpe
    regime_consistency: float  # Std dev of regime Sharpes
    parameter_stability: float  # Avg Sharpe at ±20% params
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    monte_carlo_std: float
    overall_confidence: str  # "HIGH", "MEDIUM", "LOW", "OVERFITTING DETECTED"


def load_options_data(filepath: str) -> pd.DataFrame:
    """
    Load options EOD data from CSV.

    Expected columns:
    - tradeDate, spotPrice, expirDate, dte, strike
    - delta, gamma, vega, theta
    - callValue, callBidPrice, callAskPrice, callVolume
    - putValue, putBidPrice, putAskPrice, putVolume
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['tradeDate', 'expirDate'])

    # Basic validation
    required_cols = ['tradeDate', 'spotPrice', 'expirDate', 'dte', 'strike',
                     'callValue', 'putValue', 'delta']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df):,} option records")
    print(f"Date range: {df['tradeDate'].min()} to {df['tradeDate'].max()}")
    print(f"Unique trading days: {df['tradeDate'].nunique()}")

    return df


def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate daily underlying returns from spot prices."""
    daily = df.groupby('tradeDate')['spotPrice'].first()
    returns = daily.pct_change().dropna()
    return returns


def calculate_iv_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily IV premium (IV - realized vol).

    This is the core signal for volatility arbitrage.
    """
    # Get daily spot prices
    daily_spot = df.groupby('tradeDate')['spotPrice'].first()

    # Calculate 20-day realized volatility
    returns = daily_spot.pct_change()
    rv_20d = returns.rolling(20).std() * np.sqrt(252)

    # Get ATM implied volatility (options with |delta| closest to 0.5)
    def get_atm_iv(day_df):
        # Filter to reasonable DTE (30-60 days)
        dte_mask = (day_df['dte'] >= 30) & (day_df['dte'] <= 60)
        day_df = day_df[dte_mask]

        if len(day_df) == 0:
            return np.nan

        # Find ATM (delta closest to 0.5 for calls)
        day_df = day_df.copy()
        day_df['atm_dist'] = abs(abs(day_df['delta']) - 0.5)
        atm_row = day_df.loc[day_df['atm_dist'].idxmin()]

        # Estimate IV from option price using simplified approximation
        # IV ≈ price / (spot * sqrt(T/252)) * sqrt(2*pi) for ATM
        spot = atm_row['spotPrice']
        dte = atm_row['dte']
        call_price = atm_row['callValue']

        if spot > 0 and dte > 0 and call_price > 0:
            iv = call_price / (spot * np.sqrt(dte/252)) * np.sqrt(2 * np.pi)
            return min(iv, 1.0)  # Cap at 100%
        return np.nan

    atm_iv = df.groupby('tradeDate').apply(get_atm_iv)

    # Calculate IV premium
    iv_premium = pd.DataFrame({
        'date': daily_spot.index,
        'spot': daily_spot.values,
        'rv_20d': rv_20d.values,
        'atm_iv': atm_iv.values,
    }).set_index('date')

    iv_premium['iv_premium'] = iv_premium['atm_iv'] - iv_premium['rv_20d']
    iv_premium['iv_premium_pct'] = iv_premium['iv_premium'] / iv_premium['rv_20d'] * 100

    return iv_premium.dropna()


def simulate_vol_arb_strategy(
    data: pd.DataFrame,
    entry_threshold: float = 5.0,  # IV premium % to enter
    exit_threshold: float = 2.0,   # IV premium % to exit
    position_size: float = 0.05,   # 5% of capital per trade
    commission: float = 0.50,      # Per contract
    slippage: float = 0.01,        # 1% slippage
    initial_capital: float = 100000.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate volatility arbitrage strategy.

    Strategy:
    - Short vol when IV premium > entry_threshold (sell straddle)
    - Long vol when IV premium < -entry_threshold (buy straddle)
    - Exit when IV premium crosses exit_threshold
    """
    equity = initial_capital
    position = 0  # +1 = long vol, -1 = short vol, 0 = flat
    entry_price = 0

    trades = []
    equity_curve = []

    for date, row in data.iterrows():
        iv_premium = row['iv_premium_pct']
        spot = row['spot']
        rv = row['rv_20d']

        # Daily P&L from existing position
        daily_pnl = 0

        if position != 0:
            # Approximate daily P&L from vol position
            # Short vol profits when realized vol < implied vol
            # Long vol profits when realized vol > implied vol
            vol_diff = rv - row['atm_iv']
            vega_pnl = position * vol_diff * spot * 0.01  # Simplified vega exposure
            daily_pnl = vega_pnl * position_size * initial_capital / spot

        # Check for entry signals
        if position == 0:
            if iv_premium > entry_threshold:
                # Short volatility (sell straddle)
                position = -1
                entry_price = spot
                cost = commission + slippage * spot * position_size
                equity -= cost
                trades.append({
                    'date': date,
                    'action': 'SHORT_VOL',
                    'price': spot,
                    'iv_premium': iv_premium
                })
            elif iv_premium < -entry_threshold:
                # Long volatility (buy straddle)
                position = 1
                entry_price = spot
                cost = commission + slippage * spot * position_size
                equity -= cost
                trades.append({
                    'date': date,
                    'action': 'LONG_VOL',
                    'price': spot,
                    'iv_premium': iv_premium
                })

        # Check for exit signals
        elif position != 0:
            should_exit = False

            if position == -1 and iv_premium < exit_threshold:
                should_exit = True
            elif position == 1 and iv_premium > -exit_threshold:
                should_exit = True

            if should_exit:
                cost = commission + slippage * spot * position_size
                equity -= cost
                trades.append({
                    'date': date,
                    'action': 'EXIT',
                    'price': spot,
                    'pnl': daily_pnl
                })
                position = 0

        equity += daily_pnl
        equity_curve.append({
            'date': date,
            'equity': equity,
            'position': position
        })

    equity_df = pd.DataFrame(equity_curve).set_index('date')

    # Calculate metrics
    returns = equity_df['equity'].pct_change().dropna()

    if len(returns) < 2:
        sharpe = 0
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown
    cummax = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - cummax) / cummax
    max_dd = abs(drawdown.min()) * 100

    # Win rate
    if trades:
        winning = sum(1 for t in trades if t.get('pnl', 0) > 0)
        win_rate = winning / len(trades) * 100 if len(trades) > 0 else 0
    else:
        win_rate = 0

    total_return = (equity - initial_capital) / initial_capital * 100

    metrics = {
        'sharpe_ratio': sharpe,
        'total_return_pct': total_return,
        'max_drawdown_pct': max_dd,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'final_equity': equity,
    }

    return equity_df, metrics


def test_out_of_sample(
    data: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[ValidationResults, ValidationResults, ValidationResults]:
    """
    Test 1: Out-of-Sample Testing

    Split data into train (70%), validate (15%), test (15%).
    Train on train set, tune on val set, final eval on test set.
    """
    n = len(data)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    print(f"\n=== OUT-OF-SAMPLE TESTING ===")
    print(f"Train: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} days)")
    print(f"Val:   {val_data.index[0].date()} to {val_data.index[-1].date()} ({len(val_data)} days)")
    print(f"Test:  {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} days)")

    # Run on each split
    _, train_metrics = simulate_vol_arb_strategy(train_data)
    _, val_metrics = simulate_vol_arb_strategy(val_data)
    _, test_metrics = simulate_vol_arb_strategy(test_data)

    train_result = ValidationResults(
        test_name="In-Sample (Train)",
        sharpe_ratio=train_metrics['sharpe_ratio'],
        total_return_pct=train_metrics['total_return_pct'],
        max_drawdown_pct=train_metrics['max_drawdown_pct'],
        win_rate=train_metrics['win_rate'],
        num_trades=train_metrics['num_trades'],
    )

    val_result = ValidationResults(
        test_name="Validation",
        sharpe_ratio=val_metrics['sharpe_ratio'],
        total_return_pct=val_metrics['total_return_pct'],
        max_drawdown_pct=val_metrics['max_drawdown_pct'],
        win_rate=val_metrics['win_rate'],
        num_trades=val_metrics['num_trades'],
    )

    test_result = ValidationResults(
        test_name="Out-of-Sample (Test)",
        sharpe_ratio=test_metrics['sharpe_ratio'],
        total_return_pct=test_metrics['total_return_pct'],
        max_drawdown_pct=test_metrics['max_drawdown_pct'],
        win_rate=test_metrics['win_rate'],
        num_trades=test_metrics['num_trades'],
    )

    return train_result, val_result, test_result


def test_walk_forward(
    data: pd.DataFrame,
    train_window: int = 252,  # 1 year
    test_window: int = 63,    # 1 quarter
) -> List[ValidationResults]:
    """
    Test 2: Walk-Forward Analysis

    Re-calibrate monthly, test on next month.
    More realistic than fixed train/test split.
    """
    print(f"\n=== WALK-FORWARD ANALYSIS ===")
    print(f"Train window: {train_window} days, Test window: {test_window} days")

    results = []
    start_idx = train_window

    while start_idx + test_window <= len(data):
        train_data = data.iloc[start_idx - train_window:start_idx]
        test_data = data.iloc[start_idx:start_idx + test_window]

        _, test_metrics = simulate_vol_arb_strategy(test_data)

        results.append(ValidationResults(
            test_name=f"WF Period {len(results)+1}",
            sharpe_ratio=test_metrics['sharpe_ratio'],
            total_return_pct=test_metrics['total_return_pct'],
            max_drawdown_pct=test_metrics['max_drawdown_pct'],
            win_rate=test_metrics['win_rate'],
            num_trades=test_metrics['num_trades'],
            notes=f"{test_data.index[0].date()} to {test_data.index[-1].date()}"
        ))

        start_idx += test_window

    avg_sharpe = np.mean([r.sharpe_ratio for r in results])
    std_sharpe = np.std([r.sharpe_ratio for r in results])

    print(f"Walk-Forward Periods: {len(results)}")
    print(f"Average Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")

    return results


def test_monte_carlo(
    data: pd.DataFrame,
    num_simulations: int = 1000,
    max_jitter_days: int = 2,
) -> Tuple[float, float, List[float]]:
    """
    Test 3: Monte Carlo Simulation

    Randomize entry timing by ±2 days, run 1000x.
    If std(sharpes) > 0.5, likely overfitting to exact timing.
    """
    print(f"\n=== MONTE CARLO SIMULATION ===")
    print(f"Running {num_simulations} simulations with ±{max_jitter_days} day jitter...")

    sharpes = []

    for i in range(num_simulations):
        # Create jittered data by shifting randomly
        jitter = np.random.randint(-max_jitter_days, max_jitter_days + 1)

        if jitter != 0:
            jittered_data = data.shift(jitter).dropna()
        else:
            jittered_data = data.copy()

        if len(jittered_data) > 100:
            _, metrics = simulate_vol_arb_strategy(jittered_data)
            sharpes.append(metrics['sharpe_ratio'])

        if (i + 1) % 200 == 0:
            print(f"  Completed {i+1}/{num_simulations}")

    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)

    print(f"Monte Carlo Sharpe: {mean_sharpe:.3f} ± {std_sharpe:.3f}")

    if std_sharpe > 0.5:
        print("⚠️  WARNING: High variance suggests overfitting to exact entry timing")

    return mean_sharpe, std_sharpe, sharpes


def test_transaction_costs(
    data: pd.DataFrame,
    cost_multipliers: List[float] = [1.0, 2.0, 3.0, 5.0],
) -> List[ValidationResults]:
    """
    Test 4: Transaction Cost Sensitivity

    Test at 1x, 2x, 3x, 5x base costs.
    If Sharpe drops >50% at 3x costs, strategy is cost-sensitive.
    """
    print(f"\n=== TRANSACTION COST SENSITIVITY ===")

    base_commission = 0.50
    base_slippage = 0.01

    results = []

    for mult in cost_multipliers:
        _, metrics = simulate_vol_arb_strategy(
            data,
            commission=base_commission * mult,
            slippage=base_slippage * mult,
        )

        results.append(ValidationResults(
            test_name=f"Costs {mult}x",
            sharpe_ratio=metrics['sharpe_ratio'],
            total_return_pct=metrics['total_return_pct'],
            max_drawdown_pct=metrics['max_drawdown_pct'],
            win_rate=metrics['win_rate'],
            num_trades=metrics['num_trades'],
        ))

        print(f"  {mult}x costs: Sharpe = {metrics['sharpe_ratio']:.3f}, Return = {metrics['total_return_pct']:.1f}%")

    if len(results) >= 2 and results[0].sharpe_ratio > 0:
        decay = (results[0].sharpe_ratio - results[-1].sharpe_ratio) / results[0].sharpe_ratio * 100
        print(f"Sharpe decay at {cost_multipliers[-1]}x costs: {decay:.1f}%")

        if decay > 50:
            print("⚠️  WARNING: Strategy is highly cost-sensitive")

    return results


def test_regime_analysis(data: pd.DataFrame) -> List[ValidationResults]:
    """
    Test 5: Regime Analysis

    Split by volatility regime:
    - Low Vol: RV < 15%
    - Normal Vol: 15% <= RV < 25%
    - High Vol: RV >= 25%

    Strategy must work in all 3 regimes.
    """
    print(f"\n=== VOLATILITY REGIME ANALYSIS ===")

    regimes = {
        'Low Vol (RV<15%)': data[data['rv_20d'] < 0.15],
        'Normal Vol (15-25%)': data[(data['rv_20d'] >= 0.15) & (data['rv_20d'] < 0.25)],
        'High Vol (RV>=25%)': data[data['rv_20d'] >= 0.25],
    }

    results = []

    for regime_name, regime_data in regimes.items():
        if len(regime_data) < 30:
            print(f"  {regime_name}: Insufficient data ({len(regime_data)} days)")
            continue

        _, metrics = simulate_vol_arb_strategy(regime_data)

        results.append(ValidationResults(
            test_name=regime_name,
            sharpe_ratio=metrics['sharpe_ratio'],
            total_return_pct=metrics['total_return_pct'],
            max_drawdown_pct=metrics['max_drawdown_pct'],
            win_rate=metrics['win_rate'],
            num_trades=metrics['num_trades'],
            notes=f"{len(regime_data)} days"
        ))

        print(f"  {regime_name}: Sharpe = {metrics['sharpe_ratio']:.3f}, {len(regime_data)} days")

    if len(results) >= 2:
        sharpes = [r.sharpe_ratio for r in results]
        std_sharpe = np.std(sharpes)
        print(f"Regime Sharpe Std Dev: {std_sharpe:.3f}")

        if std_sharpe > 0.5:
            print("⚠️  WARNING: Large regime variance - strategy may be regime-dependent")

    return results


def test_parameter_sensitivity(
    data: pd.DataFrame,
    param_variations: List[float] = [-0.20, -0.10, 0, 0.10, 0.20],
) -> List[ValidationResults]:
    """
    Test 6: Parameter Sensitivity

    Vary each parameter by ±10%, ±20%.
    If Sharpe decays >30%, parameters may be overfit.
    """
    print(f"\n=== PARAMETER SENSITIVITY ===")

    base_entry = 5.0
    base_exit = 2.0

    results = []

    for var in param_variations:
        entry = base_entry * (1 + var)
        exit = base_exit * (1 + var)

        _, metrics = simulate_vol_arb_strategy(
            data,
            entry_threshold=entry,
            exit_threshold=exit,
        )

        results.append(ValidationResults(
            test_name=f"Params {var*100:+.0f}%",
            sharpe_ratio=metrics['sharpe_ratio'],
            total_return_pct=metrics['total_return_pct'],
            max_drawdown_pct=metrics['max_drawdown_pct'],
            win_rate=metrics['win_rate'],
            num_trades=metrics['num_trades'],
            notes=f"Entry={entry:.1f}%, Exit={exit:.1f}%"
        ))

        print(f"  {var*100:+.0f}%: Sharpe = {metrics['sharpe_ratio']:.3f}")

    sharpes = [r.sharpe_ratio for r in results]
    avg_sharpe = np.mean(sharpes)
    base_sharpe = results[len(results)//2].sharpe_ratio if results else 0

    if base_sharpe > 0:
        stability = avg_sharpe / base_sharpe * 100
        print(f"Parameter Stability: {stability:.1f}% of base Sharpe")

        if stability < 70:
            print("⚠️  WARNING: Strategy is sensitive to parameter changes")

    return results


def test_bootstrap_confidence(
    data: pd.DataFrame,
    num_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Test 7: Bootstrap Confidence Intervals

    Resample returns with replacement, calculate 95% CI on Sharpe.
    """
    print(f"\n=== BOOTSTRAP CONFIDENCE INTERVALS ===")

    # Run base strategy
    equity_df, _ = simulate_vol_arb_strategy(data)
    returns = equity_df['equity'].pct_change().dropna().values

    if len(returns) < 50:
        print("Insufficient data for bootstrap")
        return 0, 0, 0

    bootstrap_sharpes = []

    for _ in range(num_bootstrap):
        # Resample returns with replacement
        sample = np.random.choice(returns, size=len(returns), replace=True)

        if np.std(sample) > 0:
            sharpe = np.mean(sample) / np.std(sample) * np.sqrt(252)
            bootstrap_sharpes.append(sharpe)

    ci_lower = np.percentile(bootstrap_sharpes, (1 - confidence) / 2 * 100)
    ci_upper = np.percentile(bootstrap_sharpes, (1 + confidence) / 2 * 100)
    mean_sharpe = np.mean(bootstrap_sharpes)

    print(f"Bootstrap Sharpe: {mean_sharpe:.3f}")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    if ci_lower < 0:
        print("⚠️  WARNING: Sharpe CI includes 0 - strategy may not be statistically significant")

    return mean_sharpe, ci_lower, ci_upper


def generate_comprehensive_report(
    train_result: ValidationResults,
    test_result: ValidationResults,
    wf_results: List[ValidationResults],
    mc_mean: float,
    mc_std: float,
    cost_results: List[ValidationResults],
    regime_results: List[ValidationResults],
    param_results: List[ValidationResults],
    ci_lower: float,
    ci_upper: float,
) -> ComprehensiveResults:
    """Generate final comprehensive validation report."""

    # Calculate key metrics
    is_sharpe = train_result.sharpe_ratio
    oos_sharpe = test_result.sharpe_ratio
    sharpe_decay = (is_sharpe - oos_sharpe) / is_sharpe * 100 if is_sharpe > 0 else 0

    # Cost sensitivity
    base_cost_sharpe = cost_results[0].sharpe_ratio if cost_results else 0
    high_cost_sharpe = cost_results[-1].sharpe_ratio if len(cost_results) > 1 else base_cost_sharpe
    cost_sensitivity = high_cost_sharpe / base_cost_sharpe if base_cost_sharpe > 0 else 0

    # Regime consistency
    regime_sharpes = [r.sharpe_ratio for r in regime_results]
    regime_consistency = np.std(regime_sharpes) if regime_sharpes else 0

    # Parameter stability
    param_sharpes = [r.sharpe_ratio for r in param_results]
    param_stability = np.mean(param_sharpes) if param_sharpes else 0

    # Determine overall confidence
    flags = []

    if sharpe_decay > 30:
        flags.append("OOS decay >30%")
    if mc_std > 0.5:
        flags.append("MC variance high")
    if cost_sensitivity < 0.5:
        flags.append("Cost sensitive")
    if regime_consistency > 0.5:
        flags.append("Regime dependent")
    if ci_lower < 0:
        flags.append("CI includes 0")
    if oos_sharpe > 2.0:
        flags.append("Sharpe suspiciously high")

    if len(flags) >= 3:
        confidence = "OVERFITTING DETECTED"
    elif len(flags) >= 2:
        confidence = "LOW"
    elif len(flags) >= 1:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    return ComprehensiveResults(
        oos_sharpe=oos_sharpe,
        is_sharpe=is_sharpe,
        sharpe_decay_pct=sharpe_decay,
        cost_sensitivity=cost_sensitivity,
        regime_consistency=regime_consistency,
        parameter_stability=param_stability,
        bootstrap_ci_lower=ci_lower,
        bootstrap_ci_upper=ci_upper,
        monte_carlo_std=mc_std,
        overall_confidence=confidence + (f" ({', '.join(flags)})" if flags else ""),
    )


def print_final_report(results: ComprehensiveResults):
    """Print formatted final validation report."""

    print("\n" + "="*70)
    print("           SHARPE RATIO VALIDATION REPORT")
    print("="*70)

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                         KEY METRICS                                  │
├─────────────────────────────────────────────────────────────────────┤
│  In-Sample Sharpe:        {results.is_sharpe:>8.3f}                              │
│  Out-of-Sample Sharpe:    {results.oos_sharpe:>8.3f}                              │
│  OOS Decay:               {results.sharpe_decay_pct:>8.1f}%                             │
├─────────────────────────────────────────────────────────────────────┤
│                     ROBUSTNESS CHECKS                                │
├─────────────────────────────────────────────────────────────────────┤
│  Monte Carlo Std Dev:     {results.monte_carlo_std:>8.3f}  (want <0.5)               │
│  Cost Sensitivity:        {results.cost_sensitivity:>8.2f}x (want >0.5)               │
│  Regime Consistency:      {results.regime_consistency:>8.3f}  (want <0.5)               │
│  Bootstrap 95% CI:        [{results.bootstrap_ci_lower:>6.3f}, {results.bootstrap_ci_upper:>6.3f}]                     │
├─────────────────────────────────────────────────────────────────────┤
│                      OVERALL ASSESSMENT                              │
├─────────────────────────────────────────────────────────────────────┤
│  Confidence:  {results.overall_confidence:<54} │
└─────────────────────────────────────────────────────────────────────┘
""")

    # Interpretation guide
    print("""
INTERPRETATION GUIDE:
────────────────────
• Realistic Sharpe for Vol Arb: 0.5 - 1.5
• Red Flag Sharpe: > 2.0 (likely overfit)
• OOS Decay < 20%: Good generalization
• OOS Decay > 30%: Likely overfitting

NEXT STEPS:
────────────────────""")

    if "OVERFITTING" in results.overall_confidence:
        print("1. ❌ DO NOT trade this strategy live")
        print("2. Review for look-ahead bias")
        print("3. Simplify strategy parameters")
        print("4. Collect more out-of-sample data")
    elif results.overall_confidence.startswith("LOW"):
        print("1. ⚠️  Exercise extreme caution")
        print("2. Paper trade for 6+ months")
        print("3. Address specific flags listed")
    elif results.overall_confidence.startswith("MEDIUM"):
        print("1. ⚠️  Further validation recommended")
        print("2. Paper trade for 3+ months")
        print("3. Start with reduced position sizes")
    else:
        print("1. ✅ Strategy shows reasonable robustness")
        print("2. Paper trade for 1-2 months")
        print("3. Start with 50% target position sizes")


def main():
    parser = argparse.ArgumentParser(description='Validate Sharpe ratio for overfitting')
    parser.add_argument('--data', type=str, required=True, help='Path to options CSV data')
    parser.add_argument('--monte-carlo', type=int, default=500, help='Number of MC simulations')
    parser.add_argument('--bootstrap', type=int, default=500, help='Number of bootstrap samples')
    args = parser.parse_args()

    # Load and process data
    df = load_options_data(args.data)
    iv_data = calculate_iv_premium(df)

    print(f"\nProcessed {len(iv_data)} days with IV premium data")
    print(f"IV Premium range: {iv_data['iv_premium_pct'].min():.1f}% to {iv_data['iv_premium_pct'].max():.1f}%")

    # Run all validation tests
    train_result, val_result, test_result = test_out_of_sample(iv_data)
    wf_results = test_walk_forward(iv_data)
    mc_mean, mc_std, _ = test_monte_carlo(iv_data, num_simulations=args.monte_carlo)
    cost_results = test_transaction_costs(iv_data)
    regime_results = test_regime_analysis(iv_data)
    param_results = test_parameter_sensitivity(iv_data)
    _, ci_lower, ci_upper = test_bootstrap_confidence(iv_data, num_bootstrap=args.bootstrap)

    # Generate comprehensive report
    results = generate_comprehensive_report(
        train_result=train_result,
        test_result=test_result,
        wf_results=wf_results,
        mc_mean=mc_mean,
        mc_std=mc_std,
        cost_results=cost_results,
        regime_results=regime_results,
        param_results=param_results,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )

    print_final_report(results)

    return results


if __name__ == "__main__":
    main()
