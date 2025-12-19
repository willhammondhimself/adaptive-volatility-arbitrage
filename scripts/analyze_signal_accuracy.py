#!/usr/bin/env python3
"""
Signal Accuracy Analysis for QV 6-Signal Strategy.

Analyzes which signals are most predictive of profitable trades.
Used to inform signal weight optimization.
"""

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from volatility_arbitrage.core.config import load_strategy_config

# Import from run_backtest
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_backtest import load_json_options_data, QVSignalCalculator


def run_signal_attribution(
    options_df: pd.DataFrame,
    config,
    initial_capital: float = 100000,
    config_path: str = None,
    end_date: str = None,
) -> dict:
    """
    Run backtest capturing detailed signal information for each trade.

    Returns dict with:
    - signal_stats: Per-signal accuracy metrics
    - trade_signals: Raw signal data per trade
    - correlations: Signal correlation matrix
    """
    import yaml

    print("\n" + "="*60)
    print("SIGNAL ATTRIBUTION ANALYSIS")
    print("="*60)

    # Filter by end date if specified
    if end_date:
        end_dt = pd.to_datetime(end_date)
        options_df = options_df[options_df['date'] <= end_dt]
        print(f"Filtered to end date: {end_date}")

    # Parameters
    position_size_base = float(config.position_size_pct) / 100
    option_spread_cost = 0.01
    commission_per_contract = 0.65
    min_holding_days = config.min_holding_days

    # Load raw config for risk management settings
    if config_path is None:
        config_path = 'config/volatility_arb.yaml'
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    strategy_cfg = raw_config.get('strategy', {})

    # Initialize
    qv_calc = QVSignalCalculator(config)
    equity = initial_capital
    position = 0
    entry_date = None
    entry_equity = None
    entry_iv = None
    entry_signals = {}
    entry_iv_premium_z = None
    prev_iv = None

    # Track trades with signal details
    trades_with_signals = []
    position_pnl = 0
    entry_position_size = 0

    # Pre-calculate spot and RV
    dates = sorted(options_df['date'].unique())
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Trading days: {len(dates)}")

    spot_prices = []
    for date in dates:
        day_df = options_df[options_df['date'] == date]
        calls = day_df[day_df['type'] == 'call'].copy()
        if len(calls) > 0:
            calls['dte'] = (calls['expiration'] - date).dt.days
            atm_calls = calls[(calls['dte'] >= 25) & (calls['dte'] <= 35)]
            if len(atm_calls) > 0:
                atm_idx = (atm_calls['delta'].abs() - 0.5).abs().argmin()
                atm_call = atm_calls.iloc[atm_idx]
                spot = atm_call['strike']
                spot_prices.append({'date': date, 'spot': spot})

    spot_df = pd.DataFrame(spot_prices).set_index('date')
    spot_df['returns'] = spot_df['spot'].pct_change()
    spot_df['rv_20d'] = spot_df['returns'].rolling(20).std() * np.sqrt(252)
    spot_df['rv_20d_lagged'] = spot_df['rv_20d'].shift(1)

    warmup_days = max(config.feature_window + 21, 80)
    print(f"Warmup period: {warmup_days} days")

    for i, date in enumerate(dates[warmup_days:], warmup_days):
        day_df = options_df[options_df['date'] == date]

        if date not in spot_df.index:
            continue

        rv = spot_df.loc[date, 'rv_20d_lagged']
        spot = spot_df.loc[date, 'spot']

        if pd.isna(rv):
            continue

        # Extract features and signals
        features = qv_calc.extract_daily_features(day_df, spot, rv)
        qv_calc.update_buffers(features)
        signals = qv_calc.generate_signals(features)
        consensus = qv_calc.calculate_consensus(signals)
        regime_scalar = qv_calc.get_regime_scalar(rv)

        atm_iv = features['atm_iv']

        # Calculate z-scores for each signal buffer
        z_scores = {}
        if len(qv_calc.pc_ratio_buffer) >= 20:
            z_scores['pc_ratio'] = qv_calc.calculate_z_score(
                qv_calc.pc_ratio_buffer, features['pc_ratio']
            )
            z_scores['iv_skew'] = qv_calc.calculate_z_score(
                qv_calc.iv_skew_buffer, features['iv_skew']
            )
            z_scores['iv_premium'] = qv_calc.calculate_z_score(
                qv_calc.iv_premium_buffer, features['iv_premium']
            )
            z_scores['term_structure'] = qv_calc.calculate_z_score(
                qv_calc.term_structure_buffer, features['term_structure']
            )
            z_scores['volume'] = qv_calc.calculate_z_score(
                qv_calc.volume_buffer, features['total_volume']
            )
            z_scores['sentiment'] = qv_calc.calculate_z_score(
                qv_calc.sentiment_buffer, features['sentiment']
            )

        # Daily P&L for existing position
        daily_pnl = 0
        if position != 0 and prev_iv is not None:
            iv_change = atm_iv - prev_iv
            dte_years = 45 / 365
            vega_per_straddle = 0.8 * spot * np.sqrt(dte_years)
            pnl_position_size = entry_position_size if entry_position_size > 0 else position_size_base
            notional_value = pnl_position_size * entry_equity
            avg_straddle_premium = spot * atm_iv * np.sqrt(dte_years) * 0.8
            num_straddles = notional_value / (avg_straddle_premium * 100) if avg_straddle_premium > 0 else 0
            vega_pnl = position * vega_per_straddle * (iv_change * 100) * num_straddles
            theta_daily = -avg_straddle_premium * 100 / 45
            theta_pnl = -position * abs(theta_daily) * num_straddles
            daily_pnl = vega_pnl + theta_pnl

        days_held = 0
        if entry_date is not None:
            days_held = (pd.Timestamp(date) - pd.Timestamp(entry_date)).days

        # RV percentile for regime detection
        rv_percentile = 0.5
        if len(qv_calc.rv_history) >= 20:
            rv_arr = np.array(qv_calc.rv_history)
            rv_percentile = (rv_arr < rv).sum() / len(rv_arr)

        # Z-score for IV premium
        iv_premium_z = 0.0
        if len(qv_calc.iv_premium_buffer) >= 21:
            iv_arr = np.array(list(qv_calc.iv_premium_buffer)[:-1])
            iv_mean = iv_arr.mean()
            iv_std = iv_arr.std()
            if iv_std > 1e-8:
                iv_premium_z = (features['iv_premium'] - iv_mean) / iv_std

        # Entry logic (simplified - no risk management)
        is_crisis_regime = rv_percentile > 0.85
        is_elevated_regime = rv_percentile > 0.70
        is_stressed_regime = rv_percentile > 0.50
        is_low_vol_regime = rv_percentile < 0.30

        if is_crisis_regime:
            z_threshold = 3.0
        elif is_elevated_regime:
            z_threshold = 2.5
        elif is_stressed_regime:
            z_threshold = 2.0
        elif is_low_vol_regime:
            z_threshold = 1.25
        else:
            z_threshold = 1.5

        if position == 0:
            short_vol_entry = iv_premium_z > z_threshold
            long_vol_entry = iv_premium_z < -z_threshold

            if short_vol_entry or long_vol_entry:
                position = -1 if short_vol_entry else 1
                entry_date = date
                entry_equity = equity
                entry_iv = atm_iv
                entry_signals = signals.copy()
                entry_iv_premium_z = iv_premium_z
                entry_position_size = position_size_base * regime_scalar
                position_pnl = 0

                cost = option_spread_cost * entry_position_size * equity + commission_per_contract * 2
                equity -= cost

        elif position != 0:
            should_exit = False
            exit_reason = ""

            position_pnl += daily_pnl
            position_return = position_pnl / entry_equity if entry_equity else 0

            # Profit/stop
            if position_return >= 0.10:
                should_exit = True
                exit_reason = "PROFIT_TARGET"
            elif position_return <= -0.10:
                should_exit = True
                exit_reason = "STOP_LOSS"
            elif days_held >= min_holding_days:
                if position == -1 and iv_premium_z < 0:
                    should_exit = True
                    exit_reason = "IV_REVERSION"
                elif position == 1 and iv_premium_z > 0:
                    should_exit = True
                    exit_reason = "IV_REVERSION"
            if days_held >= 30:
                should_exit = True
                exit_reason = "MAX_HOLDING"

            if should_exit:
                cost = option_spread_cost * entry_position_size * equity + commission_per_contract * 2
                equity -= cost

                # Record trade with signal details
                trades_with_signals.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'position': 'SHORT_VOL' if position == -1 else 'LONG_VOL',
                    'position_return': position_return * 100,
                    'is_winner': position_return > 0,
                    'exit_reason': exit_reason,
                    'days_held': days_held,
                    'entry_iv_premium_z': entry_iv_premium_z,
                    # Signal values at entry
                    'signal_pc_ratio': entry_signals.get('pc_ratio', 0),
                    'signal_iv_skew': entry_signals.get('iv_skew', 0),
                    'signal_iv_premium': entry_signals.get('iv_premium', 0),
                    'signal_term_structure': entry_signals.get('term_structure', 0),
                    'signal_volume': entry_signals.get('volume', 0),
                    'signal_sentiment': entry_signals.get('sentiment', 0),
                })

                position = 0
                entry_date = None
                entry_equity = None
                entry_signals = {}
                entry_iv_premium_z = None
                position_pnl = 0

        equity += daily_pnl
        prev_iv = atm_iv

    # Calculate signal statistics
    trades_df = pd.DataFrame(trades_with_signals)

    if len(trades_df) == 0:
        print("No trades found!")
        return {'signal_stats': {}, 'trade_signals': [], 'correlations': None}

    print(f"\nTotal trades analyzed: {len(trades_df)}")
    print(f"Winners: {trades_df['is_winner'].sum()} ({trades_df['is_winner'].mean()*100:.1f}%)")
    print(f"Avg return: {trades_df['position_return'].mean():.2f}%")

    # Per-signal analysis
    signal_names = ['pc_ratio', 'iv_skew', 'iv_premium', 'term_structure', 'volume', 'sentiment']
    signal_stats = {}

    print("\n" + "="*60)
    print("SIGNAL ACCURACY ANALYSIS")
    print("="*60)
    print(f"\n{'Signal':<18} {'Weight':>8} {'Hit Rate':>10} {'Avg Ret':>10} {'Contrib':>10} {'Rec':>10}")
    print("-"*68)

    weights = {
        'pc_ratio': float(config.weight_pc_ratio),
        'iv_skew': float(config.weight_iv_skew),
        'iv_premium': float(config.weight_iv_premium),
        'term_structure': float(config.weight_term_structure),
        'volume': float(config.weight_volume_spike),
        'sentiment': float(config.weight_near_term_sentiment),
    }

    for signal in signal_names:
        col = f'signal_{signal}'

        # Hit rate: signal direction matched profitable trade direction
        # For SHORT_VOL trades: signal=-1 and winner → correct
        # For LONG_VOL trades: signal=+1 and winner → correct
        correct = 0
        total_active = 0
        returns_when_active = []

        for _, row in trades_df.iterrows():
            sig_val = row[col]
            if sig_val != 0:
                total_active += 1
                is_short = row['position'] == 'SHORT_VOL'
                is_long = row['position'] == 'LONG_VOL'

                # Signal alignment: -1 signal aligns with SHORT_VOL, +1 with LONG_VOL
                aligned = (sig_val == -1 and is_short) or (sig_val == 1 and is_long)

                if aligned and row['is_winner']:
                    correct += 1
                elif not aligned and not row['is_winner']:
                    correct += 1  # Wrong signal, losing trade = signal was right to not be aligned

                returns_when_active.append(row['position_return'])

        hit_rate = correct / total_active * 100 if total_active > 0 else 50.0
        avg_return = np.mean(returns_when_active) if returns_when_active else 0

        # Contribution score: hit_rate * |avg_return| * activation_rate
        activation_rate = total_active / len(trades_df) if len(trades_df) > 0 else 0
        contrib_score = (hit_rate / 100) * abs(avg_return) * activation_rate

        # Recommendation
        current_weight = weights[signal]
        if hit_rate > 60:
            rec = "INCREASE"
        elif hit_rate < 45:
            rec = "DECREASE"
        else:
            rec = "KEEP"

        signal_stats[signal] = {
            'weight': current_weight,
            'hit_rate': hit_rate,
            'avg_return': avg_return,
            'activation_rate': activation_rate * 100,
            'contribution': contrib_score,
            'recommendation': rec,
        }

        print(f"{signal:<18} {current_weight:>8.2f} {hit_rate:>9.1f}% {avg_return:>9.2f}% {contrib_score:>9.2f} {rec:>10}")

    # Signal correlation matrix
    signal_cols = [f'signal_{s}' for s in signal_names]
    correlations = trades_df[signal_cols].corr()

    print("\n" + "="*60)
    print("SIGNAL CORRELATIONS")
    print("="*60)
    print("\n(High correlation = signals are redundant)")
    for i, sig1 in enumerate(signal_names):
        for j, sig2 in enumerate(signal_names):
            if i < j:
                corr = correlations.loc[f'signal_{sig1}', f'signal_{sig2}']
                if abs(corr) > 0.3:
                    print(f"  {sig1} <-> {sig2}: {corr:.2f}")

    # Optimal weights suggestion
    print("\n" + "="*60)
    print("SUGGESTED WEIGHT REBALANCING")
    print("="*60)

    # Calculate new weights based on contribution scores
    total_contrib = sum(s['contribution'] for s in signal_stats.values())
    if total_contrib > 0:
        suggested_weights = {}
        for signal, stats in signal_stats.items():
            # Blend current weight with contribution-based weight
            contrib_weight = stats['contribution'] / total_contrib
            suggested_weights[signal] = 0.5 * weights[signal] + 0.5 * contrib_weight

        # Normalize to sum to 1.0
        total_suggested = sum(suggested_weights.values())
        for signal in suggested_weights:
            suggested_weights[signal] /= total_suggested

        print(f"\n{'Signal':<18} {'Current':>10} {'Suggested':>10} {'Change':>10}")
        print("-"*50)
        for signal in signal_names:
            current = weights[signal]
            suggested = suggested_weights[signal]
            change = (suggested - current) / current * 100 if current > 0 else 0
            print(f"{signal:<18} {current:>10.2f} {suggested:>10.3f} {change:>+9.1f}%")

    return {
        'signal_stats': signal_stats,
        'trade_signals': trades_with_signals,
        'correlations': correlations,
        'suggested_weights': suggested_weights if total_contrib > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze signal accuracy for QV strategy')
    parser.add_argument(
        '--config',
        type=str,
        default='config/volatility_arb.yaml',
        help='Strategy config file',
    )
    parser.add_argument(
        '--data',
        type=str,
        default='src/volatility_arbitrage/data/SPY_Options_2019_24',
        help='Options data directory',
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2022-12-31',
        help='End date for analysis (training period only)',
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/signal_accuracy.csv',
        help='Output CSV path',
    )
    args = parser.parse_args()

    print("="*60)
    print("SIGNAL ACCURACY ANALYSIS")
    print("="*60)

    # Load config
    print(f"\nLoading config from {args.config}...")
    config = load_strategy_config(Path(args.config))

    # Load options data
    print(f"Loading options data from {args.data}...")
    options_df = load_json_options_data(args.data)
    print(f"  Records: {len(options_df):,}")
    print(f"  Date range: {options_df['date'].min()} to {options_df['date'].max()}")

    # Run analysis
    results = run_signal_attribution(
        options_df,
        config,
        args.capital,
        config_path=args.config,
        end_date=args.end_date,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save signal stats
    stats_df = pd.DataFrame(results['signal_stats']).T
    stats_df.to_csv(output_path)
    print(f"\nSignal stats saved to: {output_path}")

    # Save trade-level data
    trades_path = output_path.parent / 'trade_signals.csv'
    trades_df = pd.DataFrame(results['trade_signals'])
    trades_df.to_csv(trades_path, index=False)
    print(f"Trade signals saved to: {trades_path}")

    # Save suggested weights
    if results.get('suggested_weights'):
        weights_path = output_path.parent / 'suggested_weights.json'
        with open(weights_path, 'w') as f:
            json.dump(results['suggested_weights'], f, indent=2)
        print(f"Suggested weights saved to: {weights_path}")


if __name__ == "__main__":
    main()
