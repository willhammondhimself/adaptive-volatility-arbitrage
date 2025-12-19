#!/usr/bin/env python3
"""
Run full backtest with volatility arbitrage strategy.

Supports both:
- Simple IV-RV spread strategy (legacy)
- QV 6-signal consensus strategy (recommended)
"""

import argparse
import json
from collections import deque
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import numpy as np

from volatility_arbitrage.backtest.metrics import calculate_sharpe_ratio
from volatility_arbitrage.core.config import load_strategy_config
from volatility_arbitrage.risk.drawdown_manager import DrawdownManager, DrawdownConfig
from volatility_arbitrage.risk.position_sizing import KellyPositionSizer, KellyConfig
from volatility_arbitrage.strategy.veto_manager import VetoManager, VetoConfig

# Strategy Enhancements (Ablation Study)
from volatility_arbitrage.strategy.enhancements import (
    RegimeTransitionConfig, RegimeTransitionSignal, integrate_regime_signal,
    TermStructureLeverageConfig, TermStructureLeverageCalculator,
    VoVConfig, VoVSignalGenerator,
    IntradayVolConfig, IntradayVolCalculator,
    DynamicWeightingConfig, DynamicSignalWeighter, SignalOutcome,
    AsymmetricProfitConfig, AsymmetricProfitManager,
)
from volatility_arbitrage.strategy.enhancements.alt_rv_methods import calculate_rv, ensemble_rv


def load_json_options_data(data_dir: str) -> pd.DataFrame:
    """Load JSON options data from directory."""
    data_path = Path(data_dir)
    all_records = []

    json_files = sorted(data_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        print(f"Loading {json_file.name}...")
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Handle nested list structure
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                for day_records in data:
                    all_records.extend(day_records)
            else:
                all_records.extend(data)

    print(f"Total records loaded: {len(all_records):,}")

    df = pd.DataFrame(all_records)

    # Convert types
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    for col in ['strike', 'last', 'bid', 'ask', 'mark', 'volume', 'open_interest',
                'implied_volatility', 'delta', 'gamma', 'theta', 'vega']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


class QVSignalCalculator:
    """
    Calculate 6-signal QV consensus for volatility arbitrage.

    Signals:
    1. PC Ratio (Put/Call volume ratio) - contrarian
    2. IV Skew (ATM put IV - call IV) - fear premium
    3. IV Premium (IV percentile vs history) - mean reversion
    4. Term Structure (60d IV - 30d IV) - contango/backwardation
    5. Volume Spike (current vol / median) - momentum
    6. Near-term Sentiment (7d IV - 30d IV) - short-term fear
    """

    def __init__(self, config):
        self.config = config
        self.feature_window = config.feature_window

        # Rolling feature buffers
        self.pc_ratio_buffer = deque(maxlen=self.feature_window)
        self.iv_skew_buffer = deque(maxlen=self.feature_window)
        self.iv_premium_buffer = deque(maxlen=self.feature_window)
        self.term_structure_buffer = deque(maxlen=self.feature_window)
        self.volume_buffer = deque(maxlen=self.feature_window)
        self.sentiment_buffer = deque(maxlen=self.feature_window)

        # RV history for percentile calculation
        self.rv_history = deque(maxlen=config.regime_window)

    def extract_daily_features(self, day_df: pd.DataFrame, spot: float, rv: float) -> dict:
        """Extract all 6 QV features from daily options data."""
        calls = day_df[day_df['type'] == 'call'].copy()
        puts = day_df[day_df['type'] == 'put'].copy()

        today = day_df['date'].iloc[0]
        calls['dte'] = (calls['expiration'] - today).dt.days
        puts['dte'] = (puts['expiration'] - today).dt.days

        features = {}

        # 1. PC Ratio (Put volume / Call volume)
        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()
        features['pc_ratio'] = put_volume / call_volume if call_volume > 0 else 1.0

        # 2. IV Skew (ATM put IV - ATM call IV)
        atm_calls_30 = calls[(calls['dte'] >= 25) & (calls['dte'] <= 35)]
        atm_puts_30 = puts[(puts['dte'] >= 25) & (puts['dte'] <= 35)]

        if len(atm_calls_30) > 0 and len(atm_puts_30) > 0:
            atm_call = atm_calls_30.iloc[(atm_calls_30['delta'].abs() - 0.5).abs().argmin()]
            atm_put = atm_puts_30.iloc[(atm_puts_30['delta'].abs() + 0.5).abs().argmin()]
            features['iv_skew'] = abs(atm_put['implied_volatility']) - atm_call['implied_volatility']
        else:
            features['iv_skew'] = 0.0

        # 3. IV Premium (current IV percentile vs 60-day history)
        if len(atm_calls_30) > 0:
            atm_iv = atm_calls_30.iloc[(atm_calls_30['delta'].abs() - 0.5).abs().argmin()]['implied_volatility']
        else:
            atm_iv = rv  # Fallback
        features['atm_iv'] = atm_iv
        features['iv_premium'] = (atm_iv - rv) / rv if rv > 0 else 0.0

        # 4. Term Structure (60d IV - 30d IV)
        atm_calls_60 = calls[(calls['dte'] >= 55) & (calls['dte'] <= 65)]
        if len(atm_calls_30) > 0 and len(atm_calls_60) > 0:
            iv_30 = atm_calls_30.iloc[(atm_calls_30['delta'].abs() - 0.5).abs().argmin()]['implied_volatility']
            iv_60 = atm_calls_60.iloc[(atm_calls_60['delta'].abs() - 0.5).abs().argmin()]['implied_volatility']
            features['term_structure'] = iv_60 - iv_30
        else:
            features['term_structure'] = 0.0

        # 5. Volume Spike (total volume)
        features['total_volume'] = call_volume + put_volume

        # 6. Near-term Sentiment (7d IV - 30d IV)
        atm_calls_7 = calls[(calls['dte'] >= 5) & (calls['dte'] <= 10)]
        if len(atm_calls_7) > 0 and len(atm_calls_30) > 0:
            iv_7 = atm_calls_7.iloc[(atm_calls_7['delta'].abs() - 0.5).abs().argmin()]['implied_volatility']
            iv_30 = atm_calls_30.iloc[(atm_calls_30['delta'].abs() - 0.5).abs().argmin()]['implied_volatility']
            features['sentiment'] = iv_7 - iv_30
        else:
            features['sentiment'] = 0.0

        return features

    def update_buffers(self, features: dict):
        """Update rolling feature buffers."""
        self.pc_ratio_buffer.append(features['pc_ratio'])
        self.iv_skew_buffer.append(features['iv_skew'])
        self.iv_premium_buffer.append(features['iv_premium'])
        self.term_structure_buffer.append(features['term_structure'])
        self.volume_buffer.append(features['total_volume'])
        self.sentiment_buffer.append(features['sentiment'])

    def calculate_z_score(self, buffer: deque, current_value: float) -> float:
        """Calculate z-score of current value vs buffer history."""
        if len(buffer) < 20:
            return 0.0
        arr = np.array(buffer)
        mean = arr.mean()
        std = arr.std()
        if std < 1e-8:
            return 0.0
        return (current_value - mean) / std

    def generate_signals(self, features: dict) -> dict:
        """
        Generate binary signals (-1, 0, +1) for each feature.

        +1 = bullish signal (go long vol or reduce short)
        -1 = bearish signal (go short vol or reduce long)
        0 = neutral
        """
        signals = {}

        # Need sufficient history
        if len(self.pc_ratio_buffer) < 20:
            return {k: 0 for k in ['pc_ratio', 'iv_skew', 'iv_premium', 'term_structure', 'volume', 'sentiment']}

        # 1. PC Ratio - high PC ratio (fear) is contrarian bullish
        pc_z = self.calculate_z_score(self.pc_ratio_buffer, features['pc_ratio'])
        if pc_z > float(self.config.pc_ratio_threshold):
            signals['pc_ratio'] = 1  # Bullish (fear overdone)
        elif pc_z < -float(self.config.pc_ratio_threshold):
            signals['pc_ratio'] = -1  # Bearish (complacency)
        else:
            signals['pc_ratio'] = 0

        # 2. IV Skew - high put skew (fear) is contrarian bullish
        skew_z = self.calculate_z_score(self.iv_skew_buffer, features['iv_skew'])
        if skew_z > float(self.config.skew_threshold):
            signals['iv_skew'] = 1  # Bullish
        elif skew_z < -float(self.config.skew_threshold):
            signals['iv_skew'] = -1  # Bearish
        else:
            signals['iv_skew'] = 0

        # 3. IV Premium - high IV premium suggests short vol
        if features['iv_premium'] > float(self.config.premium_threshold):
            signals['iv_premium'] = -1  # Short vol (IV too high)
        elif features['iv_premium'] < -float(self.config.premium_threshold):
            signals['iv_premium'] = 1  # Long vol (IV too low)
        else:
            signals['iv_premium'] = 0

        # 4. Term Structure - positive slope (contango) is bullish
        term_z = self.calculate_z_score(self.term_structure_buffer, features['term_structure'])
        if term_z > float(self.config.term_structure_threshold):
            signals['term_structure'] = 1  # Bullish
        elif term_z < -float(self.config.term_structure_threshold):
            signals['term_structure'] = -1  # Bearish
        else:
            signals['term_structure'] = 0

        # 5. Volume Spike - high volume with direction
        vol_z = self.calculate_z_score(self.volume_buffer, features['total_volume'])
        if vol_z > float(self.config.volume_spike_threshold):
            signals['volume'] = 1  # Continuation bullish
        elif vol_z < -0.5:  # Lower threshold for low volume
            signals['volume'] = -1
        else:
            signals['volume'] = 0

        # 6. Sentiment - negative sentiment is contrarian bullish
        sent_z = self.calculate_z_score(self.sentiment_buffer, features['sentiment'])
        if sent_z > float(self.config.sentiment_threshold):
            signals['sentiment'] = -1  # Near-term fear, bearish vol
        elif sent_z < -float(self.config.sentiment_threshold):
            signals['sentiment'] = 1  # Complacency, bullish vol
        else:
            signals['sentiment'] = 0

        return signals

    def calculate_consensus(self, signals: dict) -> float:
        """
        Calculate weighted consensus score from all signals.

        Returns: float in [-1.0, 1.0]
        - Positive = bullish vol (go long)
        - Negative = bearish vol (go short)
        """
        weights = {
            'pc_ratio': float(self.config.weight_pc_ratio),
            'iv_skew': float(self.config.weight_iv_skew),
            'iv_premium': float(self.config.weight_iv_premium),
            'term_structure': float(self.config.weight_term_structure),
            'volume': float(self.config.weight_volume_spike),
            'sentiment': float(self.config.weight_near_term_sentiment),
        }

        consensus = sum(signals[k] * weights[k] for k in signals)
        return consensus

    def get_regime_scalar(self, rv: float) -> float:
        """Get position size multiplier based on vol percentile."""
        self.rv_history.append(rv)

        if len(self.rv_history) < 20:
            return 1.0

        rv_arr = np.array(self.rv_history)
        percentile = (rv_arr < rv).sum() / len(rv_arr)

        if percentile > 0.90:
            return float(self.config.regime_crisis_scalar)
        elif percentile > 0.70:
            return float(self.config.regime_elevated_scalar)
        elif percentile > 0.30:
            return float(self.config.regime_normal_scalar)
        elif percentile > 0.10:
            return float(self.config.regime_low_scalar)
        else:
            return float(self.config.regime_extreme_low_scalar)


def run_qv_backtest(options_df: pd.DataFrame, config, initial_capital: float = 100000, config_path: str = None):
    """
    Run QV 6-signal consensus backtest with improved entry/exit logic.

    Key improvements:
    1. Asymmetric entry thresholds (short vol has better edge)
    2. Term structure confirmation (contango = sell vol, backwardation = buy vol)
    3. Profit targets and trailing stops
    4. Regime-aware position sizing

    Args:
        options_df: Options data
        config: Strategy config object
        initial_capital: Starting capital
        config_path: Path to YAML config file (optional, uses config._config_path if available)
    """
    print("\n" + "="*60)
    print("QV ENHANCED STRATEGY BACKTEST (v2.0 - Risk Management)")
    print("="*60)

    # Parameters from config
    position_size_base = float(config.position_size_pct) / 100
    # Use realistic institutional spreads (1% for very liquid SPY ATM options)
    option_spread_cost = 0.01  # 1% bid-ask spread (tight market)
    commission_per_contract = 0.65
    min_holding_days = config.min_holding_days
    consensus_threshold = float(config.consensus_threshold)

    # ===== NEW: Initialize Risk Management Components =====
    # Load raw YAML for new risk management parameters (not in dataclass yet)
    # Use provided config_path, or try to get from config object, or default
    import yaml
    if config_path is None and hasattr(config, '_config_path'):
        config_path = config._config_path
    if config_path is None:
        config_path = 'config/volatility_arb.yaml'
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    strategy_cfg = raw_config.get('strategy', {})

    # Drawdown Manager
    use_drawdown_recovery = strategy_cfg.get('use_drawdown_recovery', True)
    if use_drawdown_recovery:
        dd_thresholds = strategy_cfg.get('drawdown_thresholds', [
            [0.02, 1.0], [0.04, 0.75], [0.06, 0.50], [0.08, 0.25], [0.10, 0.10]
        ])
        drawdown_config = DrawdownConfig(
            thresholds=[tuple(t) for t in dd_thresholds],
            halt_threshold=strategy_cfg.get('drawdown_halt_threshold', 0.12),
        )
        drawdown_manager = DrawdownManager(drawdown_config)
        print(f"\n  Drawdown Recovery: ENABLED (halt at {drawdown_config.halt_threshold*100:.0f}%)")
    else:
        drawdown_manager = None
        print(f"\n  Drawdown Recovery: DISABLED")

    # Kelly Position Sizer
    use_kelly_sizing = strategy_cfg.get('use_kelly_sizing', True)
    if use_kelly_sizing:
        kelly_config = KellyConfig(
            kelly_fraction=strategy_cfg.get('kelly_fraction', 0.25),
            lookback_trades=strategy_cfg.get('kelly_lookback_trades', 50),
            min_trades_for_kelly=strategy_cfg.get('kelly_min_trades', 20),
            max_position_pct=strategy_cfg.get('kelly_max_position_pct', 0.15),
            min_position_pct=strategy_cfg.get('kelly_min_position_pct', 0.01),
        )
        kelly_sizer = KellyPositionSizer(kelly_config)
        print(f"  Kelly Sizing: ENABLED ({kelly_config.kelly_fraction*100:.0f}% fractional)")
    else:
        kelly_sizer = None
        print(f"  Kelly Sizing: DISABLED")

    # Signal Veto Manager
    use_signal_veto = strategy_cfg.get('use_signal_veto', True)
    if use_signal_veto:
        veto_config = VetoConfig(
            vix_extreme_threshold=strategy_cfg.get('veto_vix_extreme_threshold', 40.0),
            vix_spike_threshold=strategy_cfg.get('veto_vix_spike_threshold', 0.30),
            vix_spike_lookback=strategy_cfg.get('veto_vix_spike_lookback', 5),
            backwardation_threshold=strategy_cfg.get('veto_backwardation_threshold', -0.15),
            signal_disagreement_threshold=strategy_cfg.get('veto_signal_disagreement', 3),
            regime_crisis_percentile=strategy_cfg.get('veto_regime_crisis_percentile', 0.90),
        )
        veto_manager = VetoManager(veto_config)
        print(f"  Signal Veto: ENABLED (VIX >{veto_config.vix_extreme_threshold}, crisis >{veto_config.regime_crisis_percentile*100:.0f}%ile)")
    else:
        veto_manager = None
        print(f"  Signal Veto: DISABLED")

    # Dynamic Consensus Threshold
    use_dynamic_threshold = strategy_cfg.get('use_dynamic_threshold', True)
    if use_dynamic_threshold:
        dynamic_thresholds = strategy_cfg.get('dynamic_threshold', {
            'crisis_threshold': 0.20,
            'elevated_threshold': 0.15,
            'normal_threshold': 0.12,
            'low_vol_threshold': 0.10,
        })
        print(f"  Dynamic Threshold: ENABLED (range {dynamic_thresholds['low_vol_threshold']}-{dynamic_thresholds['crisis_threshold']})")
    else:
        dynamic_thresholds = None
        print(f"  Dynamic Threshold: DISABLED (fixed at {consensus_threshold})")

    # ===== STRATEGY ENHANCEMENTS (ABLATION STUDY) =====
    # Enhancement 1: Regime Transition Signals
    use_regime_transition = strategy_cfg.get('use_regime_transition_signals', False)
    regime_transition_signal = None
    if use_regime_transition:
        rt_cfg = strategy_cfg.get('regime_transition', {})
        regime_transition_config = RegimeTransitionConfig(
            low_regime_boundary=rt_cfg.get('low_regime_boundary', 0.30),
            high_regime_boundary=rt_cfg.get('high_regime_boundary', 0.70),
            transition_threshold=rt_cfg.get('transition_threshold', 0.10),
            signal_weight=rt_cfg.get('signal_weight', 0.15),
        )
        regime_transition_signal = RegimeTransitionSignal(regime_transition_config)
        print(f"  E1 Regime Transition: ENABLED (weight={regime_transition_config.signal_weight})")
    else:
        print(f"  E1 Regime Transition: DISABLED")

    # Enhancement 2: Term Structure Leverage
    use_term_leverage = strategy_cfg.get('use_term_structure_leverage', False)
    term_structure_calc = None
    if use_term_leverage:
        ts_cfg = strategy_cfg.get('term_structure_leverage', {})
        term_structure_config = TermStructureLeverageConfig(
            max_leverage=ts_cfg.get('max_leverage', 2.0),
            min_leverage=ts_cfg.get('min_leverage', 0.25),
            steep_contango_threshold=ts_cfg.get('steep_contango_threshold', 0.08),
            steep_contango_leverage=ts_cfg.get('steep_contango_leverage', 2.0),
            moderate_contango_leverage=ts_cfg.get('moderate_contango_leverage', 1.5),
            veto_on_extreme_backwardation=ts_cfg.get('veto_on_extreme_backwardation', True),
        )
        term_structure_calc = TermStructureLeverageCalculator(term_structure_config)
        print(f"  E2 Term Structure Leverage: ENABLED (max={term_structure_config.max_leverage}x)")
    else:
        print(f"  E2 Term Structure Leverage: DISABLED")

    # Enhancement 3: Vol-of-Vol (VVIX) Signal
    use_vov = strategy_cfg.get('use_vov_signal', False)
    vov_generator = None
    if use_vov:
        vov_cfg = strategy_cfg.get('vov_signal', {})
        vov_config = VoVConfig(
            high_threshold=vov_cfg.get('high_vov_threshold', 0.50),
            low_threshold=vov_cfg.get('low_vov_threshold', 0.25),
            high_long_scalar=vov_cfg.get('high_vov_long_scalar', 1.3),
            high_short_scalar=vov_cfg.get('high_vov_short_scalar', 0.7),
            low_short_scalar=vov_cfg.get('low_vov_short_scalar', 1.2),
            low_long_scalar=vov_cfg.get('low_vov_long_scalar', 0.8),
            lookback_window=vov_cfg.get('lookback_days', 20),
        )
        vov_generator = VoVSignalGenerator(vov_config)
        print(f"  E3 Vol-of-Vol (VVIX): ENABLED")
    else:
        print(f"  E3 Vol-of-Vol (VVIX): DISABLED")

    # Enhancement 4: Intraday Volatility Patterns (requires OHLC data)
    use_intraday_vol = strategy_cfg.get('use_intraday_vol_decomposition', False)
    intraday_vol_calc = None
    if use_intraday_vol:
        # Check if OHLC data exists
        ohlc_path = Path('data/spy_ohlc.csv')
        if ohlc_path.exists():
            iv_cfg = strategy_cfg.get('intraday_vol', {})
            intraday_vol_config = IntradayVolConfig(
                window=iv_cfg.get('lookback_days', 20),
                overnight_dominant_threshold=iv_cfg.get('overnight_dominant_threshold', 1.5),
                intraday_dominant_threshold=iv_cfg.get('intraday_dominant_threshold', 0.67),
                overnight_dominant_scalar=iv_cfg.get('overnight_dominant_scalar', 0.75),
                intraday_dominant_scalar=iv_cfg.get('intraday_dominant_scalar', 1.1),
            )
            intraday_vol_calc = IntradayVolCalculator(intraday_vol_config)
            print(f"  E4 Intraday Vol Patterns: ENABLED (requires OHLC data)")
        else:
            print(f"  E4 Intraday Vol Patterns: DISABLED (no OHLC data at {ohlc_path})")
    else:
        print(f"  E4 Intraday Vol Patterns: DISABLED")

    # Enhancement 5: Dynamic Signal Weighting
    use_dynamic_weights = strategy_cfg.get('use_dynamic_weighting', False)
    dynamic_weighter = None
    if use_dynamic_weights:
        dw_cfg = strategy_cfg.get('dynamic_weighting', {})
        dynamic_weight_config = DynamicWeightingConfig(
            ema_decay=dw_cfg.get('ema_decay', 0.95),
            min_weight=dw_cfg.get('min_weight', 0.05),
            max_weight=dw_cfg.get('max_weight', 0.40),
            min_samples_for_adaptation=dw_cfg.get('min_samples_for_adaptation', 20),
        )
        dynamic_weighter = DynamicSignalWeighter(dynamic_weight_config)
        print(f"  E5 Dynamic Weighting: ENABLED (min_samples={dynamic_weight_config.min_samples_for_adaptation})")
    else:
        print(f"  E5 Dynamic Weighting: DISABLED")

    # Enhancement 6: Asymmetric Profit Taking
    use_asymmetric = strategy_cfg.get('use_asymmetric_targets', False)
    asymmetric_manager = None
    if use_asymmetric:
        ap_cfg = strategy_cfg.get('asymmetric_targets', {})
        asymmetric_config = AsymmetricProfitConfig(
            short_vol_profit_target=ap_cfg.get('short_vol_profit_target', 0.08),
            short_vol_stop_loss=ap_cfg.get('short_vol_stop_loss', -0.15),
            long_vol_profit_target=ap_cfg.get('long_vol_profit_target', 0.20),
            long_vol_stop_loss=ap_cfg.get('long_vol_stop_loss', -0.08),
        )
        asymmetric_manager = AsymmetricProfitManager(asymmetric_config)
        print(f"  E6 Asymmetric Targets: ENABLED (short: +{asymmetric_config.short_vol_profit_target*100:.0f}%/-{abs(asymmetric_config.short_vol_stop_loss)*100:.0f}%)")
    else:
        print(f"  E6 Asymmetric Targets: DISABLED")

    # Enhancement 7: Alternative RV Methods
    rv_method = strategy_cfg.get('rv_method', 'close_to_close')
    rv_use_ensemble = strategy_cfg.get('rv_use_ensemble', False)
    ohlc_df = None

    # Load OHLC data if using alternative RV methods OR E4 intraday vol
    needs_ohlc = (rv_method != 'close_to_close') or rv_use_ensemble or (intraday_vol_calc is not None)
    if needs_ohlc:
        ohlc_path = Path('data/spy_ohlc.csv')
        if ohlc_path.exists():
            ohlc_df = pd.read_csv(ohlc_path, parse_dates=['date'])
            ohlc_df['date'] = ohlc_df['date'].dt.date
            ohlc_df.set_index('date', inplace=True)
            if rv_method != 'close_to_close':
                print(f"  E7 RV Method: {rv_method.upper()} (OHLC data loaded: {len(ohlc_df)} days)")
            else:
                print(f"  E7 RV Method: CLOSE_TO_CLOSE (default)")
        else:
            print(f"  E7 RV Method: {rv_method} requested but no OHLC data at {ohlc_path}, falling back to close_to_close")
            rv_method = 'close_to_close'
            rv_use_ensemble = False
    else:
        print(f"  E7 RV Method: CLOSE_TO_CLOSE (default)")

    # Z-SCORE ADAPTIVE THRESHOLDS (Anti-Overfitting)
    # Instead of static percentile/premium thresholds that overfit to 2019-2022,
    # use z-scores that automatically adapt to current market regime
    z_threshold_normal = 1.0    # More trades in calm markets
    z_threshold_stressed = 1.5  # Higher bar in stressed markets

    # Profit targets and stops - none needed, IV reversion handles exits
    profit_target_pct = 0.10   # Very wide - let winners run
    stop_loss_pct = -0.10     # Very wide - give room for mean reversion

    print(f"\nConfig:")
    print(f"  - QV Strategy: {config.use_qv_strategy}")
    print(f"  - Regime Detection: {config.use_regime_detection}")
    print(f"  - Entry: Z-score adaptive (normal: {z_threshold_normal}σ, stressed: {z_threshold_stressed}σ)")
    print(f"  - Regime Split: RV percentile > 50% = stressed")
    print(f"  - Profit Target: {profit_target_pct*100:.1f}%")
    print(f"  - Stop Loss: {stop_loss_pct*100:.1f}%")
    print(f"  - Position Size: {position_size_base*100:.1f}%")

    # Initialize
    qv_calc = QVSignalCalculator(config)
    equity = initial_capital
    position = 0  # +1 long vol, -1 short vol, 0 flat
    entry_date = None
    entry_consensus = 0
    entry_equity = None  # Track equity at entry for P/L calculation
    entry_iv = None      # Track IV at entry
    entry_rv_percentile = None  # Track vol regime at entry (for "hold through vol rise")
    entry_signals = {}  # E5: Track entry signals for dynamic weighting
    prev_iv = None

    equity_curve = []
    trades = []
    position_pnl = 0  # Track cumulative P&L for current position
    entry_position_size = 0  # Track position size at entry for P&L calculation

    # Pre-calculate spot and RV
    dates = sorted(options_df['date'].unique())
    print(f"\nDate range: {dates[0]} to {dates[-1]}")
    print(f"Trading days: {len(dates)}")

    spot_prices = []
    for date in dates:
        day_df = options_df[options_df['date'] == date]
        calls = day_df[day_df['type'] == 'call'].copy()
        if len(calls) > 0:
            # Filter to 30 DTE options for better ATM detection
            calls['dte'] = (calls['expiration'] - date).dt.days
            atm_calls = calls[(calls['dte'] >= 25) & (calls['dte'] <= 35)]
            if len(atm_calls) > 0:
                atm_idx = (atm_calls['delta'].abs() - 0.5).abs().argmin()
                atm_call = atm_calls.iloc[atm_idx]
                # Use ATM strike as spot estimate (not strike + premium)
                spot = atm_call['strike']
                spot_prices.append({'date': date, 'spot': spot})

    spot_df = pd.DataFrame(spot_prices).set_index('date')
    spot_df['returns'] = spot_df['spot'].pct_change()

    # E7: Use alternative RV method if configured and OHLC data available
    if ohlc_df is not None and (rv_method != 'close_to_close' or rv_use_ensemble):
        # Calculate RV on full OHLC data, then align to spot_df
        if rv_use_ensemble:
            ohlc_rv = ensemble_rv(ohlc_df, window=20, annualize=True)
            print(f"  E7: Applied ENSEMBLE RV ({spot_df.index.isin(ohlc_df.index).sum()} days with data)")
        else:
            ohlc_rv = calculate_rv(ohlc_df, method=rv_method, window=20, annualize=True)
            print(f"  E7: Applied {rv_method} RV ({spot_df.index.isin(ohlc_df.index).sum()} days with data)")
        # Align to spot_df dates
        spot_df['rv_20d'] = ohlc_rv.reindex(spot_df.index).ffill()
    else:
        # Default: close-to-close RV
        spot_df['rv_20d'] = spot_df['returns'].rolling(20).std() * np.sqrt(252)

    spot_df['rv_20d_lagged'] = spot_df['rv_20d'].shift(1)

    warmup_days = max(config.feature_window + 21, 80)  # Need 60+ days for QV buffers
    print(f"Warmup period: {warmup_days} days")

    for i, date in enumerate(dates[warmup_days:], warmup_days):
        day_df = options_df[options_df['date'] == date]

        if date not in spot_df.index:
            equity_curve.append({'date': date, 'equity': equity, 'position': position})
            continue

        rv = spot_df.loc[date, 'rv_20d_lagged']
        spot = spot_df.loc[date, 'spot']

        if pd.isna(rv):
            equity_curve.append({'date': date, 'equity': equity, 'position': position})
            continue

        # Extract features
        features = qv_calc.extract_daily_features(day_df, spot, rv)
        qv_calc.update_buffers(features)

        # Generate signals
        signals = qv_calc.generate_signals(features)
        consensus = qv_calc.calculate_consensus(signals)
        regime_scalar = qv_calc.get_regime_scalar(rv)

        atm_iv = features['atm_iv']

        # Daily P&L for existing position
        # Using straddle approximation: Long straddle = Long 1 ATM call + Long 1 ATM put
        # Vega for ATM straddle ≈ 2 * 0.4 * S * sqrt(T) (call vega + put vega)
        daily_pnl = 0
        if position != 0 and prev_iv is not None:
            iv_change = atm_iv - prev_iv
            dte_years = 45 / 365

            # Straddle vega: ~0.8 * S * sqrt(T) per 100 shares
            vega_per_straddle = 0.8 * spot * np.sqrt(dte_years)

            # Number of straddles based on position size at entry
            # Use entry_position_size which includes enhancement scalars from trade entry
            # This allows enhancements to actually affect P&L through position sizing
            pnl_position_size = entry_position_size if entry_position_size > 0 else position_size_base
            notional_value = pnl_position_size * entry_equity  # Use entry equity for consistent sizing
            avg_straddle_premium = spot * atm_iv * np.sqrt(dte_years) * 0.8  # Approximation
            num_straddles = notional_value / (avg_straddle_premium * 100) if avg_straddle_premium > 0 else 0

            # Vega P&L: vega * iv_change * num_contracts * 100
            # IV is in decimal (0.20 = 20%), so multiply by 100 to get per-point vega P&L
            vega_pnl = position * vega_per_straddle * (iv_change * 100) * num_straddles

            # Theta P&L: ATM straddle loses roughly vega/365 per day
            theta_daily = -avg_straddle_premium * 100 / 45  # Premium decay over 45 days
            theta_pnl = -position * abs(theta_daily) * num_straddles

            daily_pnl = vega_pnl + theta_pnl

        # Calculate holding period
        days_held = 0
        if entry_date is not None:
            days_held = (pd.Timestamp(date) - pd.Timestamp(entry_date)).days

        # ===== ENHANCED POSITION SIZING =====
        # Calculate RV percentile for regime detection
        rv_percentile = 0.5
        if len(qv_calc.rv_history) >= 20:
            rv_arr = np.array(qv_calc.rv_history)
            rv_percentile = (rv_arr < rv).sum() / len(rv_arr)

        # Get drawdown scalar
        dd_scalar = 1.0
        is_drawdown_halted = False
        if drawdown_manager is not None:
            dd_scalar = drawdown_manager.get_position_scalar(equity)
            is_drawdown_halted = drawdown_manager.is_halted

        # Get dynamic consensus threshold
        current_threshold = consensus_threshold
        if dynamic_thresholds is not None:
            vix_proxy = atm_iv * 100  # Convert IV to VIX-like scale
            if rv_percentile > 0.90 or vix_proxy > 30:
                current_threshold = dynamic_thresholds['crisis_threshold']
            elif rv_percentile > 0.70 or vix_proxy > 20:
                current_threshold = dynamic_thresholds['elevated_threshold']
            elif rv_percentile > 0.30:
                current_threshold = dynamic_thresholds['normal_threshold']
            else:
                current_threshold = dynamic_thresholds['low_vol_threshold']

        # Kelly-based position sizing
        if kelly_sizer is not None:
            kelly_size = kelly_sizer.get_position_size(
                current_drawdown=drawdown_manager.current_drawdown if drawdown_manager else 0,
                vol_percentile=rv_percentile
            )
            position_size = kelly_size * regime_scalar * dd_scalar
        elif config.use_tiered_sizing:
            # Legacy quadratic scaling
            consensus_strength = abs(consensus)
            if config.position_scaling_method == "quadratic":
                size_multiplier = consensus_strength ** 2
            elif config.position_scaling_method == "cubic":
                size_multiplier = consensus_strength ** 3
            else:
                size_multiplier = consensus_strength
            position_size = position_size_base * size_multiplier * regime_scalar * dd_scalar
        else:
            position_size = position_size_base * regime_scalar * dd_scalar

        # Update veto manager with VIX data
        vix_proxy = atm_iv * 100
        if veto_manager is not None:
            veto_manager.update_vix(vix_proxy)

        # ===== Z-SCORE CALCULATION (needed for enhancements) =====
        # Calculate z-score of IV premium vs rolling window (EXCLUDING today)
        iv_premium_z = 0.0
        if len(qv_calc.iv_premium_buffer) >= 21:
            iv_arr = np.array(list(qv_calc.iv_premium_buffer)[:-1])  # Exclude today
            iv_mean = iv_arr.mean()
            iv_std = iv_arr.std()
            if iv_std > 1e-8:
                iv_premium_z = (features['iv_premium'] - iv_mean) / iv_std

        # ===== APPLY STRATEGY ENHANCEMENTS =====
        enhancement_scalar = 1.0  # Combined enhancement position scalar
        term_structure_leverage = 1.0
        term_structure_veto = False
        term_structure_veto_reason = None

        # E1: Regime Transition Signal (position scalar based on favorable transitions)
        regime_transition_value = 0
        if regime_transition_signal is not None:
            regime_transition_value = regime_transition_signal.update(rv_percentile)
            # Apply as position scalar: boost on favorable transitions
            # HIGH→NORMAL (+1) with short vol (iv_premium_z > 0) = favorable
            # LOW→NORMAL (-1) with long vol (iv_premium_z < 0) = favorable
            if regime_transition_value != 0:
                signal_alignment = regime_transition_value * np.sign(iv_premium_z)
                enhancement_scalar *= (1.0 + 0.25 * signal_alignment)

        # E2: Term Structure Leverage
        if term_structure_calc is not None:
            # Get IV at different tenors (30d and 60d)
            iv_30d = features.get('atm_iv', atm_iv)
            # For 60d IV, use term_structure feature (which is iv_60 - iv_30)
            iv_60d = iv_30d + features.get('term_structure', 0.0)
            signal_direction = "SHORT_VOL" if iv_premium_z > 0 else "LONG_VOL"
            term_structure_leverage, ts_regime, term_structure_veto, term_structure_veto_reason = \
                term_structure_calc.get_leverage(iv_30d, iv_60d, signal_direction)
            if not term_structure_veto:
                enhancement_scalar *= term_structure_leverage

        # E3: Vol-of-Vol (VVIX) Signal
        vov_scalar = 1.0
        if vov_generator is not None:
            vov_generator.update_iv(atm_iv)
            # Will apply scalar at entry based on position direction

        # E4: Intraday Volatility Patterns (requires OHLC data)
        intraday_scalar = 1.0
        if intraday_vol_calc is not None and ohlc_df is not None and date in ohlc_df.index:
            open_price = ohlc_df.loc[date, 'open']
            close_price = ohlc_df.loc[date, 'close']
            intraday_vol_calc.update(open_price, close_price)
            intraday_scalar = intraday_vol_calc.get_position_scalar()
            enhancement_scalar *= intraday_scalar

        # Apply enhancement scalar to position size
        position_size *= enhancement_scalar

        # ===== Z-SCORE BASED ENTRY (Anti-Overfitting) =====
        # Replace static thresholds with adaptive z-scores that adjust to regime
        # This fixes the 52% OOS Sharpe degradation problem
        # NOTE: iv_premium_z is already calculated above (before enhancements)

        # Multi-tier regime classification for higher signal quality
        # Crisis regime: Skip short vol entirely (too risky)
        # Elevated regime: Require very high z-score (2.5σ)
        # Stressed regime: Require high z-score (2.0σ)
        # Normal regime: Standard threshold (1.5σ)
        # Low vol regime: Can be more aggressive (1.25σ)
        is_crisis_regime = rv_percentile > 0.85
        is_elevated_regime = rv_percentile > 0.70
        is_stressed_regime = rv_percentile > 0.50
        is_low_vol_regime = rv_percentile < 0.30

        # Tiered z-thresholds for SHORT VOL (protective in high-vol regimes)
        if is_crisis_regime:
            z_threshold = 3.0  # Extremely high bar (effectively blocks most short vol)
        elif is_elevated_regime:
            z_threshold = 2.5  # Very high bar
        elif is_stressed_regime:
            z_threshold = 2.0  # High bar
        elif is_low_vol_regime:
            z_threshold = 1.25  # Slightly lower bar in calm markets
        else:
            z_threshold = 1.5  # Normal threshold

        # SYMMETRIC thresholds - same for long and short vol
        # The tiered z-thresholds above already provide regime-awareness
        long_vol_threshold = z_threshold

        # Term structure (kept for logging, not entry decision)
        term_structure = features.get('term_structure', 0.0)

        if position == 0:
            # Z-SCORE BASED ENTRY - Automatically adapts to regime
            # When spreads compress (2023-2024), rolling mean/std adjust
            # A 1.0-1.5σ signal still captures "extreme" relative to recent history

            # SHORT VOL: IV premium significantly above recent average
            # Uses protective high thresholds in high-vol regimes
            short_vol_entry = iv_premium_z > z_threshold

            # LONG VOL: IV premium significantly below recent average
            # Uses aggressive lower thresholds in high-vol regimes (opposite logic)
            long_vol_entry = iv_premium_z < -long_vol_threshold

            # ===== E2: TERM STRUCTURE VETO CHECK =====
            if term_structure_veto and (short_vol_entry or long_vol_entry):
                signal_type = "SHORT_VOL" if short_vol_entry else "LONG_VOL"
                trades.append({
                    'date': date,
                    'action': 'VETOED',
                    'veto_reason': 'TERM_STRUCTURE_VETO',
                    'veto_details': term_structure_veto_reason,
                    'consensus': consensus,
                    'iv_premium_z': iv_premium_z,
                    'would_have_been': signal_type,
                })
                short_vol_entry = False
                long_vol_entry = False

            # ===== SIGNAL VETO CHECK =====
            if veto_manager is not None and (short_vol_entry or long_vol_entry):
                signal_type = "SHORT_VOL" if short_vol_entry else "LONG_VOL"
                veto_result = veto_manager.check_veto(
                    signal_type=signal_type,
                    current_vix=vix_proxy,
                    term_structure=term_structure,
                    signals=signals,
                    rv_percentile=rv_percentile,
                    iv_premium=features['iv_premium'],
                    is_drawdown_halted=is_drawdown_halted,
                )
                if veto_result.is_vetoed:
                    # Log vetoed trade
                    trades.append({
                        'date': date,
                        'action': 'VETOED',
                        'veto_reason': veto_result.reason.value if veto_result.reason else 'UNKNOWN',
                        'veto_details': veto_result.details,
                        'consensus': consensus,
                        'iv_premium_z': iv_premium_z,
                        'would_have_been': signal_type,
                    })
                    short_vol_entry = False
                    long_vol_entry = False

            # ===== E3: APPLY VOV SCALAR AT ENTRY =====
            if vov_generator is not None and (short_vol_entry or long_vol_entry):
                signal_type = "SHORT_VOL" if short_vol_entry else "LONG_VOL"
                vov_scalar = vov_generator.get_position_scalar(signal_type)
                position_size *= vov_scalar

            if short_vol_entry:
                # Short volatility - IV is high with contango, expect mean reversion
                position = -1
                entry_date = date
                entry_consensus = consensus
                entry_equity = equity
                entry_iv = atm_iv
                entry_position_size = position_size  # Track for P&L calculation
                entry_signals = signals.copy()  # E5: Track signals for dynamic weighting
                position_pnl = 0

                cost = option_spread_cost * position_size * equity + commission_per_contract * 2
                equity -= cost

                trades.append({
                    'date': date,
                    'action': 'SHORT_VOL',
                    'consensus': consensus,
                    'iv_premium_z': iv_premium_z,
                    'z_threshold': z_threshold,
                    'is_stressed': is_stressed_regime,
                    'term_structure': term_structure,
                    'regime_scalar': regime_scalar,
                    'enhancement_scalar': enhancement_scalar,
                    'term_structure_leverage': term_structure_leverage,
                    'vov_scalar': vov_scalar if vov_generator else 1.0,
                    'iv': atm_iv,
                    'rv': rv,
                    'iv_premium': features['iv_premium'],
                    'cost': cost
                })

            elif long_vol_entry:
                # Long volatility - "Cheap Insurance" strategy
                # Buy when vol is low (cheap), hold through vol rises
                position = 1
                entry_date = date
                entry_consensus = consensus
                entry_equity = equity
                entry_iv = atm_iv
                entry_rv_percentile = rv_percentile  # Track for "hold through vol rise"
                entry_position_size = position_size  # Track for P&L calculation
                entry_signals = signals.copy()  # E5: Track signals for dynamic weighting
                position_pnl = 0

                cost = option_spread_cost * position_size * equity + commission_per_contract * 2
                equity -= cost

                trades.append({
                    'date': date,
                    'action': 'LONG_VOL',
                    'consensus': consensus,
                    'iv_premium_z': iv_premium_z,
                    'z_threshold': z_threshold,
                    'is_stressed': is_stressed_regime,
                    'term_structure': term_structure,
                    'regime_scalar': regime_scalar,
                    'enhancement_scalar': enhancement_scalar,
                    'term_structure_leverage': term_structure_leverage,
                    'vov_scalar': vov_scalar if vov_generator else 1.0,
                    'iv': atm_iv,
                    'rv': rv,
                    'iv_premium': features['iv_premium'],
                    'cost': cost
                })

        # Exit logic with profit targets and stops
        elif position != 0:
            should_exit = False
            exit_reason = ""

            # Calculate position P&L as percentage of entry equity
            position_pnl += daily_pnl
            position_return = position_pnl / entry_equity if entry_equity else 0

            # ===== E6: ASYMMETRIC PROFIT TARGETS =====
            # Get appropriate targets based on position direction
            if asymmetric_manager is not None:
                position_type = "SHORT_VOL" if position == -1 else "LONG_VOL"
                current_profit_target, current_stop_loss = asymmetric_manager.get_targets(position_type)
            else:
                current_profit_target = profit_target_pct
                current_stop_loss = stop_loss_pct

            # 1. Profit target hit
            if position_return >= current_profit_target:
                should_exit = True
                exit_reason = "PROFIT_TARGET"

            # 2. Stop loss hit
            elif position_return <= current_stop_loss:
                should_exit = True
                exit_reason = "STOP_LOSS"

            # 3. IV mean reversion complete (after min holding period)
            # Using z-score: exit when z-score crosses zero (mean reversion complete)
            elif days_held >= min_holding_days:
                if position == -1 and iv_premium_z < 0:  # Was short, IV premium below mean
                    should_exit = True
                    exit_reason = "IV_REVERSION"
                elif position == 1 and iv_premium_z > 0:  # Was long, IV premium above mean
                    should_exit = True
                    exit_reason = "IV_REVERSION"

            # 4. Max holding period
            if days_held >= 30:  # Extended from 20 to 30 days
                should_exit = True
                exit_reason = "MAX_HOLDING"

            if should_exit:
                cost = option_spread_cost * position_size * equity + commission_per_contract * 2
                equity -= cost

                # Record trade for Kelly sizer
                if kelly_sizer is not None and entry_equity is not None:
                    kelly_sizer.add_trade(
                        pnl=position_pnl,
                        entry_size=entry_equity * position_size_base
                    )

                # E5: Record outcome for dynamic signal weighting (must be before position reset)
                if dynamic_weighter is not None and entry_signals:
                    dynamic_weighter.record_outcome(
                        signals=entry_signals,
                        actual_return=position_return,
                        position_direction=position
                    )

                trades.append({
                    'date': date,
                    'action': 'EXIT',
                    'exit_reason': exit_reason,
                    'consensus': consensus,
                    'entry_consensus': entry_consensus,
                    'iv': atm_iv,
                    'rv': rv,
                    'days_held': days_held,
                    'position_return': position_return * 100,  # As percentage
                    'cost': cost,
                    'position_size': position_size,
                    'dd_scalar': dd_scalar,
                })
                position = 0
                entry_date = None
                entry_consensus = 0
                entry_equity = None
                entry_iv = None
                entry_rv_percentile = None  # Reset vol regime tracking
                entry_position_size = 0
                entry_signals = {}  # E5: Reset tracked signals
                position_pnl = 0

        equity += daily_pnl
        prev_iv = atm_iv

        equity_curve.append({
            'date': date,
            'equity': equity,
            'position': position,
            'consensus': consensus,
            'iv': atm_iv,
            'rv': rv,
            'regime_scalar': regime_scalar,
            'dd_scalar': dd_scalar,
            'rv_percentile': rv_percentile,
        })

    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    returns = equity_df['equity'].pct_change().dropna()

    total_return = (equity - initial_capital) / initial_capital * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    nw_sharpe = float(calculate_sharpe_ratio(
        returns,
        risk_free_rate=Decimal("0.05"),
        adjust_autocorrelation=True
    ))

    rolling_max = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    # Trade analysis
    long_trades = [t for t in trades if t['action'] == 'LONG_VOL']
    short_trades = [t for t in trades if t['action'] == 'SHORT_VOL']
    exit_trades = [t for t in trades if t['action'] == 'EXIT']
    vetoed_trades = [t for t in trades if t['action'] == 'VETOED']

    # Exit reason breakdown
    exit_reasons = {}
    for t in exit_trades:
        reason = t.get('exit_reason', 'UNKNOWN')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Veto reason breakdown
    veto_reasons = {}
    for t in vetoed_trades:
        reason = t.get('veto_reason', 'UNKNOWN')
        veto_reasons[reason] = veto_reasons.get(reason, 0) + 1

    # Calculate win rate
    winning_exits = [t for t in exit_trades if t.get('position_return', 0) > 0]
    losing_exits = [t for t in exit_trades if t.get('position_return', 0) < 0]
    win_rate = len(winning_exits) / len(exit_trades) * 100 if exit_trades else 0

    # Average return per trade
    avg_return = np.mean([t.get('position_return', 0) for t in exit_trades]) if exit_trades else 0
    avg_winner = np.mean([t.get('position_return', 0) for t in winning_exits]) if winning_exits else 0
    avg_loser = np.mean([t.get('position_return', 0) for t in losing_exits]) if losing_exits else 0

    # Calculate total trading costs
    total_costs = sum(t.get('cost', 0) for t in trades)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nInitial Capital:   ${initial_capital:>12,.2f}")
    print(f"Final Capital:     ${equity:>12,.2f}")
    print(f"Total Return:      {total_return:>12.2f}%")
    print(f"\nSharpe Ratio:      {sharpe:>12.2f}")
    print(f"NW-Adj Sharpe:     {nw_sharpe:>12.2f}")
    print(f"Max Drawdown:      {max_dd:>12.2f}%")
    print(f"\nTotal Trades:      {len(trades):>12}")
    print(f"  Entries:         {len(long_trades) + len(short_trades):>12}")
    print(f"    Long Vol:      {len(long_trades):>12}")
    print(f"    Short Vol:     {len(short_trades):>12}")
    print(f"  Exits:           {len(exit_trades):>12}")
    print(f"\nExit Reasons:")
    for reason, count in sorted(exit_reasons.items()):
        print(f"    {reason:14s} {count:>6}")
    print(f"\nWin Rate:          {win_rate:>12.1f}%")
    print(f"Avg Return/Trade:  {avg_return:>12.2f}%")
    print(f"Avg Winner:        {avg_winner:>12.2f}%")
    print(f"Avg Loser:         {avg_loser:>12.2f}%")
    print(f"\nTotal Trading Costs: ${total_costs:>10,.2f}")

    # ===== NEW: Risk Management Statistics =====
    if vetoed_trades:
        print(f"\n--- SIGNAL VETO ANALYSIS ---")
        print(f"  Total Vetoed:    {len(vetoed_trades):>6}")
        for reason, count in sorted(veto_reasons.items()):
            print(f"    {reason:20s} {count:>4}")

    if drawdown_manager is not None:
        dd_status = drawdown_manager.get_status()
        print(f"\n--- DRAWDOWN MANAGEMENT ---")
        print(f"  Max Drawdown Seen: {dd_status['max_drawdown_pct']:>6.2f}%")
        print(f"  Trading Halted:    {'YES' if dd_status['is_halted'] else 'NO':>6}")

    if kelly_sizer is not None:
        kelly_stats = kelly_sizer.get_statistics()
        print(f"\n--- KELLY SIZING ---")
        print(f"  Trades Recorded:   {kelly_stats['trades_recorded']:>6}")
        print(f"  Win Rate:          {kelly_stats['win_rate']:>6.1f}%")
        print(f"  Win/Loss Ratio:    {kelly_stats['win_loss_ratio']:>6.2f}")
        print(f"  Full Kelly:        {kelly_stats['kelly_full']:>6.2f}%")
        print(f"  Fractional Kelly:  {kelly_stats['kelly_fractional']:>6.2f}%")

    # Analyze short vs long vol separately
    short_exits = [t for t in exit_trades if t.get('entry_consensus', 0) < 0 or
                   any(e['action'] == 'SHORT_VOL' and
                       (pd.Timestamp(t['date']) - pd.Timestamp(e['date'])).days < 35
                       for e in short_trades)]
    long_exits = [t for t in exit_trades if t not in short_exits]

    print(f"\nShort Vol Performance:")
    print(f"  Entries:         {len(short_trades):>6}")
    short_wins = sum(1 for t in exit_trades if t.get('position_return', 0) > 0)
    print(f"  Win Rate (approx): {short_wins/len(exit_trades)*100:.1f}%")

    print(f"\nLong Vol Performance:")
    print(f"  Entries:         {len(long_trades):>6}")

    # Year-by-year breakdown
    print("\n" + "="*60)
    print("YEAR-BY-YEAR ANALYSIS")
    print("="*60)
    equity_df['year'] = equity_df.index.year
    yearly_returns = equity_df.groupby('year')['equity'].agg(['first', 'last'])
    yearly_returns['return'] = (yearly_returns['last'] - yearly_returns['first']) / yearly_returns['first'] * 100
    for year, row in yearly_returns.iterrows():
        print(f"  {year}: {row['return']:>8.2f}%")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("Realistic Sharpe for Vol Arb: 0.5 - 1.5")
    print("Red Flag Sharpe: > 2.0 (likely still has issues)")

    if sharpe > 2.0:
        print("\n[WARNING] Sharpe ratio still suspiciously high!")
    elif sharpe < 0.5:
        print("\n[INFO] Low Sharpe - strategy may need tuning")
    else:
        print("\n[OK] Sharpe ratio in realistic range for vol arb")

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'nw_sharpe': nw_sharpe,
        'max_drawdown': max_dd,
        'trades': len(trades),
        'equity_curve': equity_df
    }


def main():
    parser = argparse.ArgumentParser(description='Run volatility arbitrage backtest')
    parser.add_argument(
        '--data',
        type=str,
        default='src/volatility_arbitrage/data/SPY_Options_2019_24',
        help='Path to options data directory'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/volatility_arb.yaml',
        help='Path to strategy config file'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date filter (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date filter (YYYY-MM-DD)'
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    config = load_strategy_config(Path(args.config))

    # Load data
    print(f"\nLoading options data from {args.data}...")
    options_df = load_json_options_data(args.data)

    # Apply date filters
    if args.start_date:
        start = pd.to_datetime(args.start_date)
        options_df = options_df[options_df['date'] >= start]
        print(f"  Filtered to start: {args.start_date}")
    if args.end_date:
        end = pd.to_datetime(args.end_date)
        options_df = options_df[options_df['date'] <= end]
        print(f"  Filtered to end: {args.end_date}")

    print(f"\nData summary:")
    print(f"  Records: {len(options_df):,}")
    print(f"  Date range: {options_df['date'].min()} to {options_df['date'].max()}")
    print(f"  Unique dates: {options_df['date'].nunique()}")

    # Run backtest
    results = run_qv_backtest(options_df, config, args.capital, config_path=args.config)


if __name__ == "__main__":
    main()
