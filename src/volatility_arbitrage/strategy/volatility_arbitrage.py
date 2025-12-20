"""
Volatility Arbitrage Strategy Implementation.

Implements a delta-neutral volatility arbitrage strategy that:
1. Forecasts realized volatility using GARCH
2. Compares to implied volatility from options
3. Trades the volatility spread
4. Maintains delta-neutral hedging
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from volatility_arbitrage.core.types import OptionChain, OptionContract, Position, OptionType
from volatility_arbitrage.models.black_scholes import BlackScholesModel
from volatility_arbitrage.models.volatility import GARCHVolatility, calculate_returns
from volatility_arbitrage.risk.uncertainty_sizing import UncertaintySizer, UncertaintySizingConfig
from volatility_arbitrage.strategy.base import Strategy, Signal
from volatility_arbitrage.utils.logging import get_logger

# Optional regime detection - gracefully handle if dependencies not available
try:
    from volatility_arbitrage.models.regime import RegimeDetector
except ImportError:
    RegimeDetector = None  # type: ignore

logger = get_logger(__name__)


@dataclass
class FeatureBuffers:
    """
    Rolling buffers for QV strategy feature calculation.

    Uses deques with maxlen for O(1) append/pop operations.
    Tracks features needed for z-score calculation and signal generation.
    """

    # 20-day windows for realized volatility
    returns_20d: deque = field(default_factory=lambda: deque(maxlen=20))
    rv_20d: deque = field(default_factory=lambda: deque(maxlen=20))

    # 60-day windows for feature medians/stdevs
    pc_ratio_60d: deque = field(default_factory=lambda: deque(maxlen=60))
    iv_skew_60d: deque = field(default_factory=lambda: deque(maxlen=60))
    iv_premium_60d: deque = field(default_factory=lambda: deque(maxlen=60))
    term_structure_60d: deque = field(default_factory=lambda: deque(maxlen=60))
    volume_ratio_60d: deque = field(default_factory=lambda: deque(maxlen=60))
    near_term_sentiment_60d: deque = field(default_factory=lambda: deque(maxlen=60))

    # 252-day window for vol regime percentile
    rv_252d: deque = field(default_factory=lambda: deque(maxlen=252))


@dataclass
class VolatilitySpread:
    """
    Volatility spread information for trading decisions.

    Tracks the difference between implied and forecasted volatility.
    """

    symbol: str
    timestamp: datetime
    implied_vol: Decimal
    forecasted_vol: Decimal
    spread: Decimal  # IV - RV
    spread_pct: Decimal  # (IV - RV) / RV * 100

    @property
    def is_long_opportunity(self) -> bool:
        """True if IV < RV (buy volatility opportunity)."""
        return self.spread < 0

    @property
    def is_short_opportunity(self) -> bool:
        """True if IV > RV (sell volatility opportunity)."""
        return self.spread > 0


@dataclass
class RegimeParameters:
    """
    Regime-specific strategy parameters.

    Allows different threshold and sizing rules for different market regimes.
    """

    regime_id: int
    entry_threshold_pct: Decimal
    exit_threshold_pct: Decimal
    position_size_multiplier: Decimal = Decimal("1.0")
    max_vega_multiplier: Decimal = Decimal("1.0")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "regime_id": self.regime_id,
            "entry_threshold_pct": float(self.entry_threshold_pct),
            "exit_threshold_pct": float(self.exit_threshold_pct),
            "position_size_multiplier": float(self.position_size_multiplier),
            "max_vega_multiplier": float(self.max_vega_multiplier),
        }


@dataclass
class VolatilityArbitrageConfig:
    """Configuration for volatility arbitrage strategy."""

    # Entry/exit thresholds (baseline, used when no regime detection)
    entry_threshold_pct: Decimal = Decimal("5.0")  # 5% spread to enter
    exit_threshold_pct: Decimal = Decimal("2.0")   # 2% spread to exit

    # Time constraints
    min_days_to_expiry: int = 14
    max_days_to_expiry: int = 60

    # Delta hedging
    delta_rebalance_threshold: Decimal = Decimal("0.10")  # Rehedge at 10 delta
    delta_target: Decimal = Decimal("0.0")  # Target delta-neutral

    # Position sizing
    position_size_pct: Decimal = Decimal("5.0")  # 5% of capital per trade
    max_vega_exposure: Decimal = Decimal("1000")  # Max vega per position
    max_positions: int = 5

    # Volatility forecasting
    vol_lookback_period: int = 30
    vol_forecast_method: str = "garch"  # garch, ewma, historical

    # Risk management
    max_loss_pct: Decimal = Decimal("50.0")  # Stop loss at 50% of premium

    # Regime detection (optional)
    use_regime_detection: bool = False
    regime_params: Optional[dict[int, RegimeParameters]] = None  # regime_id -> parameters
    regime_lookback_period: int = 60  # Days for regime detection
    exit_on_regime_transition: bool = False  # Exit all positions when regime changes

    # ===== QV STRATEGY CONFIGURATION =====
    # QV Strategy Toggle
    use_qv_strategy: bool = False  # Backward compatibility flag

    # QV Feature Windows
    rv_window: int = 20           # Realized volatility lookback
    feature_window: int = 60      # Z-score calculation window
    regime_window: int = 252      # Vol percentile window

    # QV Signal Thresholds
    pc_ratio_threshold: Decimal = Decimal("1.0")    # High fear threshold
    skew_threshold: Decimal = Decimal("0.05")       # 5% skew threshold
    premium_threshold: Decimal = Decimal("0.10")    # 10% IV premium threshold
    term_structure_threshold: Decimal = Decimal("0.0")  # Positive slope threshold
    volume_spike_threshold: Decimal = Decimal("1.5")    # 1.5x median volume
    sentiment_threshold: Decimal = Decimal("-0.05")     # Negative sentiment threshold

    # QV Consensus Scoring
    consensus_threshold: Decimal = Decimal("0.2")   # Minimum consensus for entry

    # QV Signal Weights (must sum to 1.0)
    weight_pc_ratio: Decimal = Decimal("0.20")
    weight_iv_skew: Decimal = Decimal("0.20")
    weight_iv_premium: Decimal = Decimal("0.15")
    weight_term_structure: Decimal = Decimal("0.15")
    weight_volume_spike: Decimal = Decimal("0.15")
    weight_near_term_sentiment: Decimal = Decimal("0.15")

    # QV Regime Scalars
    regime_crisis_scalar: Decimal = Decimal("0.5")    # Vol >90th percentile
    regime_elevated_scalar: Decimal = Decimal("0.75") # Vol 70-90th percentile
    regime_normal_scalar: Decimal = Decimal("1.0")    # Vol 30-70th percentile
    regime_low_scalar: Decimal = Decimal("1.2")       # Vol 10-30th percentile
    regime_extreme_low_scalar: Decimal = Decimal("1.5") # Vol <10th percentile

    # Bullish Base Exposure Parameters
    base_long_bias: Decimal = Decimal("0.8")              # Minimum long exposure (0.8 = 80%)
    signal_adjustment_factor: Decimal = Decimal("0.7")    # Signal scaling factor

    # ===== TIERED POSITION SIZING =====
    use_tiered_sizing: bool = True  # Enable continuous position scaling
    min_consensus_threshold: Decimal = Decimal("0.15")  # Below this = no trade (lower for synthetic data)
    position_scaling_method: str = "quadratic"  # linear, quadratic, cubic
    min_holding_days: int = 5  # Minimum days before exit allowed

    # ===== PHASE 2: LEVERAGE =====
    use_leverage: bool = False
    short_vol_leverage: Decimal = Decimal("1.3")
    long_vol_leverage: Decimal = Decimal("2.0")
    max_leveraged_notional_pct: Decimal = Decimal("0.80")
    leverage_drawdown_reduction: bool = True
    leverage_dd_threshold: Decimal = Decimal("0.10")

    # ===== PHASE 2: BAYESIAN LSTM VOLATILITY FORECASTING =====
    bayesian_lstm_hidden_size: int = 64
    bayesian_lstm_dropout_p: float = 0.2
    bayesian_lstm_sequence_length: int = 20
    bayesian_lstm_n_mc_samples: int = 50

    # ===== PHASE 2: UNCERTAINTY-ADJUSTED POSITION SIZING =====
    use_uncertainty_sizing: bool = False
    uncertainty_penalty: float = 2.0
    uncertainty_min_position_pct: float = 0.01
    uncertainty_max_position_pct: float = 0.15

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        base = {
            "entry_threshold_pct": float(self.entry_threshold_pct),
            "exit_threshold_pct": float(self.exit_threshold_pct),
            "min_days_to_expiry": self.min_days_to_expiry,
            "max_days_to_expiry": self.max_days_to_expiry,
            "delta_rebalance_threshold": float(self.delta_rebalance_threshold),
            "position_size_pct": float(self.position_size_pct),
            "max_vega_exposure": float(self.max_vega_exposure),
            "use_regime_detection": self.use_regime_detection,
        }

        if self.regime_params:
            base["regime_params"] = {k: v.to_dict() for k, v in self.regime_params.items()}

        return base


class VolatilityArbitrageStrategy(Strategy):
    """
    Delta-neutral volatility arbitrage strategy.

    Strategy logic:
    1. Forecast realized volatility using GARCH
    2. Compare to implied volatility from ATM options
    3. Long volatility when IV < RV (buy straddle + hedge)
    4. Short volatility when IV > RV (sell straddle + hedge)
    5. Maintain delta-neutral through rebalancing
    """

    def __init__(
        self,
        config: VolatilityArbitrageConfig,
        regime_detector: Optional[RegimeDetector] = None,
    ) -> None:
        """
        Initialize volatility arbitrage strategy.

        Args:
            config: Strategy configuration
            regime_detector: Optional regime detector for adaptive parameters
        """
        self.config = config
        self.regime_detector = regime_detector

        # Volatility forecaster
        if config.vol_forecast_method == "garch":
            self.vol_forecaster = GARCHVolatility(p=1, q=1)
        elif config.vol_forecast_method == "bayesian_lstm":
            from volatility_arbitrage.models.bayesian_forecaster import BayesianLSTMForecaster
            self.vol_forecaster = BayesianLSTMForecaster(
                hidden_size=config.bayesian_lstm_hidden_size,
                dropout_p=config.bayesian_lstm_dropout_p,
                sequence_length=config.bayesian_lstm_sequence_length,
                n_mc_samples=config.bayesian_lstm_n_mc_samples,
            )
        else:
            from volatility_arbitrage.models.volatility import HistoricalVolatility
            self.vol_forecaster = HistoricalVolatility(window=config.vol_lookback_period)

        # Uncertainty-adjusted position sizing (Phase 2)
        self._last_uncertainty: Decimal = Decimal("0")
        if config.use_uncertainty_sizing:
            self.uncertainty_sizer: Optional[UncertaintySizer] = UncertaintySizer(
                UncertaintySizingConfig(
                    uncertainty_penalty=config.uncertainty_penalty,
                    min_position_pct=config.uncertainty_min_position_pct,
                    max_position_pct=config.uncertainty_max_position_pct,
                )
            )
        else:
            self.uncertainty_sizer = None

        # Track price history for volatility forecasting
        self.price_history: dict[str, pd.Series] = {}

        # Track option positions and their entry details
        self.option_positions: dict[str, dict] = {}  # symbol -> position details

        # Regime detection state
        self.current_regime: Optional[int] = None
        self.regime_history: list[tuple[datetime, int]] = []  # (timestamp, regime)
        self.returns_for_regime: dict[str, pd.Series] = {}  # symbol -> returns

        # QV strategy state (only used if use_qv_strategy=True)
        self.qv_buffers: Dict[str, FeatureBuffers] = {}  # symbol -> feature buffers
        self.last_consensus: Dict[str, Decimal] = {}  # symbol -> last consensus score

        # Entry tracking for tiered sizing and holding period
        self.entry_timestamps: Dict[str, datetime] = {}  # symbol -> entry time
        self.entry_consensus: Dict[str, Decimal] = {}    # symbol -> consensus at entry

        # Validate regime configuration
        if config.use_regime_detection:
            if regime_detector is None:
                logger.warning("use_regime_detection=True but no regime_detector provided")
                config.use_regime_detection = False
            elif config.regime_params is None or len(config.regime_params) == 0:
                logger.warning("use_regime_detection=True but no regime_params provided")
                config.use_regime_detection = False

        # Validate bullish base exposure parameters
        if not (Decimal("-2.0") <= self.config.base_long_bias <= Decimal("2.0")):
            raise ValueError(
                f"base_long_bias must be in range [-2.0, 2.0], got {self.config.base_long_bias}"
            )

        if not (Decimal("0.0") <= self.config.signal_adjustment_factor <= Decimal("2.0")):
            raise ValueError(
                f"signal_adjustment_factor must be in range [0.0, 2.0], got {self.config.signal_adjustment_factor}"
            )

        # Validate exposure range consistency
        max_possible_exposure = (
            self.config.base_long_bias + self.config.signal_adjustment_factor
        ) * self.config.regime_extreme_low_scalar  # Maximum regime scalar

        min_possible_exposure = (
            self.config.base_long_bias - self.config.signal_adjustment_factor
        ) * self.config.regime_crisis_scalar  # Minimum with crisis scaling

        logger.info(
            f"Exposure range with current params: "
            f"[{min_possible_exposure:.2f}, {max_possible_exposure:.2f}]"
        )

        if max_possible_exposure > Decimal("2.0"):
            logger.warning(
                f"Maximum possible exposure ({max_possible_exposure:.2f}) exceeds typical limits. "
                f"Consider reducing base_long_bias or regime_extreme_low_scalar."
            )

        logger.info(
            "Initialized VolatilityArbitrageStrategy",
            extra=self.config.to_dict()
        )

    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: dict[str, Position],
        # ADD THESE ARGUMENTS:
        cash: float = 0.0,
        portfolio_greeks: Optional[dict] = None,
        **kwargs
    ) -> list[Signal]:

        """
        Generate trading signals based on volatility spread.

        Integrates regime detection to apply adaptive thresholds and sizing.

        Args:
            timestamp: Current timestamp
            market_data: Market data (must include option chain data)
            positions: Current positions

        Returns:
            List of trading signals
        """
        signals: list[Signal] = []

        # Extract symbols with option data
        if "option_chain" not in market_data.columns:
            logger.warning("No option chain data in market_data")
            return signals

        # Detect regime if enabled (use first symbol for market-wide regime)
        primary_symbol = market_data["symbol"].iloc[0] if len(market_data) > 0 else None
        if self.config.use_regime_detection and primary_symbol:
            detected_regime = self._detect_current_regime(primary_symbol, timestamp)

            if detected_regime is not None:
                # Check for regime transition
                if self.current_regime is not None and self.current_regime != detected_regime:
                    # Handle regime transition
                    transition_signals = self._handle_regime_transition(detected_regime, positions)
                    signals.extend(transition_signals)

                    # Record transition
                    self.regime_history.append((timestamp, detected_regime))

                # Update current regime
                self.current_regime = detected_regime

                logger.debug(
                    f"Current regime: {self.current_regime}",
                    extra={"regime": self.current_regime, "timestamp": timestamp}
                )

        # Process each symbol
        for symbol in market_data["symbol"].unique():
            symbol_data = market_data[market_data["symbol"] == symbol]

            if symbol_data.empty:
                continue

            # Update price history
            current_price = symbol_data.iloc[0].get("close")
            if current_price is not None:
                self._update_price_history(symbol, timestamp, float(current_price))

            # Get option chain
            option_chain = symbol_data.iloc[0].get("option_chain")
            if option_chain is None:
                continue

            # Calculate volatility spread (may be None if insufficient price history)
            vol_spread = self._calculate_volatility_spread(
                symbol, timestamp, option_chain
            )

            # Get regime-specific parameters
            entry_threshold, exit_threshold, position_multiplier = self._get_regime_parameters(
                self.current_regime
            )

            # Check for entry signals
            # QV strategy can run without vol_spread; legacy requires it
            if self.config.use_qv_strategy or vol_spread is not None:
                entry_signals = self._check_entry_signals(
                    timestamp=timestamp,
                    symbol=symbol,
                    option_chain=option_chain,
                    positions=positions,
                    cash=Decimal(str(cash)),
                    vol_spread=vol_spread,
                    entry_threshold=entry_threshold,
                    position_multiplier=position_multiplier
                )
                signals.extend(entry_signals)

            # Check for exit signals (requires vol_spread for legacy strategy)
            if vol_spread is not None:
                exit_signals = self._check_exit_signals(
                    vol_spread, positions, exit_threshold
                )
                signals.extend(exit_signals)

            # Check for delta rebalancing
            rebalance_signals = self._check_delta_rebalancing(
                symbol, option_chain, positions
            )
            signals.extend(rebalance_signals)

        return signals

    def _update_price_history(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
    ) -> None:
        """Update price history for volatility forecasting."""
        if symbol not in self.price_history:
            self.price_history[symbol] = pd.Series(dtype=float)

        self.price_history[symbol][timestamp] = price

        # Keep only recent history
        max_history = max(252, self.config.vol_lookback_period * 2)
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol].iloc[-max_history:]

    def _calculate_volatility_spread(
        self,
        symbol: str,
        timestamp: datetime,
        option_chain: OptionChain,
    ) -> Optional[VolatilitySpread]:
        """
        Calculate volatility spread between IV and forecasted RV.

        Args:
            symbol: Underlying symbol
            timestamp: Current timestamp
            option_chain: Option chain data

        Returns:
            VolatilitySpread or None if calculation fails
        """
        # Get price history
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.config.vol_lookback_period:
            logger.debug(f"Insufficient price history for {symbol}")
            return None

        # Calculate returns
        prices = self.price_history[symbol]
        returns = calculate_returns(prices, method="log")

        if len(returns) < self.config.vol_lookback_period:
            return None

        # Forecast realized volatility (with uncertainty if available)
        try:
            forecast_result = self.vol_forecaster.forecast_with_uncertainty(returns, horizon=1)
            forecasted_vol = forecast_result["mean_vol"]
            self._last_uncertainty = forecast_result.get("epistemic_uncertainty", Decimal("0"))
        except Exception as e:
            logger.warning(f"Volatility forecast failed for {symbol}: {e}")
            return None

        # Get ATM implied volatility
        atm_strike = option_chain.get_atm_strike()
        if atm_strike is None:
            return None

        # Find ATM call and put
        atm_call = next((c for c in option_chain.calls if c.strike == atm_strike), None)
        atm_put = next((p for p in option_chain.puts if p.strike == atm_strike), None)

        if atm_call is None or atm_put is None:
            return None

        if atm_call.implied_volatility is None or atm_put.implied_volatility is None:
            return None

        # Average call and put IV
        implied_vol = (atm_call.implied_volatility + atm_put.implied_volatility) / Decimal("2")

        # Calculate spread
        spread = implied_vol - forecasted_vol
        spread_pct = (spread / forecasted_vol) * Decimal("100") if forecasted_vol > 0 else Decimal("0")

        vol_spread = VolatilitySpread(
            symbol=symbol,
            timestamp=timestamp,
            implied_vol=implied_vol,
            forecasted_vol=forecasted_vol,
            spread=spread,
            spread_pct=spread_pct,
        )

        logger.debug(
            f"Volatility spread for {symbol}",
            extra={
                "symbol": symbol,
                "iv": float(implied_vol),
                "rv": float(forecasted_vol),
                "spread_pct": float(spread_pct),
            }
        )

        return vol_spread

    def _check_entry_signals(
        self,
        timestamp: datetime,
        symbol: str,
        option_chain: OptionChain,
        positions: dict[str, Position],
        cash: Decimal,
        vol_spread: Optional[VolatilitySpread] = None,
        entry_threshold: Optional[Decimal] = None,
        position_multiplier: Optional[Decimal] = None,
    ) -> list[Signal]:
        """
        Check for entry signals using either legacy or QV strategy.

        BACKWARD COMPATIBILITY: Branch based on config.use_qv_strategy

        Args:
            timestamp: Current timestamp
            symbol: Underlying symbol
            option_chain: Option chain data
            positions: Current positions
            cash: Available account equity
            vol_spread: Volatility spread (for legacy strategy)
            entry_threshold: Entry threshold (for legacy strategy)
            position_multiplier: Position multiplier (for legacy strategy)

        Returns:
            List of entry signals
        """
        if self.config.use_qv_strategy:
            # QV strategy: Use consensus-based entry logic
            return self._generate_qv_entry_logic(timestamp, option_chain, cash)
        else:
            # Legacy strategy: Use IV-RV spread logic
            if vol_spread is None or entry_threshold is None or position_multiplier is None:
                logger.warning("Legacy strategy requires vol_spread, entry_threshold, and position_multiplier")
                return []
            return self._generate_legacy_entry_signals(
                vol_spread, option_chain, positions, entry_threshold, position_multiplier
            )

    def _generate_legacy_entry_signals(
        self,
        vol_spread: VolatilitySpread,
        option_chain: OptionChain,
        positions: dict[str, Position],
        entry_threshold: Decimal,
        position_multiplier: Decimal,
    ) -> list[Signal]:
        """
        Legacy IV-RV spread strategy for entry signals.

        This is the original naive strategy logic for comparison.

        Args:
            vol_spread: Calculated volatility spread
            option_chain: Option chain data
            positions: Current positions
            entry_threshold: Regime-specific entry threshold
            position_multiplier: Regime-specific position size multiplier

        Returns:
            List of entry signals
        """
        signals: list[Signal] = []

        # Check if already have position
        if vol_spread.symbol in self.option_positions:
            return signals

        # Check position limits
        if len(self.option_positions) >= self.config.max_positions:
            return signals

        # Check time to expiry
        days_to_expiry = (option_chain.expiry - vol_spread.timestamp).days
        if days_to_expiry < self.config.min_days_to_expiry:
            return signals
        if days_to_expiry > self.config.max_days_to_expiry:
            return signals

        # Check entry threshold (regime-aware)
        if abs(vol_spread.spread_pct) < entry_threshold:
            return signals

        # Generate signals based on spread direction
        atm_strike = option_chain.get_atm_strike()
        if atm_strike is None:
            return signals

        # Find ATM options
        atm_call = next((c for c in option_chain.calls if c.strike == atm_strike), None)
        atm_put = next((p for p in option_chain.puts if p.strike == atm_strike), None)

        if atm_call is None or atm_put is None:
            return signals

        # Determine position direction and size
        if vol_spread.is_short_opportunity:
            # IV > RV: Sell straddle
            action = "sell"
            reason = f"Short volatility: IV {vol_spread.implied_vol:.1%} > RV {vol_spread.forecasted_vol:.1%}"
        else:
            # IV < RV: Buy straddle
            action = "buy"
            reason = f"Long volatility: IV {vol_spread.implied_vol:.1%} < RV {vol_spread.forecasted_vol:.1%}"

        # Position sizing with regime-aware multiplier
        base_quantity = 1
        quantity = int(base_quantity * float(position_multiplier))
        quantity = max(1, quantity)  # Ensure at least 1 contract

        if self.current_regime is not None:
            reason += f" (regime {self.current_regime}, multiplier {position_multiplier:.2f})"

        # Create signals for straddle (call + put at same strike)
        signals.append(Signal(
            symbol=f"{vol_spread.symbol}_CALL_{atm_strike}_{option_chain.expiry.strftime('%Y%m%d')}",
            action=action,
            quantity=quantity,
            reason=reason
        ))

        signals.append(Signal(
            symbol=f"{vol_spread.symbol}_PUT_{atm_strike}_{option_chain.expiry.strftime('%Y%m%d')}",
            action=action,
            quantity=quantity,
            reason=reason
        ))

        # Calculate initial delta and add hedge
        call_greeks = BlackScholesModel.greeks(
            S=option_chain.underlying_price,
            K=atm_call.strike,
            T=option_chain.time_to_expiry,
            r=option_chain.risk_free_rate,
            sigma=atm_call.implied_volatility or Decimal("0.2"),
            option_type=OptionType.CALL
        )

        put_greeks = BlackScholesModel.greeks(
            S=option_chain.underlying_price,
            K=atm_put.strike,
            T=option_chain.time_to_expiry,
            r=option_chain.risk_free_rate,
            sigma=atm_put.implied_volatility or Decimal("0.2"),
            option_type=OptionType.PUT
        )

        # Net delta of straddle position
        position_multiplier = 1 if action == "buy" else -1
        net_delta = (call_greeks.delta + put_greeks.delta) * Decimal(quantity) * Decimal(position_multiplier)

        # Delta-neutral hedge (was 0.20 which introduced long bias)
        TARGET_DELTA_PER_CONTRACT = Decimal("0.0") 
        
        target_shares = int(Decimal(quantity) * 100 * TARGET_DELTA_PER_CONTRACT)
        
        hedge_quantity = target_shares - int(net_delta * 100)

        if hedge_quantity != 0:
             # If positive, we need to buy shares (Long Bias)
             # If negative, we sell (unlikely with this logic)
             hedge_action = "buy" if hedge_quantity > 0 else "sell"
             
             signals.append(Signal(
                symbol=vol_spread.symbol,
                action=hedge_action,
                quantity=abs(hedge_quantity),
                reason=f"Structural Long Hedge: Target {target_shares} shares"
            ))

        # Track position entry
        self.option_positions[vol_spread.symbol] = {
            "entry_timestamp": vol_spread.timestamp,
            "entry_iv": vol_spread.implied_vol,
            "entry_rv": vol_spread.forecasted_vol,
            "direction": action,
            "strike": atm_strike,
            "expiry": option_chain.expiry,
        }

        logger.info(
            f"Volatility arbitrage entry: {action} {vol_spread.symbol} straddle",
            extra={
                "symbol": vol_spread.symbol,
                "action": action,
                "strike": float(atm_strike),
                "spread_pct": float(vol_spread.spread_pct),
            }
        )

        for s in signals:
            print(f"DEBUG SIGNAL: {s.symbol} {s.action} {s.quantity} ({s.reason})")
            
        return signals

    def _check_exit_signals(
        self,
        vol_spread: VolatilitySpread,
        positions: dict[str, Position],
        exit_threshold: Decimal,
    ) -> list[Signal]:
        """
        Check for position exit signals.

        Args:
            vol_spread: Calculated volatility spread
            positions: Current positions
            exit_threshold: Regime-specific exit threshold

        Returns:
            List of exit signals
        """
        signals: list[Signal] = []

        # Check minimum holding period (tiered sizing feature)
        if not self._check_holding_period(vol_spread.symbol, vol_spread.timestamp):
            return []  # Cannot exit yet - must hold for min_holding_days

        # Check if we have a position in this symbol
        if vol_spread.symbol not in self.option_positions:
            return signals

        position_info = self.option_positions[vol_spread.symbol]

        # Check exit threshold: spread has converged (regime-aware)
        if abs(vol_spread.spread_pct) < exit_threshold:
            # Close position
            signals.extend(self._generate_close_signals(vol_spread.symbol, positions, "Spread converged"))
            return signals

        # Check if spread reversed (became unfavorable)
        entry_direction = position_info["direction"]
        if entry_direction == "buy" and vol_spread.is_short_opportunity:
            # We're long vol but IV > RV now
            signals.extend(self._generate_close_signals(vol_spread.symbol, positions, "Spread reversed"))
            return signals
        elif entry_direction == "sell" and vol_spread.is_long_opportunity:
            # We're short vol but IV < RV now
            signals.extend(self._generate_close_signals(vol_spread.symbol, positions, "Spread reversed"))
            return signals

        # TODO: Check stop loss based on P&L

        return signals

    def _check_delta_rebalancing(
        self,
        symbol: str,
        option_chain: OptionChain,
        positions: dict[str, Position],
    ) -> list[Signal]:
        """
        Check if delta rebalancing is needed.

        Args:
            symbol: Underlying symbol
            option_chain: Current option chain
            positions: Current positions

        Returns:
            List of rebalancing signals
        """
        signals: list[Signal] = []

        # Check if we have option positions for this symbol
        if symbol not in self.option_positions:
            return signals

        # Calculate current portfolio delta
        # TODO: Implement portfolio delta calculation from positions

        return signals

    def _generate_close_signals(
        self,
        symbol: str,
        positions: dict[str, Position],
        reason: str,
    ) -> list[Signal]:
        """
        Generate signals to close all positions for a symbol.

        Args:
            symbol: Symbol to close
            positions: Current positions
            reason: Reason for closing

        Returns:
            List of closing signals
        """
        signals: list[Signal] = []

        # Find all positions related to this symbol
        for pos_symbol, pos in positions.items():
            if symbol in pos_symbol:
                # Opposite action to close
                action = "sell" if pos.quantity > 0 else "buy"
                signals.append(Signal(
                    symbol=pos_symbol,
                    action=action,
                    quantity=abs(pos.quantity),
                    reason=f"Close position: {reason}"
                ))

        # Remove from tracking
        if symbol in self.option_positions:
            del self.option_positions[symbol]

        # Clean up entry tracking (tiered sizing feature)
        self.entry_timestamps.pop(symbol, None)
        self.entry_consensus.pop(symbol, None)

        logger.info(
            f"Closing volatility arbitrage position for {symbol}",
            extra={"symbol": symbol, "reason": reason}
        )

        return signals

    def _detect_current_regime(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Optional[int]:
        """
        Detect current market regime for a symbol.

        Args:
            symbol: Symbol to analyze
            timestamp: Current timestamp

        Returns:
            Regime ID or None if detection fails
        """
        if not self.config.use_regime_detection or self.regime_detector is None:
            return None

        # Calculate returns for regime detection
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.config.regime_lookback_period:
            return None

        # Use recent returns for regime detection
        prices = self.price_history[symbol].tail(self.config.regime_lookback_period)
        returns = calculate_returns(prices, method="log")

        if len(returns) < 10:  # Minimum data requirement
            return None

        try:
            # Predict current regime
            regime_series = self.regime_detector.predict(returns)
            current_regime = int(regime_series.iloc[-1])

            return current_regime

        except Exception as e:
            logger.warning(f"Regime detection failed for {symbol}: {e}")
            return None

    def _handle_regime_transition(
        self,
        new_regime: int,
        positions: dict[str, Position],
    ) -> list[Signal]:
        """
        Handle regime transition by optionally closing positions.

        Args:
            new_regime: New regime ID
            positions: Current positions

        Returns:
            List of exit signals if configured to exit on transition
        """
        signals: list[Signal] = []

        if self.config.exit_on_regime_transition:
            # Close all option positions on regime change
            for symbol in list(self.option_positions.keys()):
                signals.extend(
                    self._generate_close_signals(
                        symbol, positions, f"Regime transition: {self.current_regime} -> {new_regime}"
                    )
                )

        logger.info(
            f"Regime transition detected: {self.current_regime} -> {new_regime}",
            extra={
                "old_regime": self.current_regime,
                "new_regime": new_regime,
                "exit_on_transition": self.config.exit_on_regime_transition,
            }
        )

        return signals

    def _get_regime_parameters(self, regime: Optional[int]) -> tuple[Decimal, Decimal, Decimal]:
        """
        Get regime-specific parameters or baseline values.

        Args:
            regime: Current regime ID or None

        Returns:
            Tuple of (entry_threshold, exit_threshold, position_multiplier)
        """
        if regime is not None and self.config.regime_params and regime in self.config.regime_params:
            params = self.config.regime_params[regime]
            return (
                params.entry_threshold_pct,
                params.exit_threshold_pct,
                params.position_size_multiplier,
            )
        else:
            # Use baseline parameters
            return (
                self.config.entry_threshold_pct,
                self.config.exit_threshold_pct,
                Decimal("1.0"),
            )

    # ===== QV STRATEGY METHODS =====

    def _has_multiple_expiries(self, option_chain: OptionChain) -> bool:
        """
        Check if option_chain contains options with multiple expiries.

        Used for term structure signal - need at least 2 different DTEs.

        Args:
            option_chain: Option chain to check

        Returns:
            True if multiple expiries detected
        """
        if not option_chain.calls:
            return False

        # Get unique expiry dates from option chain
        unique_expiries = set()
        for call in option_chain.calls:
            unique_expiries.add(call.expiry.date())

        # Need at least 2 different expiries for term structure
        return len(unique_expiries) >= 2

    def _get_iv_for_dte(self, option_chain: OptionChain, target_dte: int) -> Decimal:
        """
        Get ATM IV for options closest to target DTE.

        Args:
            option_chain: Option chain
            target_dte: Target days to expiry

        Returns:
            ATM implied volatility for closest expiry, or Decimal("0") if not found
        """
        atm_strike = option_chain.underlying_price
        current_time = option_chain.timestamp

        # Filter options by DTE range (±5 days tolerance) and near ATM
        candidates = []
        for opt in option_chain.calls:
            # Calculate DTE in days
            dte_days = (opt.expiry - current_time).days

            # Check if within target DTE range and near ATM
            if (abs(dte_days - target_dte) <= 5 and
                abs(opt.strike - atm_strike) < atm_strike * Decimal("0.05")):
                candidates.append(opt)

        if not candidates:
            return Decimal("0")

        # Return IV of closest ATM option
        atm_option = min(candidates, key=lambda opt: abs(opt.strike - atm_strike))
        return atm_option.implied_volatility or Decimal("0")

    def _extract_daily_features(self, option_chain: OptionChain) -> Dict[str, Decimal]:
        """
        Extract daily features from option chain for QV strategy.

        Features:
        1. PC Ratio: Put Volume / Call Volume
        2. IV Skew: ATM Put IV - ATM Call IV
        3. IV Premium: IV - RV
        4. Term Structure: (60d IV - 30d IV) / 30
        5. Volume Ratio: Current Volume / 20d Median
        6. Near-term Sentiment: 7d IV - 30d IV
        7. RV 20d: Realized volatility

        Args:
            option_chain: Current option chain

        Returns:
            Dictionary with feature names → values
        """
        features: Dict[str, Decimal] = {}

        # 1. PC Ratio (Put Volume / Call Volume)
        total_put_volume = sum(float(p.volume or 0) for p in option_chain.puts)
        total_call_volume = sum(float(c.volume or 0) for c in option_chain.calls)
        features['pc_ratio'] = (
            Decimal(str(total_put_volume / total_call_volume))
            if total_call_volume > 0 else Decimal("1.0")
        )

        # 2. IV Skew (ATM Put IV - ATM Call IV)
        atm_strike = option_chain.underlying_price
        atm_put = min(option_chain.puts, key=lambda p: abs(p.strike - atm_strike), default=None)
        atm_call = min(option_chain.calls, key=lambda c: abs(c.strike - atm_strike), default=None)

        if atm_put and atm_call and atm_put.implied_volatility and atm_call.implied_volatility:
            features['iv_skew'] = atm_put.implied_volatility - atm_call.implied_volatility
        else:
            features['iv_skew'] = Decimal("0")

        # 3. IV Percentile (replaces backward-looking IV Premium)
        # IMPROVEMENT: Use forward-looking IV percentile instead of IV - RV
        # IV Percentile = current ATM IV / 252-day IV history percentile
        # This measures if IV is cheap/expensive relative to its own history
        symbol = option_chain.symbol

        # Get ATM IV average
        if atm_put and atm_call and atm_put.implied_volatility and atm_call.implied_volatility:
            atm_iv = (atm_put.implied_volatility + atm_call.implied_volatility) / Decimal("2")
        else:
            atm_iv = Decimal("0")

        # Calculate IV percentile from 60-day IV history (stored in iv_premium buffer)
        # High percentile = IV expensive = contrarian bullish (fear overdone)
        # Low percentile = IV cheap = contrarian bearish (complacency)
        if symbol in self.qv_buffers and len(self.qv_buffers[symbol].iv_premium_60d) >= 20:
            iv_history = np.array(list(self.qv_buffers[symbol].iv_premium_60d))
            # Calculate percentile rank of current IV
            if atm_iv > 0:
                iv_percentile = (iv_history <= float(atm_iv)).sum() / len(iv_history)
                features['iv_premium'] = Decimal(str(iv_percentile))  # 0.0-1.0 scale
            else:
                features['iv_premium'] = Decimal("0.5")  # Neutral
        else:
            features['iv_premium'] = Decimal("0.5")  # Neutral until enough data

        # Store ATM IV for history tracking (instead of iv_premium = atm_iv - rv)
        features['atm_iv'] = atm_iv

        # Still calculate RV for regime detection
        rv_20d = self._calculate_realized_vol_20d(symbol)
        features['rv_20d'] = rv_20d

        # 4. Term Structure (60d IV - 30d IV) / 30 days
        # USER DECISION: Implement with robust fallback
        if self._has_multiple_expiries(option_chain):
            iv_30d = self._get_iv_for_dte(option_chain, target_dte=30)
            iv_60d = self._get_iv_for_dte(option_chain, target_dte=60)
            if iv_30d > 0 and iv_60d > 0:
                features['term_structure'] = (iv_60d - iv_30d) / Decimal("30")
            else:
                features['term_structure'] = Decimal("0")
        else:
            features['term_structure'] = Decimal("0")  # Neutral signal if data unavailable

        # 5. Volume Ratio (Current Volume / 20d Median Volume)
        current_volume = total_put_volume + total_call_volume
        if symbol in self.qv_buffers and len(self.qv_buffers[symbol].volume_ratio_60d) >= 20:
            volume_history = list(self.qv_buffers[symbol].volume_ratio_60d)[-20:]
            median_volume = Decimal(str(np.median(volume_history)))
            features['volume_ratio'] = (
                Decimal(str(current_volume)) / median_volume
                if median_volume > 0 else Decimal("1.0")
            )
        else:
            features['volume_ratio'] = Decimal("1.0")  # Neutral until enough data

        # 6. Near-term Sentiment (7d IV - 30d IV)
        if self._has_multiple_expiries(option_chain):
            iv_7d = self._get_iv_for_dte(option_chain, target_dte=7)
            iv_30d = self._get_iv_for_dte(option_chain, target_dte=30)
            if iv_7d > 0 and iv_30d > 0:
                features['near_term_sentiment'] = iv_7d - iv_30d
            else:
                features['near_term_sentiment'] = Decimal("0")
        else:
            features['near_term_sentiment'] = Decimal("0")  # Neutral if unavailable

        return features

    def _calculate_realized_vol_20d(self, symbol: str) -> Decimal:
        """
        Calculate 20-day realized volatility from price history.

        IMPORTANT: Uses lagged data to avoid look-ahead bias.
        When making trading decisions at time T, we can only use
        returns through T-1 (today's return isn't known yet).

        Args:
            symbol: Underlying symbol

        Returns:
            Annualized realized volatility (stdev of log returns)
        """
        # Need 22 prices: 21 for lagged calculation + 1 for today (excluded)
        if symbol not in self.price_history or len(self.price_history[symbol]) < 22:
            return Decimal("0.20")  # Default 20% vol

        # Get last 22 prices, then exclude today (last price) to avoid look-ahead bias
        # This gives us 21 prices through yesterday, yielding 20 returns
        prices = self.price_history[symbol].tail(22).iloc[:-1]  # Exclude today
        returns = calculate_returns(prices, method="log")

        if len(returns) < 20:
            return Decimal("0.20")

        # Calculate annualized volatility
        daily_vol = float(returns.std())
        annualized_vol = daily_vol * np.sqrt(252)

        return Decimal(str(annualized_vol))

    def _update_qv_features(self, symbol: str, features: Dict[str, Decimal]) -> None:
        """
        Update rolling buffers with daily features.

        Args:
            symbol: Asset symbol
            features: Dictionary from _extract_daily_features()
        """
        # Initialize buffers if needed
        if symbol not in self.qv_buffers:
            self.qv_buffers[symbol] = FeatureBuffers()

        buffers = self.qv_buffers[symbol]

        # Update 60-day feature buffers
        buffers.pc_ratio_60d.append(float(features['pc_ratio']))
        buffers.iv_skew_60d.append(float(features['iv_skew']))
        # Store ATM IV for percentile calculation (not the percentile itself)
        buffers.iv_premium_60d.append(float(features.get('atm_iv', features['iv_premium'])))
        buffers.term_structure_60d.append(float(features['term_structure']))
        buffers.volume_ratio_60d.append(float(features['volume_ratio']))
        buffers.near_term_sentiment_60d.append(float(features['near_term_sentiment']))

        # Update 252-day RV buffer for regime detection
        buffers.rv_252d.append(float(features['rv_20d']))

    def _calculate_z_score(self, value: Decimal, buffer: deque) -> Decimal:
        """
        Calculate z-score: (value - median) / stdev

        Args:
            value: Current value
            buffer: 60-day historical buffer

        Returns:
            Z-score (standardized value)
        """
        if len(buffer) < 20:  # Minimum data requirement
            return Decimal("0")

        buffer_array = np.array(list(buffer))
        median = np.median(buffer_array)
        stdev = np.std(buffer_array)

        if stdev == 0:
            return Decimal("0")

        z_score = (float(value) - median) / stdev
        return Decimal(str(z_score))

    def _generate_binary_signals(self, symbol: str, features: Dict[str, Decimal]) -> Dict[str, int]:
        """
        Generate 6 binary signals based on z-scores and thresholds.

        Returns:
            Dictionary with signal names mapped to -1 (bearish), 0 (neutral), +1 (bullish)
        """
        buffers = self.qv_buffers[symbol]
        signals: Dict[str, int] = {}

        # 1. PC Ratio Signal (Contrarian: High PC Ratio → Bullish)
        # Uses config threshold for z-score comparison
        pc_z = self._calculate_z_score(features['pc_ratio'], buffers.pc_ratio_60d)
        pc_thresh = self.config.pc_ratio_threshold  # From config (e.g., 0.3)
        if pc_z > pc_thresh:  # High PC ratio (fear) → Bullish
            signals['pc_ratio'] = 1
        elif pc_z < -pc_thresh:  # Low PC ratio (greed) → Bearish
            signals['pc_ratio'] = -1
        else:
            signals['pc_ratio'] = 0

        # 2. IV Skew Signal (Contrarian: High Skew → Bullish)
        # Uses config threshold for z-score comparison
        skew_z = self._calculate_z_score(features['iv_skew'], buffers.iv_skew_60d)
        skew_thresh = self.config.skew_threshold  # From config (e.g., 0.3)
        if skew_z > skew_thresh:  # High skew (put premium) → Bullish
            signals['iv_skew'] = 1
        elif skew_z < -skew_thresh:  # Low skew → Bearish
            signals['iv_skew'] = -1
        else:
            signals['iv_skew'] = 0

        # 3. IV Percentile Signal (Contrarian: High IV Percentile → Bullish)
        # IV Percentile is already in 0.0-1.0 range from _extract_daily_features
        # Uses config threshold as offset from 0.5 neutral point
        iv_percentile = features['iv_premium']  # Now stores percentile 0.0-1.0
        prem_thresh = self.config.premium_threshold  # From config (e.g., 0.05)
        high_pct = Decimal("0.5") + prem_thresh  # e.g., 0.55 with 0.05 threshold
        low_pct = Decimal("0.5") - prem_thresh   # e.g., 0.45 with 0.05 threshold
        if iv_percentile > high_pct:  # High IV percentile → Bullish (fear overdone)
            signals['iv_premium'] = 1
        elif iv_percentile < low_pct:  # Low IV percentile → Bearish (complacency)
            signals['iv_premium'] = -1
        else:
            signals['iv_premium'] = 0

        # 4. Term Structure Signal (Positive slope → Bullish)
        # Already uses config threshold
        if features['term_structure'] > self.config.term_structure_threshold:
            signals['term_structure'] = 1  # Upward sloping → Bullish
        elif features['term_structure'] < -self.config.term_structure_threshold:
            signals['term_structure'] = -1  # Inverted → Bearish
        else:
            signals['term_structure'] = 0

        # 5. Volume Spike Signal (High volume → continuation)
        # Uses config threshold for z-score comparison
        volume_z = self._calculate_z_score(features['volume_ratio'], buffers.volume_ratio_60d)
        vol_thresh = self.config.volume_spike_threshold  # From config (e.g., 0.7)
        if volume_z > vol_thresh:  # Unusual high volume
            signals['volume_spike'] = 1  # Assume bullish continuation
        elif volume_z < -vol_thresh / 2:  # Unusually low volume (half threshold)
            signals['volume_spike'] = -1  # Bearish
        else:
            signals['volume_spike'] = 0

        # 6. Near-term Sentiment Signal (Negative → Bullish contrarian)
        # Uses config threshold for z-score comparison
        sentiment_z = self._calculate_z_score(features['near_term_sentiment'], buffers.near_term_sentiment_60d)
        sent_thresh = self.config.sentiment_threshold  # From config (e.g., 0.3)
        if sentiment_z < -sent_thresh:  # Negative near-term sentiment → Bullish
            signals['near_term_sentiment'] = 1
        elif sentiment_z > sent_thresh:  # Positive near-term sentiment → Bearish
            signals['near_term_sentiment'] = -1
        else:
            signals['near_term_sentiment'] = 0

        return signals

    def _calculate_consensus_score(self, binary_signals: Dict[str, int]) -> Decimal:
        """
        Calculate weighted consensus score from binary signals.

        Args:
            binary_signals: Dict with signal names → {-1, 0, +1}

        Returns:
            Consensus score in range [-1.0, +1.0]
        """
        weighted_sum = (
            Decimal(str(binary_signals['pc_ratio'])) * self.config.weight_pc_ratio +
            Decimal(str(binary_signals['iv_skew'])) * self.config.weight_iv_skew +
            Decimal(str(binary_signals['iv_premium'])) * self.config.weight_iv_premium +
            Decimal(str(binary_signals['term_structure'])) * self.config.weight_term_structure +
            Decimal(str(binary_signals['volume_spike'])) * self.config.weight_volume_spike +
            Decimal(str(binary_signals['near_term_sentiment'])) * self.config.weight_near_term_sentiment
        )

        # Normalize to [-1.0, +1.0] range
        # Maximum possible weighted sum = 1.0 (all signals +1 with weights summing to 1.0)
        consensus = weighted_sum  # Already normalized if weights sum to 1.0

        # Clamp to valid range
        consensus = max(Decimal("-1.0"), min(Decimal("1.0"), consensus))

        return consensus

    def _calculate_position_multiplier(self, consensus: Decimal) -> Decimal:
        """
        Calculate position size multiplier based on signal strength.

        Uses configurable scaling (linear/quadratic/cubic) to reward
        high-conviction signals with larger positions.

        Quadratic scaling rewards stronger signals disproportionately:
        - consensus = 0.3 (min threshold) → multiplier = 0.00 (no trade)
        - consensus = 0.4 → multiplier = 0.02 (2% position)
        - consensus = 0.5 → multiplier = 0.08 (8% position)
        - consensus = 0.6 → multiplier = 0.18 (18% position)
        - consensus = 0.7 → multiplier = 0.33 (33% position)
        - consensus = 0.8 → multiplier = 0.51 (51% position)
        - consensus = 0.9 → multiplier = 0.73 (73% position)
        - consensus = 1.0 → multiplier = 1.00 (100% position)

        Args:
            consensus: Consensus score [-1.0, +1.0]

        Returns:
            Position multiplier [0.0, 1.0] where 0 = no trade, 1 = full position
        """
        abs_consensus = abs(consensus)
        min_thresh = self.config.min_consensus_threshold

        # Below minimum threshold = no trade
        if abs_consensus < min_thresh:
            return Decimal("0")

        # Normalize to 0-1 range above threshold
        normalized = (abs_consensus - min_thresh) / (Decimal("1.0") - min_thresh)

        # Apply scaling method
        if self.config.position_scaling_method == "linear":
            return normalized
        elif self.config.position_scaling_method == "quadratic":
            return normalized ** 2
        elif self.config.position_scaling_method == "cubic":
            return normalized ** 3
        else:
            return normalized  # Default to linear

    def _check_holding_period(self, symbol: str, current_time: datetime) -> bool:
        """
        Check if minimum holding period has elapsed.

        Args:
            symbol: Asset symbol
            current_time: Current timestamp

        Returns:
            True if can exit, False if must continue holding
        """
        if symbol not in self.entry_timestamps:
            return True

        entry_time = self.entry_timestamps[symbol]
        days_held = (current_time - entry_time).days

        return days_held >= self.config.min_holding_days

    def _get_leverage_multiplier(self, signal_direction: str) -> Decimal:
        """
        Get leverage multiplier based on signal direction.

        Conservative asymmetric leverage:
        - Short vol: 1.3x (limited upside, higher risk)
        - Long vol: 2.0x (unlimited upside, defined risk)
        - Neutral: 1.0x (no leverage on hedges)

        Args:
            signal_direction: "short_vol", "long_vol", or "neutral"

        Returns:
            Leverage multiplier as Decimal
        """
        if not self.config.use_leverage:
            return Decimal("1.0")

        if signal_direction == "short_vol":
            return self.config.short_vol_leverage
        elif signal_direction == "long_vol":
            return self.config.long_vol_leverage
        else:
            return Decimal("1.0")

    def _calculate_qv_position_size(
        self,
        symbol: str,
        consensus: Decimal,
        features: Dict[str, Decimal]
    ) -> Decimal:
        """
        Calculate position size (exposure scalar) based on consensus and vol regime.

        Args:
            symbol: Asset symbol
            consensus: Consensus score [-1.0, +1.0]
            features: Daily features (contains rv_20d)

        Returns:
            Exposure scalar (target delta as % of equity)
        """
        # 1. Calculate vol percentile (current RV vs 252-day history)
        buffers = self.qv_buffers[symbol]
        if len(buffers.rv_252d) < 60:  # Minimum data for percentile
            vol_percentile = Decimal("0.5")  # Assume normal regime
        else:
            rv_array = np.array(list(buffers.rv_252d))
            current_rv = float(features['rv_20d'])
            # Calculate percentile: what % of historical values are <= current value
            percentile = (rv_array <= current_rv).sum() / len(rv_array)
            vol_percentile = Decimal(str(percentile))

        # 2. Determine regime scalar based on percentile
        if vol_percentile > Decimal("0.90"):  # Crisis (>90th percentile)
            regime_scalar = self.config.regime_crisis_scalar  # 0.5x
        elif vol_percentile > Decimal("0.70"):  # Elevated (70-90th)
            regime_scalar = self.config.regime_elevated_scalar  # 0.75x
        elif vol_percentile > Decimal("0.30"):  # Normal (30-70th)
            regime_scalar = self.config.regime_normal_scalar  # 1.0x
        elif vol_percentile > Decimal("0.10"):  # Low (10-30th)
            regime_scalar = self.config.regime_low_scalar  # 1.2x
        else:  # Extreme Low (<10th percentile)
            regime_scalar = self.config.regime_extreme_low_scalar  # 1.5x

        # 3. Calculate exposure with bullish base bias
        # Formula: exposure = (base_long_bias + signal_adjustment) * regime_scalar
        # Where signal_adjustment = consensus * signal_adjustment_factor
        signal_adjustment = consensus * self.config.signal_adjustment_factor
        base_exposure = self.config.base_long_bias + signal_adjustment
        exposure = base_exposure * regime_scalar

        # 4. Clamp to user-specified limits (-1.0 to +1.5)
        # User preference: Allow shorting but prevent excessive leverage
        exposure = max(Decimal("-1.0"), min(Decimal("1.5"), exposure))

        # Log exposure components for debugging
        logger.debug(
            f"Exposure calculation: consensus={consensus:.2f}, "
            f"signal_adj={signal_adjustment:.2f}, base={base_exposure:.2f}, "
            f"regime={regime_scalar:.2f}, final={exposure:.2f}"
        )

        return exposure

    def _exposure_to_option_quantity(
        self,
        exposure: Decimal,
        account_equity: Decimal,
        underlying_price: Decimal,
        option_delta: Decimal,
        signal_direction: str = "neutral"
    ) -> int:
        """
        Convert exposure scalar to option contract quantity with leverage support.

        Position Sizing:
        - Exposure represents target delta as fraction of equity (e.g., 0.5 = 50%)
        - Apply leverage multiplier based on signal direction (Phase 2)
        - Cap at position_size_pct to prevent oversized single positions
        - Target Notional = min(exposure, position_size_pct/100) × Account Equity × Leverage

        Args:
            exposure: Exposure scalar (target delta as fraction of equity, e.g., 0.5 = 50%)
            account_equity: Current account equity (cash)
            underlying_price: Current underlying price
            option_delta: Delta of selected option
            signal_direction: "short_vol", "long_vol", or "neutral" (Phase 2)

        Returns:
            Number of contracts to trade
        """
        # Apply leverage multiplier (Phase 2)
        leverage_multiplier = self._get_leverage_multiplier(signal_direction)

        # Cap exposure at position_size_pct to limit single-trade risk
        # exposure is already a fraction (0.5 = 50%), position_size_pct is in percentage (5 = 5%)
        max_exposure = self.config.position_size_pct / Decimal("100")
        capped_exposure = min(abs(exposure), max_exposure)

        # Calculate target delta-adjusted notional WITH leverage
        # This is the total delta exposure we want in dollar terms
        target_delta_notional = capped_exposure * account_equity * leverage_multiplier

        # Each option contract controls 100 shares
        # Delta-adjusted notional = num_contracts * underlying_price * option_delta * 100
        # Solve for num_contracts:
        denominator = underlying_price * abs(option_delta) * Decimal("100")

        if denominator == 0:
            return 0

        num_contracts = target_delta_notional / denominator

        # Round to integer contracts (minimum 1 if we have a signal)
        result = max(1, int(num_contracts)) if num_contracts >= Decimal("0.5") else 0

        return result

    def _generate_qv_entry_signals(
        self,
        option_chain: OptionChain,
        cash: Decimal,
        consensus: Decimal,
        exposure: Decimal
    ) -> List[Signal]:
        """
        Generate entry signals based on QV consensus score.

        USER DECISION: Sell ATM Puts (bullish) / Sell ATM Calls (bearish)

        Args:
            option_chain: Current option chain
            cash: Available account equity
            consensus: Consensus score [-1.0, +1.0]
            exposure: Position size (delta exposure as % of equity)

        Returns:
            List of Signal objects for option trades
        """
        signals: List[Signal] = []

        # NOTE: Consensus threshold is checked in tiered sizing (_generate_qv_entry_logic)
        # using min_consensus_threshold. No need to check again here.

        # Determine direction based on consensus sign
        if consensus > Decimal("0"):
            # Bullish: Sell ATM Puts
            signals = self._create_long_delta_signals(option_chain, cash, exposure)
        elif consensus < Decimal("0"):
            # Bearish: Sell ATM Calls
            signals = self._create_short_delta_signals(option_chain, cash, exposure)

        return signals

    def _create_long_delta_signals(
        self,
        option_chain: OptionChain,
        cash: Decimal,
        exposure: Decimal
    ) -> List[Signal]:
        """
        Create signals for bullish position: Sell ATM Puts.

        Selling puts:
        - Collects premium (Short Vega)
        - Gains positive delta exposure
        - Benefits from rising prices and/or vol contraction

        Args:
            option_chain: Current option chain
            cash: Available account equity
            exposure: Position size (delta exposure as % of equity)

        Returns:
            List of Signal objects
        """
        signals: List[Signal] = []

        # Find ATM put
        atm_strike = option_chain.underlying_price
        atm_put = min(
            option_chain.puts,
            key=lambda p: abs(p.strike - atm_strike),
            default=None
        )

        if not atm_put:
            return signals

        # Calculate Greeks for position sizing
        greeks = BlackScholesModel.greeks(
            S=option_chain.underlying_price,
            K=atm_put.strike,
            T=option_chain.time_to_expiry,
            r=option_chain.risk_free_rate,
            sigma=atm_put.implied_volatility or Decimal("0.20"),
            option_type=OptionType.PUT
        )

        # Calculate quantity based on delta-adjusted notional
        quantity = self._exposure_to_option_quantity(
            exposure=exposure,
            account_equity=cash,
            underlying_price=option_chain.underlying_price,
            option_delta=greeks.delta,
            signal_direction="long_vol"  # Phase 2: Apply long vol leverage
        )

        if quantity > 0:
            # Format signal symbol for option
            signal_symbol = f"{option_chain.symbol}_PUT_{float(atm_put.strike):.0f}_{option_chain.expiry.strftime('%Y%m%d')}"

            signals.append(Signal(
                symbol=signal_symbol,
                action="sell",  # Sell put (Short position)
                quantity=quantity,
                reason=f"QV Bullish: Sell ATM Put | Exposure={float(exposure):.2f}"
            ))

        return signals

    def _create_short_delta_signals(
        self,
        option_chain: OptionChain,
        cash: Decimal,
        exposure: Decimal
    ) -> List[Signal]:
        """
        Create signals for bearish position: Sell ATM Calls.

        Selling calls:
        - Collects premium (Short Vega)
        - Gains negative delta exposure
        - Benefits from falling prices and/or vol contraction

        Args:
            option_chain: Current option chain
            cash: Available account equity
            exposure: Position size (delta exposure as % of equity)

        Returns:
            List of Signal objects
        """
        signals: List[Signal] = []

        # Find ATM call
        atm_strike = option_chain.underlying_price
        atm_call = min(
            option_chain.calls,
            key=lambda c: abs(c.strike - atm_strike),
            default=None
        )

        if not atm_call:
            return signals

        # Calculate Greeks
        greeks = BlackScholesModel.greeks(
            S=option_chain.underlying_price,
            K=atm_call.strike,
            T=option_chain.time_to_expiry,
            r=option_chain.risk_free_rate,
            sigma=atm_call.implied_volatility or Decimal("0.20"),
            option_type=OptionType.CALL
        )

        # Calculate quantity (use absolute value of exposure for bearish)
        quantity = self._exposure_to_option_quantity(
            exposure=abs(exposure),  # Magnitude only
            account_equity=cash,
            underlying_price=option_chain.underlying_price,
            option_delta=greeks.delta,
            signal_direction="short_vol"  # Phase 2: Apply short vol leverage
        )

        if quantity > 0:
            # Format signal symbol for option
            signal_symbol = f"{option_chain.symbol}_CALL_{float(atm_call.strike):.0f}_{option_chain.expiry.strftime('%Y%m%d')}"

            signals.append(Signal(
                symbol=signal_symbol,
                action="sell",  # Sell call (Short position)
                quantity=quantity,
                reason=f"QV Bearish: Sell ATM Call | Exposure={float(exposure):.2f}"
            ))

        return signals

    def _generate_qv_entry_logic(
        self,
        timestamp: datetime,
        option_chain: OptionChain,
        cash: Decimal
    ) -> List[Signal]:
        """
        QV strategy entry logic with tiered position sizing.

        Workflow:
        1. DTE filter (check min/max days to expiry)
        2. Extract daily features
        3. Update rolling buffers
        4. Generate binary signals
        5. Calculate consensus score
        6. Apply tiered position sizing (quadratic scaling)
        7. Calculate base position size with regime adaptation
        8. Generate entry signals (sell ATM puts/calls)

        Args:
            timestamp: Current timestamp
            option_chain: Current option chain
            cash: Available account equity

        Returns:
            List of trading signals
        """
        symbol = option_chain.symbol

        # Step 1: DTE filter
        days_to_expiry = (option_chain.expiry - timestamp).days
        if days_to_expiry < self.config.min_days_to_expiry:
            return []
        if days_to_expiry > self.config.max_days_to_expiry:
            return []

        # Step 2: Extract features
        features = self._extract_daily_features(option_chain)

        # Step 3: Update buffers
        self._update_qv_features(symbol, features)

        # Step 4: Check if we have enough data (reduced from 60 to 30 for more trades)
        if symbol not in self.qv_buffers or len(self.qv_buffers[symbol].rv_252d) < 30:
            return []  # Not enough data yet

        # Step 5: Generate binary signals
        binary_signals = self._generate_binary_signals(symbol, features)

        # Step 6: Calculate consensus
        consensus = self._calculate_consensus_score(binary_signals)

        # Step 7: Apply TIERED SIZING - calculate position multiplier
        if self.config.use_tiered_sizing:
            position_mult = self._calculate_position_multiplier(consensus)

            if position_mult == Decimal("0"):
                logger.debug(f"Skipping weak signal: {symbol} consensus={consensus:.3f}")
                return []
        else:
            # Legacy fixed threshold
            if abs(consensus) < self.config.consensus_threshold:
                return []
            position_mult = Decimal("1.0")

        # Step 8: Calculate base position size with regime adaptation
        base_exposure = self._calculate_qv_position_size(symbol, consensus, features)

        # Step 9: Apply tiered multiplier to exposure
        adjusted_exposure = base_exposure * position_mult

        # Step 9b: Apply uncertainty discount (Phase 2)
        if self.uncertainty_sizer is not None and self._last_uncertainty > Decimal("0"):
            # Use uncertainty sizer to discount position based on epistemic uncertainty
            confidence_factor = self.uncertainty_sizer.calculate_size_pct(
                signal_strength=float(abs(consensus)),
                uncertainty=float(self._last_uncertainty),
            )
            adjusted_exposure = adjusted_exposure * Decimal(str(confidence_factor))
            logger.debug(
                f"Uncertainty discount: {symbol} | uncertainty={self._last_uncertainty:.4f} | "
                f"confidence_factor={confidence_factor:.3f}"
            )

        # Step 10: Generate entry signals
        entry_signals = self._generate_qv_entry_signals(option_chain, cash, consensus, adjusted_exposure)

        # Track entry for holding period enforcement
        if entry_signals:
            self.entry_timestamps[symbol] = timestamp
            self.entry_consensus[symbol] = consensus

            logger.info(
                f"QV Entry: {symbol} | consensus={consensus:.3f} | "
                f"multiplier={position_mult:.2f} | exposure={adjusted_exposure:.2f}"
            )

        # Track consensus for monitoring
        self.last_consensus[symbol] = consensus

        logger.debug(
            f"QV Strategy: {symbol}",
            extra={
                "symbol": symbol,
                "consensus": float(consensus),
                "position_mult": float(position_mult),
                "base_exposure": float(base_exposure),
                "adjusted_exposure": float(adjusted_exposure),
                "epistemic_uncertainty": float(self._last_uncertainty),
                "signals": binary_signals,
            }
        )

        return entry_signals
