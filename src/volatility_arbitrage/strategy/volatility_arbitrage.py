"""
Volatility Arbitrage Strategy Implementation.

Implements a delta-neutral volatility arbitrage strategy that:
1. Forecasts realized volatility using GARCH
2. Compares to implied volatility from options
3. Trades the volatility spread
4. Maintains delta-neutral hedging
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

from volatility_arbitrage.core.types import OptionChain, OptionContract, Position, OptionType
from volatility_arbitrage.models.black_scholes import BlackScholesModel
from volatility_arbitrage.models.volatility import GARCHVolatility, calculate_returns
from volatility_arbitrage.strategy.base import Strategy, Signal
from volatility_arbitrage.utils.logging import get_logger

# Optional regime detection - gracefully handle if dependencies not available
try:
    from volatility_arbitrage.models.regime import RegimeDetector
except ImportError:
    RegimeDetector = None  # type: ignore

logger = get_logger(__name__)


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
        else:
            from volatility_arbitrage.models.volatility import HistoricalVolatility
            self.vol_forecaster = HistoricalVolatility(window=config.vol_lookback_period)

        # Track price history for volatility forecasting
        self.price_history: dict[str, pd.Series] = {}

        # Track option positions and their entry details
        self.option_positions: dict[str, dict] = {}  # symbol -> position details

        # Regime detection state
        self.current_regime: Optional[int] = None
        self.regime_history: list[tuple[datetime, int]] = []  # (timestamp, regime)
        self.returns_for_regime: dict[str, pd.Series] = {}  # symbol -> returns

        # Validate regime configuration
        if config.use_regime_detection:
            if regime_detector is None:
                logger.warning("use_regime_detection=True but no regime_detector provided")
                config.use_regime_detection = False
            elif config.regime_params is None or len(config.regime_params) == 0:
                logger.warning("use_regime_detection=True but no regime_params provided")
                config.use_regime_detection = False

        logger.info(
            "Initialized VolatilityArbitrageStrategy",
            extra=self.config.to_dict()
        )

    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: dict[str, Position],
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

            # Calculate volatility spread
            vol_spread = self._calculate_volatility_spread(
                symbol, timestamp, option_chain
            )

            if vol_spread is None:
                continue

            # Get regime-specific parameters
            entry_threshold, exit_threshold, position_multiplier = self._get_regime_parameters(
                self.current_regime
            )

            # Check for entry signals
            entry_signals = self._check_entry_signals(
                vol_spread, option_chain, positions, entry_threshold, position_multiplier
            )
            signals.extend(entry_signals)

            # Check for exit signals
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

        # Forecast realized volatility
        try:
            forecasted_vol = self.vol_forecaster.forecast(returns, horizon=1)
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
        vol_spread: VolatilitySpread,
        option_chain: OptionChain,
        positions: dict[str, Position],
        entry_threshold: Decimal,
        position_multiplier: Decimal,
    ) -> list[Signal]:
        """
        Check for new position entry signals.

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

        # Hedge to delta-neutral
        hedge_quantity = int(-net_delta * 100)  # Assuming 100 multiplier for options

        if hedge_quantity != 0:
            hedge_action = "buy" if hedge_quantity > 0 else "sell"
            signals.append(Signal(
                symbol=vol_spread.symbol,
                action=hedge_action,
                quantity=abs(hedge_quantity),
                reason=f"Delta hedge: target neutral, net delta {net_delta:.2f}"
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
