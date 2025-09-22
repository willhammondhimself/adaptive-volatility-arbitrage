"""
Multi-asset backtesting engine for options and underlying stocks.

Extends the base BacktestEngine to support simultaneous trading of:
- Options (calls and puts)
- Underlying stocks (for delta hedging)

Tracks portfolio-level Greeks and handles option-specific execution details.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Literal

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from volatility_arbitrage.backtest.engine import BacktestEngine, BacktestResult
from volatility_arbitrage.core.config import BacktestConfig
from volatility_arbitrage.core.types import Trade, TradeType, OptionType
from volatility_arbitrage.models.black_scholes import BlackScholesModel, Greeks
from volatility_arbitrage.strategy.base import Strategy, Signal
from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PortfolioGreeks:
    """
    Aggregated portfolio-level Greeks.

    Represents total risk exposure across all option positions.
    """

    delta: Decimal
    gamma: Decimal
    vega: Decimal
    theta: Decimal
    rho: Decimal

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "delta": float(self.delta),
            "gamma": float(self.gamma),
            "vega": float(self.vega),
            "theta": float(self.theta),
            "rho": float(self.rho),
        }


class MultiAssetPosition(BaseModel):
    """
    Position in either stock or option.

    Tracks all necessary information for P&L and risk calculations.
    """

    model_config = ConfigDict(frozen=False)

    symbol: str
    asset_type: Literal["stock", "option"]
    quantity: int
    entry_price: Decimal
    current_price: Decimal
    last_update: datetime

    # Option-specific fields
    option_type: Optional[OptionType] = None
    strike: Optional[Decimal] = None
    expiry: Optional[datetime] = None
    underlying_price: Optional[Decimal] = None
    implied_volatility: Optional[Decimal] = None
    risk_free_rate: Optional[Decimal] = None

    # Greeks (calculated, not stored)
    _greeks: Optional[Greeks] = None

    @property
    def is_option(self) -> bool:
        """Check if this is an option position."""
        return self.asset_type == "option"

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value."""
        if self.is_option:
            # Options have 100 multiplier
            return Decimal(self.quantity) * self.current_price * Decimal("100")
        else:
            return Decimal(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> Decimal:
        """Calculate original cost basis."""
        if self.is_option:
            return Decimal(abs(self.quantity)) * self.entry_price * Decimal("100")
        else:
            return Decimal(abs(self.quantity)) * self.entry_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        if self.quantity > 0:  # Long
            return (self.current_price - self.entry_price) * Decimal(abs(self.quantity)) * (
                Decimal("100") if self.is_option else Decimal("1")
            )
        else:  # Short
            return (self.entry_price - self.current_price) * Decimal(abs(self.quantity)) * (
                Decimal("100") if self.is_option else Decimal("1")
            )

    def calculate_greeks(self) -> Optional[Greeks]:
        """
        Calculate Greeks for option position.

        Returns:
            Greeks object or None if not an option or missing data
        """
        if not self.is_option:
            return None

        if (
            self.underlying_price is None
            or self.strike is None
            or self.expiry is None
            or self.implied_volatility is None
            or self.risk_free_rate is None
        ):
            logger.warning(f"Missing data for Greeks calculation: {self.symbol}")
            return None

        # Calculate time to expiry
        time_to_expiry = (self.expiry - self.last_update).days / Decimal("365")
        if time_to_expiry <= 0:
            return None

        try:
            greeks = BlackScholesModel.greeks(
                S=self.underlying_price,
                K=self.strike,
                T=time_to_expiry,
                r=self.risk_free_rate,
                sigma=self.implied_volatility,
                option_type=self.option_type,
            )

            # Scale by position size (positive for long, negative for short)
            multiplier = Decimal(self.quantity)

            scaled_greeks = Greeks(
                delta=greeks.delta * multiplier,
                gamma=greeks.gamma * multiplier,
                vega=greeks.vega * multiplier,
                theta=greeks.theta * multiplier,
                rho=greeks.rho * multiplier,
            )

            self._greeks = scaled_greeks
            return scaled_greeks

        except Exception as e:
            logger.warning(f"Greeks calculation failed for {self.symbol}: {e}")
            return None

    def update_price(
        self,
        new_price: Decimal,
        timestamp: datetime,
        underlying_price: Optional[Decimal] = None,
        implied_vol: Optional[Decimal] = None,
    ) -> None:
        """Update position with new market data."""
        self.current_price = new_price
        self.last_update = timestamp

        if self.is_option:
            if underlying_price is not None:
                self.underlying_price = underlying_price
            if implied_vol is not None:
                self.implied_volatility = implied_vol

            # Recalculate Greeks
            self.calculate_greeks()


class MultiAssetBacktestEngine(BacktestEngine):
    """
    Enhanced backtest engine for multi-asset portfolios.

    Supports:
    - Stocks and options in same portfolio
    - Portfolio-level Greeks tracking
    - Options-specific execution model
    - Option expiration handling
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy: Strategy,
        option_commission_per_contract: Decimal = Decimal("0.65"),
        option_slippage_pct: Decimal = Decimal("0.01"),  # 1% for options
    ) -> None:
        """
        Initialize multi-asset backtest engine.

        Args:
            config: Backtest configuration
            strategy: Trading strategy
            option_commission_per_contract: Commission per option contract
            option_slippage_pct: Slippage percentage for options
        """
        super().__init__(config, strategy)

        self.option_commission_per_contract = option_commission_per_contract
        self.option_slippage_pct = option_slippage_pct

        # Enhanced position tracking
        self.multi_positions: dict[str, MultiAssetPosition] = {}

        # Greeks history
        self.greeks_history: list[dict] = []

        logger.info(
            "Initialized MultiAssetBacktestEngine",
            extra={
                "option_commission": float(option_commission_per_contract),
                "option_slippage": float(option_slippage_pct),
            },
        )

    def _execute_signal(
        self,
        signal: Signal,
        timestamp: datetime,
        day_data: pd.DataFrame,
    ) -> None:
        """
        Execute a trading signal (stock or option).

        Overrides base implementation to handle options differently.

        Args:
            signal: Trading signal
            timestamp: Current timestamp
            day_data: Market data for this day
        """
        # Determine if this is an option signal (check symbol format)
        is_option = "_CALL_" in signal.symbol or "_PUT_" in signal.symbol

        if is_option:
            self._execute_option_signal(signal, timestamp, day_data)
        else:
            # Use base implementation for stocks
            super()._execute_signal(signal, timestamp, day_data)

    def _execute_option_signal(
        self,
        signal: Signal,
        timestamp: datetime,
        day_data: pd.DataFrame,
    ) -> None:
        """
        Execute an option trade with option-specific costs.

        Args:
            signal: Option trading signal
            timestamp: Current timestamp
            day_data: Market data
        """
        # Parse option symbol: SYMBOL_TYPE_STRIKE_EXPIRY
        parts = signal.symbol.split("_")
        if len(parts) < 4:
            logger.warning(f"Invalid option symbol format: {signal.symbol}")
            return

        underlying_symbol = parts[0]
        option_type_str = parts[1]
        strike = Decimal(parts[2])
        expiry_str = parts[3]

        # Get underlying price
        symbol_data = day_data[day_data["symbol"] == underlying_symbol]
        if symbol_data.empty:
            logger.warning(f"No price data for underlying {underlying_symbol}")
            return

        underlying_price = Decimal(str(symbol_data.iloc[0]["close"]))

        # For now, use a simplified pricing model
        # In production, would use actual option chain data
        # Estimate option price using Black-Scholes
        time_to_expiry = Decimal("0.08")  # ~1 month placeholder
        risk_free_rate = self.config.risk_free_rate
        implied_vol = Decimal("0.25")  # Placeholder

        option_type = OptionType.CALL if option_type_str == "CALL" else OptionType.PUT

        theoretical_price = BlackScholesModel.price(
            S=underlying_price,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=implied_vol,
            option_type=option_type,
        )

        # Apply slippage
        if signal.action == "buy":
            execution_price = theoretical_price * (Decimal("1") + self.option_slippage_pct)
        else:
            execution_price = theoretical_price * (Decimal("1") - self.option_slippage_pct)

        # Calculate costs: per-contract commission
        commission = self.option_commission_per_contract * Decimal(abs(signal.quantity))

        # Options have 100 multiplier
        trade_value = execution_price * Decimal(abs(signal.quantity)) * Decimal("100")

        # Check cash for buys
        if signal.action == "buy":
            total_cost = trade_value + commission
            if total_cost > self.cash:
                logger.warning(
                    f"Insufficient cash for option trade",
                    extra={
                        "required": float(total_cost),
                        "available": float(self.cash),
                    },
                )
                return

        # Execute trade
        trade = Trade(
            timestamp=timestamp,
            symbol=signal.symbol,
            trade_type=TradeType.BUY if signal.action == "buy" else TradeType.SELL,
            quantity=abs(signal.quantity),
            price=execution_price,
            commission=commission,
        )

        self.trades.append(trade)

        # Update cash
        if signal.action == "buy":
            self.cash -= trade_value + commission
        else:
            self.cash += trade_value - commission

        # Update multi-asset positions
        self._update_multi_position_from_trade(
            trade, timestamp, underlying_price, strike, option_type, risk_free_rate, implied_vol
        )

        logger.debug(
            f"Executed option trade: {signal.action} {signal.quantity} {signal.symbol}",
            extra={
                "symbol": signal.symbol,
                "action": signal.action,
                "quantity": signal.quantity,
                "price": float(execution_price),
                "commission": float(commission),
            },
        )

    def _update_multi_position_from_trade(
        self,
        trade: Trade,
        timestamp: datetime,
        underlying_price: Decimal,
        strike: Decimal,
        option_type: OptionType,
        risk_free_rate: Decimal,
        implied_vol: Decimal,
    ) -> None:
        """Update multi-asset position tracking."""
        symbol = trade.symbol
        is_option = "_CALL_" in symbol or "_PUT_" in symbol

        # Determine expiry (placeholder - should come from actual data)
        expiry = timestamp + timedelta(days=30)

        if symbol not in self.multi_positions:
            # New position
            if trade.trade_type == TradeType.BUY:
                quantity = trade.quantity
            else:
                quantity = -trade.quantity

            self.multi_positions[symbol] = MultiAssetPosition(
                symbol=symbol,
                asset_type="option" if is_option else "stock",
                quantity=quantity,
                entry_price=trade.price,
                current_price=trade.price,
                last_update=timestamp,
                option_type=option_type if is_option else None,
                strike=strike if is_option else None,
                expiry=expiry if is_option else None,
                underlying_price=underlying_price if is_option else None,
                implied_volatility=implied_vol if is_option else None,
                risk_free_rate=risk_free_rate if is_option else None,
            )
        else:
            # Update existing position
            pos = self.multi_positions[symbol]

            if trade.trade_type == TradeType.BUY:
                new_quantity = pos.quantity + trade.quantity
            else:
                new_quantity = pos.quantity - trade.quantity

            if new_quantity == 0:
                # Position closed
                del self.multi_positions[symbol]
            else:
                # Update position
                # Recalculate average entry price if increasing position
                if (pos.quantity > 0 and trade.trade_type == TradeType.BUY) or (
                    pos.quantity < 0 and trade.trade_type == TradeType.SELL
                ):
                    old_value = pos.entry_price * Decimal(abs(pos.quantity))
                    new_value = trade.price * Decimal(trade.quantity)
                    total_quantity = abs(new_quantity)
                    pos.entry_price = (old_value + new_value) / Decimal(total_quantity)

                pos.quantity = new_quantity
                pos.current_price = trade.price
                pos.last_update = timestamp

    def _calculate_portfolio_greeks(self) -> PortfolioGreeks:
        """
        Calculate aggregate portfolio Greeks.

        Returns:
            PortfolioGreeks with sum of all option Greeks
        """
        total_delta = Decimal("0")
        total_gamma = Decimal("0")
        total_vega = Decimal("0")
        total_theta = Decimal("0")
        total_rho = Decimal("0")

        for pos in self.multi_positions.values():
            if pos.is_option:
                greeks = pos.calculate_greeks()
                if greeks is not None:
                    total_delta += greeks.delta
                    total_gamma += greeks.gamma
                    total_vega += greeks.vega
                    total_theta += greeks.theta
                    total_rho += greeks.rho

        return PortfolioGreeks(
            delta=total_delta,
            gamma=total_gamma,
            vega=total_vega,
            theta=total_theta,
            rho=total_rho,
        )

    def _handle_expirations(self, timestamp: datetime) -> None:
        """
        Check for and handle option expirations.

        Closes positions 1 day before expiry.

        Args:
            timestamp: Current date
        """
        expiring_positions = []

        for symbol, pos in self.multi_positions.items():
            if pos.is_option and pos.expiry is not None:
                # Close 1 day before expiry
                days_to_expiry = (pos.expiry - timestamp).days
                if days_to_expiry <= 1:
                    expiring_positions.append(symbol)

        for symbol in expiring_positions:
            pos = self.multi_positions[symbol]

            # Create closing trade
            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                trade_type=TradeType.SELL if pos.quantity > 0 else TradeType.BUY,
                quantity=abs(pos.quantity),
                price=pos.current_price,
                commission=self.option_commission_per_contract * Decimal(abs(pos.quantity)),
            )

            self.trades.append(trade)

            # Update cash
            if pos.quantity > 0:  # Long position
                self.cash += pos.current_price * Decimal(abs(pos.quantity)) * Decimal("100")
                self.cash -= trade.commission
            else:  # Short position
                self.cash -= pos.current_price * Decimal(abs(pos.quantity)) * Decimal("100")
                self.cash -= trade.commission

            # Remove position
            del self.multi_positions[symbol]

            logger.info(
                f"Closed expiring option position: {symbol}",
                extra={"symbol": symbol, "expiry": pos.expiry},
            )

    def _process_day(self, timestamp: datetime, day_data: pd.DataFrame) -> None:
        """
        Process a single trading day with multi-asset support.

        Overrides base implementation to add:
        - Multi-asset position updates
        - Greeks tracking
        - Option expiration handling

        Args:
            timestamp: Current timestamp
            day_data: Market data for this day
        """
        # Handle option expirations first
        self._handle_expirations(timestamp)

        # Update positions (use multi_positions instead of base positions)
        self._update_multi_positions(timestamp, day_data)

        # Calculate and record portfolio Greeks
        greeks = self._calculate_portfolio_greeks()
        self.greeks_history.append(
            {
                "timestamp": timestamp,
                **greeks.to_dict(),
            }
        )

        # Generate signals from strategy
        # Convert multi_positions to base Position format for strategy
        base_positions = {}
        for symbol, multi_pos in self.multi_positions.items():
            from volatility_arbitrage.core.types import Position

            base_positions[symbol] = Position(
                symbol=symbol,
                quantity=multi_pos.quantity,
                avg_entry_price=multi_pos.entry_price,
                current_price=multi_pos.current_price,
                last_update=multi_pos.last_update,
            )

        signals = self.strategy.generate_signals(timestamp, day_data, base_positions)

        # Execute trades based on signals
        for signal in signals:
            self._execute_signal(signal, timestamp, day_data)

        # Record equity
        equity = self._calculate_multi_equity(day_data)
        self.equity_history.append(
            {
                "timestamp": timestamp,
                "cash": float(self.cash),
                "positions_value": float(equity - self.cash),
                "total_equity": float(equity),
                "portfolio_delta": float(greeks.delta),
                "portfolio_vega": float(greeks.vega),
            }
        )

    def _update_multi_positions(self, timestamp: datetime, day_data: pd.DataFrame) -> None:
        """Update all multi-asset positions with current prices."""
        for symbol, pos in list(self.multi_positions.items()):
            if pos.is_option:
                # Extract underlying symbol
                underlying_symbol = symbol.split("_")[0]
                symbol_data = day_data[day_data["symbol"] == underlying_symbol]

                if not symbol_data.empty:
                    underlying_price = Decimal(str(symbol_data.iloc[0]["close"]))

                    # Recalculate option price (simplified - should use market data)
                    if pos.expiry and pos.strike and pos.implied_volatility:
                        time_to_expiry = max(
                            Decimal((pos.expiry - timestamp).days) / Decimal("365"), Decimal("0.001")
                        )

                        new_price = BlackScholesModel.price(
                            S=underlying_price,
                            K=pos.strike,
                            T=time_to_expiry,
                            r=pos.risk_free_rate or self.config.risk_free_rate,
                            sigma=pos.implied_volatility,
                            option_type=pos.option_type,
                        )

                        pos.update_price(new_price, timestamp, underlying_price)
            else:
                # Stock position
                symbol_data = day_data[day_data["symbol"] == symbol]
                if not symbol_data.empty:
                    current_price = Decimal(str(symbol_data.iloc[0]["close"]))
                    pos.update_price(current_price, timestamp)

    def _calculate_multi_equity(self, day_data: pd.DataFrame) -> Decimal:
        """Calculate total equity across all assets."""
        positions_value = Decimal("0")

        for pos in self.multi_positions.values():
            positions_value += pos.market_value

        return self.cash + positions_value
