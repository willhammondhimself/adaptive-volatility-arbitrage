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

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from volatility_arbitrage.backtest.engine import BacktestEngine, BacktestResult
from volatility_arbitrage.core.config import BacktestConfig
from volatility_arbitrage.core.types import Trade, TradeType, OptionType
from volatility_arbitrage.models.black_scholes import BlackScholesModel, Greeks
from volatility_arbitrage.execution.costs import SquareRootImpactModel
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
    - Realistic margin requirements for short options
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy: Strategy,
        option_commission_per_contract: Decimal = Decimal("0.65"),
        option_slippage_pct: Decimal = Decimal("0.025"),  # 2.5% for options (realistic)
        margin_requirement_pct: Decimal = Decimal("0.25"),  # 25% margin for short options
    ) -> None:
        """
        Initialize multi-asset backtest engine.

        Args:
            config: Backtest configuration
            strategy: Trading strategy
            option_commission_per_contract: Commission per option contract
            option_slippage_pct: Slippage percentage for options (default 2.5%)
            margin_requirement_pct: Margin requirement for short options (default 25%)
        """
        super().__init__(config, strategy)

        self.option_commission_per_contract = option_commission_per_contract
        self.option_slippage_pct = option_slippage_pct
        self.margin_requirement_pct = margin_requirement_pct

        # Square-root impact model (Phase 2)
        if config.use_impact_model:
            self.cost_model: Optional[SquareRootImpactModel] = SquareRootImpactModel(
                half_spread_bps=float(config.impact_half_spread_bps),
                impact_coeff=float(config.impact_coefficient),
            )
            logger.info(
                "Using SquareRootImpactModel for transaction costs",
                extra={
                    "half_spread_bps": float(config.impact_half_spread_bps),
                    "impact_coeff": float(config.impact_coefficient),
                },
            )
        else:
            self.cost_model = None

        # Enhanced position tracking
        self.multi_positions: dict[str, MultiAssetPosition] = {}

        # Margin tracking for short options
        self.margin_used: Decimal = Decimal("0")  # Current margin in use
        self.margin_history: list[dict] = []  # Track margin over time

        # Greeks history
        self.greeks_history: list[dict] = []

        logger.info(
            "Initialized MultiAssetBacktestEngine",
            extra={
                "option_commission": float(option_commission_per_contract),
                "option_slippage": float(option_slippage_pct),
                "margin_requirement": float(margin_requirement_pct),
            },
        )

        self.history = []

    def _calculate_option_margin(
        self,
        underlying_price: Decimal,
        strike: Decimal,
        option_price: Decimal,
        quantity: int,
        is_short: bool,
    ) -> Decimal:
        """
        Calculate margin requirement for an option position.

        Uses a simplified Reg-T margin calculation:
        - Short options: max(25% of underlying + premium, 10% of strike + premium)
        - Long options: No margin required (paid in full)

        Args:
            underlying_price: Current price of underlying
            strike: Option strike price
            option_price: Current option price
            quantity: Number of contracts
            is_short: True if selling/shorting the option

        Returns:
            Margin requirement in dollars
        """
        if not is_short:
            # Long options have no margin requirement
            return Decimal("0")

        # Option multiplier
        multiplier = Decimal("100")
        abs_quantity = Decimal(abs(quantity))

        # Simplified Reg-T margin for short options:
        # 25% of underlying value + option premium (received)
        # Minimum: 10% of strike + premium
        underlying_component = underlying_price * self.margin_requirement_pct * multiplier * abs_quantity
        premium_value = option_price * multiplier * abs_quantity

        # Standard margin: 25% of underlying + premium
        standard_margin = underlying_component + premium_value

        # Minimum margin: 10% of strike + premium
        min_margin = strike * Decimal("0.10") * multiplier * abs_quantity + premium_value

        return max(standard_margin, min_margin)

    def _get_available_buying_power(self) -> Decimal:
        """
        Calculate available buying power considering margin requirements.

        Buying power = Cash - Margin Used

        Returns:
            Available buying power in dollars
        """
        return max(self.cash - self.margin_used, Decimal("0"))

    def _update_margin_from_position(
        self,
        symbol: str,
        underlying_price: Decimal,
        strike: Decimal,
        option_price: Decimal,
        quantity: int,
    ) -> None:
        """
        Update margin requirements after position change.

        Args:
            symbol: Option symbol
            underlying_price: Current underlying price
            strike: Option strike price
            option_price: Current option price
            quantity: New position quantity (negative = short)
        """
        if quantity < 0:
            # Short position - requires margin
            margin = self._calculate_option_margin(
                underlying_price=underlying_price,
                strike=strike,
                option_price=option_price,
                quantity=quantity,
                is_short=True,
            )
            # Store margin requirement for this position
            self._position_margins = getattr(self, '_position_margins', {})
            self._position_margins[symbol] = margin
        else:
            # Long position or closed - no margin
            self._position_margins = getattr(self, '_position_margins', {})
            self._position_margins.pop(symbol, None)

        # Recalculate total margin used
        self._position_margins = getattr(self, '_position_margins', {})
        self.margin_used = sum(self._position_margins.values(), Decimal("0"))

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
        Execute an option trade with option-specific costs and margin requirements.

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

        # Apply transaction costs
        if self.cost_model is not None:
            # Use square-root impact model (Phase 2)
            daily_volume = float(symbol_data.iloc[0].get("volume", 100_000))
            order_shares = abs(signal.quantity) * 100  # Options = 100 shares each

            impact_cost = self.cost_model.calculate_cost(
                order_size=order_shares,
                price=float(theoretical_price),
                volatility=float(implied_vol),
                daily_volume=daily_volume,
            )

            # Convert to percentage of trade value
            notional = float(theoretical_price) * order_shares
            impact_pct = Decimal(str(impact_cost / notional)) if notional > 0 else Decimal("0")

            if signal.action == "buy":
                execution_price = theoretical_price * (Decimal("1") + impact_pct)
            else:
                execution_price = theoretical_price * (Decimal("1") - impact_pct)
        else:
            # Original fixed slippage (2.5% default - realistic for options)
            if signal.action == "buy":
                execution_price = theoretical_price * (Decimal("1") + self.option_slippage_pct)
            else:
                execution_price = theoretical_price * (Decimal("1") - self.option_slippage_pct)

        # Calculate costs: per-contract commission
        commission = self.option_commission_per_contract * Decimal(abs(signal.quantity))

        # Options have 100 multiplier
        trade_value = execution_price * Decimal(abs(signal.quantity)) * Decimal("100")

        # Determine if this is opening a short position (selling to open)
        existing_position = self.multi_positions.get(signal.symbol)
        is_opening_short = signal.action == "sell" and (
            existing_position is None or existing_position.quantity >= 0
        )

        # Check buying power / margin requirements
        if signal.action == "buy":
            # Long options: need cash for premium + commission
            total_cost = trade_value + commission
            if total_cost > self.cash:
                logger.warning(
                    f"Insufficient cash for option buy",
                    extra={
                        "required": float(total_cost),
                        "available": float(self.cash),
                    },
                )
                return
        elif is_opening_short:
            # Short options: check margin requirement
            margin_required = self._calculate_option_margin(
                underlying_price=underlying_price,
                strike=strike,
                option_price=execution_price,
                quantity=signal.quantity,
                is_short=True,
            )
            available_buying_power = self._get_available_buying_power()

            if margin_required > available_buying_power:
                logger.warning(
                    f"Insufficient buying power for short option",
                    extra={
                        "margin_required": float(margin_required),
                        "available_buying_power": float(available_buying_power),
                        "current_margin_used": float(self.margin_used),
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
            # Selling options: receive premium minus commission
            self.cash += trade_value - commission

        # Update multi-asset positions
        self._update_multi_position_from_trade(
            trade, timestamp, underlying_price, strike, option_type, risk_free_rate, implied_vol
        )

        # Update margin tracking for short positions
        if signal.symbol in self.multi_positions:
            new_position = self.multi_positions[signal.symbol]
            self._update_margin_from_position(
                symbol=signal.symbol,
                underlying_price=underlying_price,
                strike=strike,
                option_price=execution_price,
                quantity=new_position.quantity,
            )
        else:
            # Position was closed
            self._update_margin_from_position(
                symbol=signal.symbol,
                underlying_price=underlying_price,
                strike=strike,
                option_price=execution_price,
                quantity=0,
            )

        logger.debug(
            f"Executed option trade: {signal.action} {signal.quantity} {signal.symbol}",
            extra={
                "symbol": signal.symbol,
                "action": signal.action,
                "quantity": signal.quantity,
                "price": float(execution_price),
                "commission": float(commission),
                "margin_used": float(self.margin_used),
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

        signals = self.strategy.generate_signals(
            timestamp=timestamp,
            market_data=day_data,
            positions=base_positions,
            cash=self.cash,
            portfolio_greeks=greeks,
        )

        # Execute trades based on signals
        for signal in signals:
            self._execute_signal(signal, timestamp, day_data)

        # Deduct daily carrying costs
        # 1. Margin financing on short option margin (daily rate = annual / 252)
        daily_margin_rate = self.config.margin_rate / Decimal("252") if hasattr(self.config, 'margin_rate') else Decimal("0.05") / Decimal("252")
        margin_financing_cost = self.margin_used * daily_margin_rate
        self.cash -= margin_financing_cost

        # 2. Daily delta hedge rebalancing cost (applied to notional exposure)
        daily_hedge_cost_rate = self.config.daily_hedge_cost if hasattr(self.config, 'daily_hedge_cost') else Decimal("0.0002")
        portfolio_delta = self._calculate_portfolio_greeks().delta
        # Hedge cost proportional to delta exposure (approximate notional)
        hedge_notional = abs(portfolio_delta) * Decimal("100")  # Delta * 100 shares per contract
        daily_hedge_cost = hedge_notional * daily_hedge_cost_rate
        self.cash -= daily_hedge_cost

        # Record equity and margin (after costs)
        equity = self._calculate_multi_equity(day_data)
        buying_power = self._get_available_buying_power()

        self.equity_history.append(
            {
                "timestamp": timestamp,
                "cash": float(self.cash),
                "positions_value": float(equity - self.cash),
                "total_equity": float(equity),
                "portfolio_delta": float(greeks.delta),
                "portfolio_vega": float(greeks.vega),
                "margin_used": float(self.margin_used),
                "buying_power": float(buying_power),
            }
        )
        self.margin_history.append({
            "timestamp": timestamp,
            "margin_used": float(self.margin_used),
            "buying_power": float(buying_power),
            "margin_utilization": float(self.margin_used / self.cash) if self.cash > 0 else 0.0,
        })
        self.history.append({
            "timestamp": timestamp,
            "equity": float(equity),
            "cash": float(self.cash),
            "margin_used": float(self.margin_used),
        })


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

    def get_results(self) -> dict:
        """
        Calculates and returns the backtest results.
        """
        # SAFETY CHECK: If history doesn't exist, return empty results
        if not hasattr(self, 'history') or not self.history:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "equity_curve": [],
            }

        equity_curve = pd.DataFrame(self.history).set_index("timestamp")["equity"]
        
        # Total Return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Sharpe Ratio
        daily_returns = equity_curve.pct_change().dropna()
        sharpe_ratio = 0.0
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            
        # Max Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "total_trades": len(self.trades),
            "equity_curve": self.history,
        }
