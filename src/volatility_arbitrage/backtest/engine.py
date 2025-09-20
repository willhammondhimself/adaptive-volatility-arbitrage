"""
Backtesting engine for volatility arbitrage strategies.

Event-driven backtesting with daily loop, position tracking, and P&L calculation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

import pandas as pd

from volatility_arbitrage.core.config import BacktestConfig
from volatility_arbitrage.core.types import Trade, Position, TradeType
from volatility_arbitrage.strategy.base import Strategy, Signal
from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    Contains performance metrics and trade history.
    """

    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_return: Decimal
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions_history: list[dict] = field(default_factory=list)

    @property
    def total_return_pct(self) -> Decimal:
        """Calculate total return as percentage."""
        if self.initial_capital == 0:
            return Decimal("0")
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * Decimal(
            "100"
        )

    @property
    def num_trades(self) -> int:
        """Total number of trades executed."""
        return len(self.trades)

    @property
    def num_winning_trades(self) -> int:
        """Number of profitable round-trip trades."""
        # This is simplified - proper calculation requires matching buys/sells
        return 0  # TODO: Implement round-trip trade matching

    @property
    def num_losing_trades(self) -> int:
        """Number of losing round-trip trades."""
        return 0  # TODO: Implement round-trip trade matching


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates trading strategy execution on historical data with realistic
    transaction costs and position tracking.
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy: Strategy,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            strategy: Trading strategy to test
        """
        self.config = config
        self.strategy = strategy

        # Account state
        self.cash = config.initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_history: list[dict] = []

        logger.info(
            "Initialized BacktestEngine",
            extra={
                "initial_capital": float(config.initial_capital),
                "commission_rate": float(config.commission_rate),
                "max_positions": config.max_positions,
            },
        )

    def run(
        self,
        market_data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            market_data: DataFrame with columns: timestamp, symbol, close, volume
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResult with performance metrics

        Raises:
            ValueError: If market_data is invalid
        """
        if market_data.empty:
            raise ValueError("Market data is empty")

        # Filter by date range
        if start_date is not None:
            market_data = market_data[market_data["timestamp"] >= start_date]
        if end_date is not None:
            market_data = market_data[market_data["timestamp"] <= end_date]

        if market_data.empty:
            raise ValueError("No data in specified date range")

        logger.info(
            f"Starting backtest",
            extra={
                "start": market_data["timestamp"].min(),
                "end": market_data["timestamp"].max(),
                "rows": len(market_data),
            },
        )

        # Reset state
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []

        # Group by timestamp for daily processing
        grouped = market_data.groupby("timestamp")

        # Daily loop
        for timestamp, day_data in grouped:
            self._process_day(timestamp, day_data)

        # Close all positions at end
        final_timestamp = market_data["timestamp"].max()
        self._close_all_positions(final_timestamp, market_data)

        # Create result
        result = self._create_result(market_data)

        logger.info(
            "Backtest completed",
            extra={
                "final_capital": float(result.final_capital),
                "total_return": float(result.total_return_pct),
                "num_trades": result.num_trades,
            },
        )

        return result

    def _process_day(self, timestamp: datetime, day_data: pd.DataFrame) -> None:
        """
        Process a single trading day.

        Args:
            timestamp: Current timestamp
            day_data: Market data for this day
        """
        # Update positions with current prices
        self._update_positions(timestamp, day_data)

        # Generate signals from strategy
        signals = self.strategy.generate_signals(timestamp, day_data, self.positions)

        # Execute trades based on signals
        for signal in signals:
            self._execute_signal(signal, timestamp, day_data)

        # Record equity
        equity = self._calculate_equity(day_data)
        self.equity_history.append(
            {
                "timestamp": timestamp,
                "cash": float(self.cash),
                "positions_value": float(equity - self.cash),
                "total_equity": float(equity),
            }
        )

    def _update_positions(self, timestamp: datetime, day_data: pd.DataFrame) -> None:
        """
        Update position prices with current market data.

        Args:
            timestamp: Current timestamp
            day_data: Market data for this day
        """
        for symbol in list(self.positions.keys()):
            symbol_data = day_data[day_data["symbol"] == symbol]

            if not symbol_data.empty:
                current_price = Decimal(str(symbol_data.iloc[0]["close"]))
                self.positions[symbol].update_price(current_price, timestamp)
            else:
                logger.warning(
                    f"No price data for {symbol} on {timestamp}",
                    extra={"symbol": symbol, "timestamp": timestamp},
                )

    def _execute_signal(
        self,
        signal: Signal,
        timestamp: datetime,
        day_data: pd.DataFrame,
    ) -> None:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal to execute
            timestamp: Current timestamp
            day_data: Market data for this day
        """
        # Get current price
        symbol_data = day_data[day_data["symbol"] == signal.symbol]
        if symbol_data.empty:
            logger.warning(
                f"Cannot execute signal: no price for {signal.symbol}",
                extra={"symbol": signal.symbol},
            )
            return

        current_price = Decimal(str(symbol_data.iloc[0]["close"]))

        # Apply slippage
        if signal.action == "buy":
            execution_price = current_price * (Decimal("1") + self.config.slippage)
        else:  # sell
            execution_price = current_price * (Decimal("1") - self.config.slippage)

        # Calculate trade size
        trade_value = execution_price * Decimal(abs(signal.quantity))
        commission = trade_value * self.config.commission_rate

        # Check cash availability for buys
        if signal.action == "buy":
            total_cost = trade_value + commission
            if total_cost > self.cash:
                logger.warning(
                    f"Insufficient cash for trade",
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
            self.cash -= trade.total_cost
        else:  # sell
            self.cash += trade.total_cost

        # Update positions
        self._update_position_from_trade(trade, timestamp)

        logger.debug(
            f"Executed trade: {signal.action} {signal.quantity} {signal.symbol} @ {execution_price}",
            extra={
                "symbol": signal.symbol,
                "action": signal.action,
                "quantity": signal.quantity,
                "price": float(execution_price),
                "commission": float(commission),
            },
        )

    def _update_position_from_trade(self, trade: Trade, timestamp: datetime) -> None:
        """
        Update position tracking after a trade.

        Args:
            trade: Executed trade
            timestamp: Current timestamp
        """
        symbol = trade.symbol

        if symbol not in self.positions:
            # New position
            if trade.trade_type == TradeType.BUY:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    avg_entry_price=trade.price,
                    current_price=trade.price,
                    last_update=timestamp,
                )
        else:
            # Update existing position
            pos = self.positions[symbol]

            if trade.trade_type == TradeType.BUY:
                # Add to position
                new_quantity = pos.quantity + trade.quantity
                new_avg_price = (
                    pos.avg_entry_price * Decimal(pos.quantity)
                    + trade.price * Decimal(trade.quantity)
                ) / Decimal(new_quantity)

                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=new_quantity,
                    avg_entry_price=new_avg_price,
                    current_price=trade.price,
                    last_update=timestamp,
                )
            else:  # SELL
                # Reduce or close position
                new_quantity = pos.quantity - trade.quantity

                if new_quantity <= 0:
                    # Close position
                    del self.positions[symbol]
                else:
                    # Reduce position (keep same avg entry price)
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=new_quantity,
                        avg_entry_price=pos.avg_entry_price,
                        current_price=trade.price,
                        last_update=timestamp,
                    )

    def _calculate_equity(self, day_data: pd.DataFrame) -> Decimal:
        """
        Calculate total equity (cash + position values).

        Args:
            day_data: Current market data

        Returns:
            Total equity value
        """
        positions_value = Decimal("0")

        for symbol, pos in self.positions.items():
            positions_value += pos.market_value

        return self.cash + positions_value

    def _close_all_positions(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
    ) -> None:
        """
        Close all open positions at end of backtest.

        Args:
            timestamp: Final timestamp
            market_data: Market data
        """
        final_data = market_data[market_data["timestamp"] == timestamp]

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]

            symbol_data = final_data[final_data["symbol"] == symbol]
            if symbol_data.empty:
                logger.warning(f"No final price for {symbol}, using last known price")
                final_price = pos.current_price
            else:
                final_price = Decimal(str(symbol_data.iloc[0]["close"]))

            # Create closing trade
            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                trade_type=TradeType.SELL,
                quantity=pos.quantity,
                price=final_price,
                commission=final_price * Decimal(pos.quantity) * self.config.commission_rate,
            )

            self.trades.append(trade)
            self.cash += trade.total_cost

        # Clear positions
        self.positions = {}

    def _create_result(self, market_data: pd.DataFrame) -> BacktestResult:
        """
        Create BacktestResult from engine state.

        Args:
            market_data: Historical market data

        Returns:
            BacktestResult instance
        """
        equity_df = pd.DataFrame(self.equity_history)

        return BacktestResult(
            start_date=market_data["timestamp"].min(),
            end_date=market_data["timestamp"].max(),
            initial_capital=self.config.initial_capital,
            final_capital=self.cash,
            total_return=self.cash - self.config.initial_capital,
            trades=self.trades,
            equity_curve=equity_df,
        )
