"""
Paper Trading System.

Async trading loop that uses Bayesian LSTM forecaster to make trading decisions
based on epistemic uncertainty. Auto-trains on startup with 1 year of SPY returns.

Features:
- 30-second tick interval
- Uncertainty-based trade filtering
- SQLite persistence
- Graceful shutdown
- Colorama terminal output
"""

import asyncio
import math
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from colorama import Fore, Style, init as colorama_init

# Add src path for models
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "live_trading"))

from backend.services.paper_trading_db import PaperTradingDB
from backend.services.live_market_service import LiveMarketService
from mock_execution_gateway import MockExchangeGateway, Order, HeartbeatManager

# Lazy import to avoid circular imports
_forecaster_class = None


def _get_forecaster_class():
    global _forecaster_class
    if _forecaster_class is None:
        from volatility_arbitrage.models.bayesian_forecaster import BayesianLSTMForecaster
        _forecaster_class = BayesianLSTMForecaster
    return _forecaster_class


colorama_init(autoreset=True)


class PaperTradingSystem:
    """
    Singleton async trading system with 30s loop.

    Uses Bayesian LSTM for volatility forecasting and only trades
    when epistemic uncertainty is below threshold.
    """

    _instance: Optional["PaperTradingSystem"] = None

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        uncertainty_threshold: float = 0.02,
        position_pct: float = 0.10,
        loop_interval: float = 30.0,
        db_path: Optional[str] = None,
    ):
        self.initial_capital = initial_capital
        self.uncertainty_threshold = uncertainty_threshold
        self.position_pct = position_pct
        self.loop_interval = loop_interval

        # Dependencies
        self.db = PaperTradingDB(db_path)
        self.market_service = LiveMarketService()
        self.gateway = MockExchangeGateway(disconnect_prob=0.0)
        self.heartbeat_mgr: Optional[HeartbeatManager] = None

        # Forecaster (lazy init - requires training)
        self.forecaster = None
        self._forecaster_trained = False

        # State
        self.session_id: Optional[int] = None
        self.capital = initial_capital
        self.position = 0
        self.avg_cost = 0.0
        self.cumulative_pnl = 0.0
        self.total_trades = 0
        self.total_skipped = 0
        self.last_update: Optional[str] = None
        self.tick_count = 0

        # Recent returns cache for forecasting
        self._returns_cache: Optional[pd.Series] = None
        self._returns_updated: Optional[datetime] = None

        # Async
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "PaperTradingSystem":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    async def start(self) -> int:
        """
        Start trading loop.

        - Trains forecaster on 1 year of SPY data
        - Connects to mock exchange
        - Starts heartbeat manager
        - Begins trading loop

        Returns:
            session_id
        """
        async with self._lock:
            if self._running:
                raise RuntimeError("Already running")

            self._print_header()
            print(f"{Fore.YELLOW}Initializing...{Style.RESET_ALL}")

            # Train forecaster if not done
            if not self._forecaster_trained:
                await self._train_forecaster()

            # Connect to exchange
            await self.gateway.connect()
            self.heartbeat_mgr = HeartbeatManager(self.gateway, interval_s=15.0)
            await self.heartbeat_mgr.start()

            # Resume or create session
            self.session_id = self.db.resume_or_create_session(self.capital)
            print(f"{Fore.GREEN}Session {self.session_id} started{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Capital: ${self.capital:,.2f} | Threshold: {self.uncertainty_threshold}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Position Size: {self.position_pct * 100:.0f}% | Loop: {self.loop_interval}s{Style.RESET_ALL}")
            print("-" * 60)

            self._running = True
            self._task = asyncio.create_task(self._trading_loop())
            return self.session_id

    async def stop(self) -> dict:
        """
        Graceful shutdown.

        Returns:
            Final session statistics
        """
        async with self._lock:
            self._running = False

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=60.0)
            except asyncio.TimeoutError:
                self._task.cancel()

        if self.heartbeat_mgr:
            await self.heartbeat_mgr.stop()

        if self.gateway.connected:
            await self.gateway.disconnect()

        # Close any open position at current price
        if self.position > 0:
            try:
                quote = self.market_service.get_quote("SPY")
                await self._execute_sell(quote.price, 0.0, 0.0, force=True)
            except Exception:
                pass

        # Record final session stats
        if self.session_id:
            self.db.end_session(
                self.session_id,
                self.capital,
                self.total_trades,
                self.total_skipped,
            )

        stats = self.get_stats()
        self._print_summary(stats)
        return stats

    def get_stats(self) -> dict:
        """Get current session statistics."""
        if self.session_id:
            return self.db.get_stats(self.session_id)
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_estimate": 0.0,
            "skipped_ticks": 0,
        }

    async def _train_forecaster(self) -> None:
        """
        Train Bayesian LSTM on 1 year of SPY returns.

        Takes ~5 seconds.
        """
        print(f"{Fore.YELLOW}Fetching 1 year of SPY data...{Style.RESET_ALL}")

        try:
            # Fetch 1 year of SPY data
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1y")

            if len(hist) < 50:
                raise ValueError("Insufficient historical data")

            # Calculate returns
            returns = hist["Close"].pct_change().dropna()

            # Calculate realized volatility (20-day rolling)
            realized_vol = returns.rolling(20).std() * math.sqrt(252)
            realized_vol = realized_vol.dropna()

            # Align series
            common_idx = returns.index.intersection(realized_vol.index)
            returns = returns.loc[common_idx]
            realized_vol = realized_vol.loc[common_idx]

            print(f"{Fore.YELLOW}Training Bayesian LSTM ({len(returns)} samples)...{Style.RESET_ALL}")

            # Create and train forecaster
            ForecasterClass = _get_forecaster_class()
            self.forecaster = ForecasterClass(
                sequence_length=20,
                n_mc_samples=20,
                hidden_size=64,
            )

            losses = self.forecaster.fit(returns, realized_vol, epochs=10, lr=0.01)

            if losses:
                print(f"{Fore.GREEN}Training complete. Final loss: {losses[-1]:.6f}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Training skipped (fallback mode){Style.RESET_ALL}")

            # Cache returns for forecasting
            self._returns_cache = returns
            self._returns_updated = datetime.now()
            self._forecaster_trained = True

        except Exception as e:
            print(f"{Fore.RED}Training failed: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Using fallback forecaster{Style.RESET_ALL}")
            ForecasterClass = _get_forecaster_class()
            self.forecaster = ForecasterClass(sequence_length=20, n_mc_samples=20)
            self._forecaster_trained = True

    async def _trading_loop(self) -> None:
        """Main 30s loop with error isolation."""
        while self._running:
            loop = asyncio.get_event_loop()
            start = loop.time()

            try:
                await self._tick()
            except Exception as e:
                self._print_error(f"Tick error: {e}")

            elapsed = loop.time() - start
            sleep_time = max(0, self.loop_interval - elapsed)

            if self._running:
                await asyncio.sleep(sleep_time)

    async def _tick(self) -> None:
        """Single tick: fetch -> forecast -> decide -> execute."""
        self.tick_count += 1
        self.last_update = datetime.now().isoformat()

        # 1. Fetch SPY price
        quote = self.market_service.get_quote("SPY")
        price = quote.price

        # 2. Update returns cache and get forecast
        returns = self._get_recent_returns()
        forecast = self.forecaster.forecast_with_uncertainty(returns)
        uncertainty = float(forecast["epistemic_uncertainty"])
        forecast_vol = float(forecast["mean_vol"])

        # 3. Decision logic
        if uncertainty >= self.uncertainty_threshold:
            self._print_skip(price, uncertainty, forecast_vol)
            self.total_skipped += 1
            self.db.increment_skipped(self.session_id)
            return

        if self.position == 0:
            # Entry: buy when uncertainty is low
            qty = int((self.capital * self.position_pct) / price)
            if qty > 0:
                await self._execute_buy(qty, price, forecast_vol, uncertainty)
        elif forecast_vol > 0.30:
            # Exit: sell when vol forecast > 30%
            await self._execute_sell(price, forecast_vol, uncertainty)
        else:
            # Hold
            self._print_hold(price, uncertainty, forecast_vol)

        # Print summary every 10 ticks
        if self.tick_count % 10 == 0:
            self._print_periodic_summary()

    def _get_recent_returns(self) -> pd.Series:
        """Get recent returns for forecasting, updating cache if stale."""
        now = datetime.now()

        # Refresh cache if older than 5 minutes or missing
        if (
            self._returns_cache is None
            or self._returns_updated is None
            or (now - self._returns_updated).seconds > 300
        ):
            try:
                spy = yf.Ticker("SPY")
                hist = spy.history(period="1mo")
                returns = hist["Close"].pct_change().dropna()
                self._returns_cache = returns
                self._returns_updated = now
            except Exception:
                pass

        return self._returns_cache if self._returns_cache is not None else pd.Series(dtype=float)

    async def _execute_buy(
        self, qty: int, price: float, forecast_vol: float, uncertainty: float
    ) -> None:
        """Execute buy order."""
        order = Order(
            symbol="SPY",
            side="BUY",
            quantity=qty,
            price=price,
            order_id=f"buy_{uuid.uuid4().hex[:8]}",
            order_type="MARKET",
        )

        try:
            fill = await self.gateway.send_order(order)

            # Update state
            self.position = fill.fill_quantity
            self.avg_cost = fill.fill_price
            self.capital -= fill.fill_quantity * fill.fill_price
            self.total_trades += 1

            # Record trade
            self.db.record_trade(
                session_id=self.session_id,
                symbol="SPY",
                side="BUY",
                quantity=fill.fill_quantity,
                price=fill.fill_price,
                forecast_vol=forecast_vol,
                uncertainty=uncertainty,
            )

            self._print_trade("BUY", fill.fill_quantity, fill.fill_price, uncertainty, None)

        except ConnectionError as e:
            self._print_error(f"Buy failed: {e}")
            # Reconnect
            await self.gateway.connect()

    async def _execute_sell(
        self,
        price: float,
        forecast_vol: float,
        uncertainty: float,
        force: bool = False,
    ) -> None:
        """Execute sell order."""
        if self.position <= 0:
            return

        order = Order(
            symbol="SPY",
            side="SELL",
            quantity=self.position,
            price=price,
            order_id=f"sell_{uuid.uuid4().hex[:8]}",
            order_type="MARKET",
        )

        try:
            fill = await self.gateway.send_order(order)

            # Calculate P&L
            pnl = (fill.fill_price - self.avg_cost) * fill.fill_quantity
            self.cumulative_pnl += pnl

            # Update state
            self.capital += fill.fill_quantity * fill.fill_price
            self.position = 0
            self.avg_cost = 0.0
            self.total_trades += 1

            # Record trade
            self.db.record_trade(
                session_id=self.session_id,
                symbol="SPY",
                side="SELL",
                quantity=fill.fill_quantity,
                price=fill.fill_price,
                forecast_vol=forecast_vol,
                uncertainty=uncertainty,
                pnl=pnl,
                cumulative_pnl=self.cumulative_pnl,
            )

            self._print_trade("SELL", fill.fill_quantity, fill.fill_price, uncertainty, pnl)

        except ConnectionError as e:
            self._print_error(f"Sell failed: {e}")
            await self.gateway.connect()

    # Terminal output methods

    def _print_header(self) -> None:
        """Print ASCII art banner."""
        print(Fore.CYAN + """
    ╔═══════════════════════════════════════════╗
    ║   PAPER TRADING SYSTEM - Volatility Arb   ║
    ╚═══════════════════════════════════════════╝
        """ + Style.RESET_ALL)

    def _print_trade(
        self,
        side: str,
        qty: int,
        price: float,
        uncertainty: float,
        pnl: Optional[float],
    ) -> None:
        """Print trade execution."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = Fore.GREEN if side == "BUY" else Fore.RED

        pnl_str = f" | P&L: ${pnl:+.2f}" if pnl is not None else ""
        print(
            f"{Fore.WHITE}[{timestamp}] "
            f"SPY: ${price:.2f} | "
            f"Unc: {uncertainty:.4f} | "
            f"{color}{side} {qty} shares{Style.RESET_ALL}"
            f"{pnl_str}"
        )

    def _print_skip(self, price: float, uncertainty: float, forecast_vol: float) -> None:
        """Print skip message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"{Fore.YELLOW}[{timestamp}] "
            f"SPY: ${price:.2f} | "
            f"Unc: {uncertainty:.4f} > {self.uncertainty_threshold} | "
            f"Vol: {forecast_vol:.2%} | "
            f"SKIP{Style.RESET_ALL}"
        )

    def _print_hold(self, price: float, uncertainty: float, forecast_vol: float) -> None:
        """Print hold message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        unrealized = (price - self.avg_cost) * self.position if self.position > 0 else 0
        print(
            f"{Fore.CYAN}[{timestamp}] "
            f"SPY: ${price:.2f} | "
            f"Unc: {uncertainty:.4f} | "
            f"Vol: {forecast_vol:.2%} | "
            f"HOLD {self.position} @ ${self.avg_cost:.2f} | "
            f"Unrealized: ${unrealized:+.2f}{Style.RESET_ALL}"
        )

    def _print_error(self, msg: str) -> None:
        """Print error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{Fore.RED}[{timestamp}] ERROR: {msg}{Style.RESET_ALL}")

    def _print_periodic_summary(self) -> None:
        """Print summary every 10 ticks."""
        stats = self.get_stats()
        print(
            f"{Fore.MAGENTA}--- "
            f"Tick {self.tick_count} | "
            f"Trades: {stats['total_trades']} | "
            f"Win: {stats['win_rate']:.1f}% | "
            f"P&L: ${stats['total_pnl']:+.2f} | "
            f"Skipped: {stats['skipped_ticks']}"
            f" ---{Style.RESET_ALL}"
        )

    def _print_summary(self, stats: dict) -> None:
        """Print final session summary."""
        print(f"\n{Fore.CYAN}{'=' * 50}")
        print(f"Session {self.session_id} Summary")
        print(f"{'=' * 50}{Style.RESET_ALL}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Total P&L: ${stats['total_pnl']:+.2f}")
        print(f"Max Drawdown: {stats['max_drawdown']:.1f}%")
        print(f"Sharpe Estimate: {stats['sharpe_estimate']:.2f}")
        print(f"Skipped Ticks: {stats['skipped_ticks']}")
        print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
