"""
Mock exchange connection for testing execution infrastructure.
Simulates real-world conditions: disconnects, latency jitter, heartbeats.
"""
import asyncio
import random
import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    symbol: str
    side: str  # 'BUY' | 'SELL'
    quantity: int
    price: float
    order_id: str
    order_type: str = "LIMIT"  # 'LIMIT' | 'MARKET'


@dataclass
class Fill:
    order_id: str
    status: OrderStatus
    fill_price: float
    fill_quantity: int
    latency_ms: float
    timestamp: float


class MockExchangeGateway:
    """
    Async exchange simulator with realistic failure modes.

    - Random disconnects (configurable probability per message)
    - Latency jitter (10-50ms by default)
    - Requires heartbeat every 30s or connection drops
    - Partial fills on large orders
    """

    def __init__(
        self,
        disconnect_prob: float = 0.05,
        min_latency_ms: float = 10.0,
        max_latency_ms: float = 50.0,
        heartbeat_timeout_s: float = 30.0,
        partial_fill_threshold: int = 100,
    ):
        self._connected = False
        self._last_heartbeat: float = 0.0
        self._disconnect_prob = disconnect_prob
        self._min_latency = min_latency_ms / 1000.0
        self._max_latency = max_latency_ms / 1000.0
        self._heartbeat_timeout = heartbeat_timeout_s
        self._partial_fill_threshold = partial_fill_threshold
        self._logger = logging.getLogger(__name__)
        self._order_count = 0

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Establish connection with simulated network latency."""
        await asyncio.sleep(random.uniform(0.1, 0.3))
        self._connected = True
        self._last_heartbeat = asyncio.get_event_loop().time()
        self._logger.info("Connected to mock exchange")

    async def disconnect(self) -> None:
        """Graceful disconnect."""
        self._connected = False
        self._logger.info("Disconnected from mock exchange")

    async def send_order(self, order: Order) -> Fill:
        """
        Submit order to mock exchange.

        Raises:
            ConnectionError: If not connected or heartbeat timeout
        """
        self._check_connection()
        await self._simulate_latency()

        # Random disconnect
        if random.random() < self._disconnect_prob:
            self._connected = False
            self._logger.warning("Connection lost during order submission")
            raise ConnectionError("Exchange connection lost")

        latency = random.uniform(self._min_latency * 1000, self._max_latency * 1000)

        # Simulate partial fills for large orders
        if order.quantity > self._partial_fill_threshold:
            fill_qty = random.randint(
                order.quantity // 2, order.quantity
            )
            status = OrderStatus.PARTIAL if fill_qty < order.quantity else OrderStatus.FILLED
        else:
            fill_qty = order.quantity
            status = OrderStatus.FILLED

        # Price slippage simulation
        slippage = random.uniform(-0.001, 0.001)
        fill_price = order.price * (1 + slippage)

        self._order_count += 1
        self._logger.debug(
            f"Order {order.order_id}: {status.value} {fill_qty}/{order.quantity} @ {fill_price:.4f}"
        )

        return Fill(
            order_id=order.order_id,
            status=status,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            latency_ms=latency,
            timestamp=asyncio.get_event_loop().time(),
        )

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order.

        Returns True if cancel succeeded, False if order already filled.
        """
        self._check_connection()
        await self._simulate_latency()

        # 80% chance cancel succeeds (order not yet filled)
        success = random.random() < 0.8
        if success:
            self._logger.debug(f"Order {order_id} cancelled")
        else:
            self._logger.debug(f"Order {order_id} already filled, cancel failed")
        return success

    async def heartbeat(self) -> None:
        """Send heartbeat to keep connection alive."""
        self._check_connection()
        self._last_heartbeat = asyncio.get_event_loop().time()
        self._logger.debug("Heartbeat sent")

    async def get_position(self, symbol: str) -> dict:
        """Query current position (mock always returns 0)."""
        self._check_connection()
        await self._simulate_latency()
        return {"symbol": symbol, "quantity": 0, "avg_price": 0.0}

    async def _simulate_latency(self) -> None:
        """Add realistic network latency."""
        await asyncio.sleep(random.uniform(self._min_latency, self._max_latency))

    def _check_connection(self) -> None:
        """Verify connection is alive and heartbeat is current."""
        if not self._connected:
            raise ConnectionError("Not connected to exchange")

        now = asyncio.get_event_loop().time()
        if now - self._last_heartbeat > self._heartbeat_timeout:
            self._connected = False
            self._logger.warning("Heartbeat timeout, connection dropped")
            raise ConnectionError("Heartbeat timeout")


class HeartbeatManager:
    """Background task to maintain connection with periodic heartbeats."""

    def __init__(self, gateway: MockExchangeGateway, interval_s: float = 15.0):
        self._gateway = gateway
        self._interval = interval_s
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start heartbeat loop."""
        self._task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop heartbeat loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats at regular intervals."""
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._gateway.heartbeat()
            except ConnectionError:
                break
            except asyncio.CancelledError:
                break


async def _demo():
    """Demo usage of mock exchange gateway."""
    logging.basicConfig(level=logging.DEBUG)

    gateway = MockExchangeGateway(disconnect_prob=0.02)
    heartbeat_mgr = HeartbeatManager(gateway, interval_s=10.0)

    await gateway.connect()
    await heartbeat_mgr.start()

    try:
        for i in range(5):
            order = Order(
                symbol="SPY",
                side="BUY",
                quantity=10,
                price=450.0,
                order_id=f"order_{i}",
            )
            try:
                fill = await gateway.send_order(order)
                print(f"Fill: {fill.status.value} @ {fill.fill_price:.2f} ({fill.latency_ms:.1f}ms)")
            except ConnectionError as e:
                print(f"Order failed: {e}")
                await gateway.connect()
    finally:
        await heartbeat_mgr.stop()
        await gateway.disconnect()


if __name__ == "__main__":
    asyncio.run(_demo())
