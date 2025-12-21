#!/usr/bin/env python3
"""
Standalone paper trading CLI.

Usage:
    python live_trading/demo_paper_trader.py
    python live_trading/demo_paper_trader.py --capital 50000 --threshold 0.015
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from backend.services.paper_trading_system import PaperTradingSystem


async def main(capital: float, threshold: float, position_pct: float, interval: float):
    """Run paper trading system."""
    system = PaperTradingSystem(
        initial_capital=capital,
        uncertainty_threshold=threshold,
        position_pct=position_pct,
        loop_interval=interval,
    )

    # Graceful shutdown flag
    shutdown_event = asyncio.Event()

    def shutdown_handler(sig, frame):
        print("\n\nShutdown signal received...")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        await system.start()

        # Wait until shutdown signal
        await shutdown_event.wait()

    finally:
        if system._running:
            await system.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper trading CLI for volatility arbitrage"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Uncertainty threshold (default: 0.02)",
    )
    parser.add_argument(
        "--position-pct",
        type=float,
        default=0.10,
        help="Position size as fraction (default: 0.10)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Loop interval in seconds (default: 30)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.capital, args.threshold, args.position_pct, args.interval))
    except KeyboardInterrupt:
        pass
