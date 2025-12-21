"""
SQLite database layer for paper trading persistence.

Tables:
- sessions: Trading session metadata with start/end times and aggregate stats
- trades: Individual trade records with P&L tracking
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "live_trading" / "trades.db"


class PaperTradingDB:
    """SQLite persistence for paper trading sessions and trades."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a connection with row factory for dict-like access."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    initial_capital REAL NOT NULL,
                    final_capital REAL,
                    total_trades INTEGER DEFAULT 0,
                    total_skipped INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL DEFAULT 'SPY',
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    forecast_vol REAL NOT NULL,
                    uncertainty REAL NOT NULL,
                    pnl REAL,
                    cumulative_pnl REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id);
            """)

    def resume_or_create_session(self, initial_capital: float) -> int:
        """
        Resume a crashed session or create a new one.

        If there's a running session that wasn't properly closed, resume it.
        Otherwise, create a fresh session.

        Returns:
            session_id
        """
        with self._get_conn() as conn:
            # Check for unfinished session
            cursor = conn.execute(
                "SELECT id, initial_capital FROM sessions WHERE status = 'running' ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()

            if row:
                # Resume existing session
                return row["id"]

            # Create new session
            cursor = conn.execute(
                "INSERT INTO sessions (started_at, status, initial_capital) VALUES (?, 'running', ?)",
                (datetime.now().isoformat(), initial_capital),
            )
            return cursor.lastrowid

    def end_session(
        self, session_id: int, final_capital: float, total_trades: int, total_skipped: int
    ) -> None:
        """Mark session as stopped with final stats."""
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET ended_at = ?, status = 'stopped', final_capital = ?,
                    total_trades = ?, total_skipped = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(), final_capital, total_trades, total_skipped, session_id),
            )

    def record_trade(
        self,
        session_id: int,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        forecast_vol: float,
        uncertainty: float,
        pnl: Optional[float] = None,
        cumulative_pnl: Optional[float] = None,
    ) -> int:
        """Insert trade record, return trade_id."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades
                (session_id, timestamp, symbol, side, quantity, price, forecast_vol, uncertainty, pnl, cumulative_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    datetime.now().isoformat(),
                    symbol,
                    side,
                    quantity,
                    price,
                    forecast_vol,
                    uncertainty,
                    pnl,
                    cumulative_pnl,
                ),
            )
            return cursor.lastrowid

    def increment_skipped(self, session_id: int) -> None:
        """Increment skipped tick counter for session."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET total_skipped = total_skipped + 1 WHERE id = ?",
                (session_id,),
            )

    def get_session(self, session_id: int) -> Optional[dict]:
        """Get session by ID."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_trades(self, session_id: int, limit: int = 100) -> list[dict]:
        """Get trades for a session, most recent first."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM trades WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_trades(self, limit: int = 100) -> list[dict]:
        """Get all trades across sessions, most recent first."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self, session_id: int) -> dict:
        """Calculate trading statistics for a session."""
        with self._get_conn() as conn:
            # Get session info
            session = self.get_session(session_id)
            if not session:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_estimate": 0.0,
                    "skipped_ticks": 0,
                }

            # Get trades
            cursor = conn.execute(
                "SELECT pnl, cumulative_pnl FROM trades WHERE session_id = ? AND pnl IS NOT NULL ORDER BY id",
                (session_id,),
            )
            trades = cursor.fetchall()

            if not trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_estimate": 0.0,
                    "skipped_ticks": session["total_skipped"] or 0,
                }

            # Calculate stats
            pnls = [t["pnl"] for t in trades]
            cumulative = [t["cumulative_pnl"] for t in trades]

            wins = sum(1 for p in pnls if p > 0)
            total_pnl = sum(pnls)
            win_rate = (wins / len(pnls)) * 100 if pnls else 0

            # Max drawdown from cumulative P&L
            peak = 0
            max_dd = 0
            for c in cumulative:
                peak = max(peak, c)
                dd = (peak - c) / max(peak, 1) * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)

            # Sharpe estimate (simplified)
            if len(pnls) > 1:
                import statistics
                mean_pnl = statistics.mean(pnls)
                std_pnl = statistics.stdev(pnls)
                sharpe = (mean_pnl / std_pnl) * (252 ** 0.5) if std_pnl > 0 else 0
            else:
                sharpe = 0

            return {
                "total_trades": len(pnls),
                "win_rate": round(win_rate, 1),
                "total_pnl": round(total_pnl, 2),
                "max_drawdown": round(max_dd, 1),
                "sharpe_estimate": round(sharpe, 2),
                "skipped_ticks": session["total_skipped"] or 0,
            }
