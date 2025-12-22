"""SQLite database service for IV surface snapshots."""

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from backend.schemas.options import SnapshotDetail, SnapshotMetadata


def _sanitize_for_json(values: List[List[float]]) -> List[List[Optional[float]]]:
    """Convert NaN values to None for JSON serialization."""
    result = []
    for row in values:
        sanitized_row = []
        for v in row:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                sanitized_row.append(None)
            else:
                sanitized_row.append(v)
        result.append(sanitized_row)
    return result


class SnapshotDB:
    """SQLite-backed storage for IV surface snapshots."""

    def __init__(self, db_path: str = "data/snapshots.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    underlying_price REAL NOT NULL,
                    vix_level REAL,
                    strikes TEXT NOT NULL,
                    maturities TEXT NOT NULL,
                    iv_values TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_symbol
                ON snapshots(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_captured_at
                ON snapshots(captured_at)
            """)
            conn.commit()

    def save_snapshot(
        self,
        symbol: str,
        underlying_price: float,
        strikes: List[float],
        maturities: List[float],
        values: List[List[float]],
        vix_level: Optional[float] = None,
    ) -> int:
        """
        Save an IV surface snapshot.

        Args:
            symbol: Underlying symbol
            underlying_price: Current underlying price
            strikes: List of strike prices
            maturities: List of maturities in years
            values: 2D array of IV values [maturity_idx][strike_idx]
            vix_level: Optional VIX level at capture time

        Returns:
            Snapshot ID
        """
        captured_at = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO snapshots
                (symbol, captured_at, underlying_price, vix_level, strikes, maturities, iv_values)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.upper(),
                    captured_at,
                    underlying_price,
                    vix_level,
                    json.dumps(strikes),
                    json.dumps(maturities),
                    json.dumps(_sanitize_for_json(values)),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def list_snapshots(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SnapshotMetadata]:
        """
        List snapshot metadata.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of snapshot metadata
        """
        with self._get_connection() as conn:
            if symbol:
                rows = conn.execute(
                    """
                    SELECT id, symbol, captured_at, underlying_price, vix_level
                    FROM snapshots
                    WHERE symbol = ?
                    ORDER BY captured_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (symbol.upper(), limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, symbol, captured_at, underlying_price, vix_level
                    FROM snapshots
                    ORDER BY captured_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()

            return [
                SnapshotMetadata(
                    id=row["id"],
                    symbol=row["symbol"],
                    captured_at=row["captured_at"],
                    underlying_price=row["underlying_price"],
                    vix_level=row["vix_level"],
                )
                for row in rows
            ]

    def get_snapshot(self, snapshot_id: int) -> Optional[SnapshotDetail]:
        """
        Get full snapshot data by ID.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            Full snapshot data or None if not found
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, symbol, captured_at, underlying_price, vix_level,
                       strikes, maturities, iv_values
                FROM snapshots
                WHERE id = ?
                """,
                (snapshot_id,),
            ).fetchone()

            if not row:
                return None

            return SnapshotDetail(
                id=row["id"],
                symbol=row["symbol"],
                captured_at=row["captured_at"],
                underlying_price=row["underlying_price"],
                vix_level=row["vix_level"],
                strikes=json.loads(row["strikes"]),
                maturities=json.loads(row["maturities"]),
                values=json.loads(row["iv_values"]),
            )

    def delete_snapshot(self, snapshot_id: int) -> bool:
        """
        Delete a snapshot by ID.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM snapshots WHERE id = ?",
                (snapshot_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_symbols(self) -> List[str]:
        """Get list of unique symbols with snapshots."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM snapshots ORDER BY symbol"
            ).fetchall()
            return [row["symbol"] for row in rows]


# Singleton instance
_snapshot_db: Optional[SnapshotDB] = None


def get_snapshot_db() -> SnapshotDB:
    """Get or create singleton SnapshotDB instance."""
    global _snapshot_db
    if _snapshot_db is None:
        _snapshot_db = SnapshotDB()
    return _snapshot_db
