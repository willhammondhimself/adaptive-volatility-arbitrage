"""Snapshot API endpoints for IV surface capture and replay."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from backend.schemas.options import SnapshotDetail, SnapshotMetadata
from backend.services.live_market_service import LiveMarketService
from backend.services.snapshot_db import get_snapshot_db

router = APIRouter(prefix="/api/v1/snapshots", tags=["snapshots"])

# Services
market_service = LiveMarketService()


@router.post("/capture/{symbol}", response_model=SnapshotMetadata)
async def capture_snapshot(symbol: str, expiry_count: int = 5) -> SnapshotMetadata:
    """
    Capture current IV surface for a symbol.

    Fetches live option chains and saves the IV surface as a snapshot.

    Args:
        symbol: Underlying symbol (e.g., SPY)
        expiry_count: Number of expirations to include (1-10)
    """
    try:
        # Get live IV surface
        from datetime import datetime

        first_chain = market_service.get_option_chain(symbol)
        underlying_price = first_chain.underlying_price
        available_expiries = first_chain.available_expiries[:expiry_count]

        # Collect strikes and IVs
        all_strikes = set()
        expiry_ivs = []

        for expiry in available_expiries:
            chain = market_service.get_option_chain(symbol, expiry)
            iv_by_strike = {}
            for call in chain.calls:
                if call.implied_volatility and call.implied_volatility > 0:
                    all_strikes.add(call.strike)
                    iv_by_strike[call.strike] = call.implied_volatility
            expiry_ivs.append(iv_by_strike)

        strikes = sorted(list(all_strikes))

        # Calculate maturities
        today = datetime.now()
        maturities = []
        for expiry in available_expiries:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            days = (expiry_date - today).days
            maturities.append(max(days / 365.0, 0.01))

        # Build IV matrix
        values: List[List[float]] = []
        for iv_by_strike in expiry_ivs:
            row = []
            for strike in strikes:
                iv = iv_by_strike.get(strike)
                row.append(iv if iv is not None else float("nan"))
            values.append(row)

        # Try to get VIX level
        vix_level = None
        try:
            vix_chain = market_service.get_option_chain("^VIX")
            vix_level = vix_chain.underlying_price
        except Exception:
            pass

        # Save snapshot
        db = get_snapshot_db()
        snapshot_id = db.save_snapshot(
            symbol=symbol,
            underlying_price=underlying_price,
            strikes=strikes,
            maturities=maturities,
            values=values,
            vix_level=vix_level,
        )

        # Return metadata
        snapshot = db.get_snapshot(snapshot_id)
        return SnapshotMetadata(
            id=snapshot.id,
            symbol=snapshot.symbol,
            captured_at=snapshot.captured_at,
            underlying_price=snapshot.underlying_price,
            vix_level=snapshot.vix_level,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[SnapshotMetadata])
async def list_snapshots(
    symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[SnapshotMetadata]:
    """
    List available snapshots.

    Args:
        symbol: Filter by symbol (optional)
        limit: Maximum number of results (default: 100)
        offset: Offset for pagination
    """
    db = get_snapshot_db()
    return db.list_snapshots(symbol=symbol, limit=limit, offset=offset)


@router.get("/symbols", response_model=List[str])
async def get_symbols() -> List[str]:
    """Get list of unique symbols with saved snapshots."""
    db = get_snapshot_db()
    return db.get_symbols()


@router.get("/{snapshot_id}", response_model=SnapshotDetail)
async def get_snapshot(snapshot_id: int) -> SnapshotDetail:
    """
    Get full snapshot data by ID.

    Args:
        snapshot_id: Snapshot ID
    """
    db = get_snapshot_db()
    snapshot = db.get_snapshot(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return snapshot


@router.delete("/{snapshot_id}")
async def delete_snapshot(snapshot_id: int):
    """
    Delete a snapshot.

    Args:
        snapshot_id: Snapshot ID
    """
    db = get_snapshot_db()
    if not db.delete_snapshot(snapshot_id):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {"status": "deleted", "id": snapshot_id}
