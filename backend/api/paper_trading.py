"""
FastAPI endpoints for paper trading control and monitoring.
"""

from fastapi import APIRouter, HTTPException

from backend.schemas.paper_trading import (
    StartRequest,
    StartResponse,
    StopResponse,
    StatusResponse,
    TradesResponse,
    TradeRecord,
    StatsResponse,
)
from backend.services.paper_trading_system import PaperTradingSystem

router = APIRouter(prefix="/api/v1/paper-trading", tags=["paper-trading"])


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current paper trading status."""
    system = PaperTradingSystem.get_instance()
    return StatusResponse(
        is_running=system._running,
        session_id=system.session_id,
        capital=system.capital,
        position=system.position,
        avg_cost=system.avg_cost,
        cumulative_pnl=system.cumulative_pnl,
        last_update=system.last_update,
        tick_count=system.tick_count,
    )


@router.get("/trades", response_model=TradesResponse)
async def get_trades(limit: int = 100):
    """Get recent trades for current session."""
    system = PaperTradingSystem.get_instance()

    if not system.session_id:
        return TradesResponse(session_id=None, trades=[])

    trades_raw = system.db.get_trades(system.session_id, limit)
    trades = [
        TradeRecord(
            id=t["id"],
            timestamp=t["timestamp"],
            symbol=t["symbol"],
            side=t["side"],
            quantity=t["quantity"],
            price=t["price"],
            forecast_vol=t["forecast_vol"],
            uncertainty=t["uncertainty"],
            pnl=t["pnl"],
            cumulative_pnl=t["cumulative_pnl"],
        )
        for t in trades_raw
    ]
    return TradesResponse(session_id=system.session_id, trades=trades)


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get trading statistics for current session."""
    system = PaperTradingSystem.get_instance()
    stats = system.get_stats()
    return StatsResponse(**stats)


@router.post("/start", response_model=StartResponse)
async def start_trading(request: StartRequest):
    """Start paper trading with given configuration."""
    system = PaperTradingSystem.get_instance()

    if system._running:
        raise HTTPException(status_code=400, detail="Trading already running")

    # Update config
    system.initial_capital = request.initial_capital
    system.capital = request.initial_capital
    system.uncertainty_threshold = request.uncertainty_threshold
    system.position_pct = request.position_pct

    try:
        session_id = await system.start()
        return StartResponse(session_id=session_id, message="Trading started")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=StopResponse)
async def stop_trading():
    """Stop paper trading and return final stats."""
    system = PaperTradingSystem.get_instance()

    if not system._running:
        raise HTTPException(status_code=400, detail="Trading not running")

    try:
        stats = await system.stop()
        return StopResponse(
            session_id=system.session_id,
            stats=StatsResponse(**stats),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
