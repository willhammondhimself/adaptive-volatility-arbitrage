"""
FastAPI application for Volatility Arbitrage Dashboard.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import heston, forecast, costs, backtest

app = FastAPI(
    title="Volatility Arbitrage Dashboard API",
    description="Interactive dashboard for Heston option pricing and volatility arbitrage backtesting",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(heston.router)
app.include_router(forecast.router)
app.include_router(costs.router)
app.include_router(backtest.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Volatility Arbitrage Dashboard API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "volatility-arbitrage-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
