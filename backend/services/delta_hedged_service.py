"""
Delta-hedged backtest service.
"""

import time
from datetime import timedelta
from decimal import Decimal

import numpy as np

from backend.schemas.backtest import (
    AttributionPoint,
    DeltaHedgedMetrics,
    DeltaHedgedRequest,
    DeltaHedgedResponse,
)
from volatility_arbitrage.backtest.delta_hedged import (
    DeltaHedgedBacktest,
    HedgeConfig,
    RebalanceFrequency,
)
from volatility_arbitrage.backtest.delta_hedged.engine import (
    generate_gbm_with_stochastic_vol,
)
from volatility_arbitrage.models.black_scholes import OptionType


class DeltaHedgedService:
    """Service for running delta-hedged backtests."""

    FREQ_MAP = {
        "continuous": RebalanceFrequency.CONTINUOUS,
        "hourly": RebalanceFrequency.HOURLY,
        "four_hour": RebalanceFrequency.FOUR_HOUR,
        "daily": RebalanceFrequency.DAILY,
    }

    def run(self, request: DeltaHedgedRequest) -> DeltaHedgedResponse:
        """Run delta-hedged backtest with synthetic data."""
        start_time = time.time()

        # Generate synthetic price/IV path (returns DataFrame)
        data = generate_gbm_with_stochastic_vol(
            days=request.days,
            initial_spot=request.initial_spot,
            initial_iv=request.initial_iv,
            vol_of_vol=request.vol_of_vol,
            mean_reversion=request.mean_reversion,
        )

        # Configure hedge
        config = HedgeConfig(
            rebalance_frequency=self.FREQ_MAP[request.rebalance_frequency],
            delta_threshold=Decimal(str(request.delta_threshold)),
            daily_volume=request.daily_volume,
        )

        # Calculate expiry from start timestamp
        start_timestamp = data.iloc[0]["timestamp"]
        expiry = start_timestamp + timedelta(days=request.expiry_days)

        # Run backtest
        option_type = OptionType.CALL if request.option_type == "call" else OptionType.PUT
        backtest = DeltaHedgedBacktest(
            config=config,
            option_position=request.option_position,
        )
        metrics = backtest.run(
            data=data,
            strike=Decimal(str(request.strike)),
            expiry=expiry,
            option_type=option_type,
        )

        # Get attribution DataFrame with cumulative columns
        attr_df = backtest.get_attribution_df()

        # Build attribution time series from cumulative columns
        attribution = []
        if not attr_df.empty:
            for _, row in attr_df.iterrows():
                # timestamp is already a string from to_dict()
                ts = row["timestamp"]
                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                attribution.append(
                    AttributionPoint(
                        timestamp=ts_str,
                        total_pnl=float(row["cum_total_pnl"]),
                        delta_pnl=float(row["cum_delta_pnl"]),
                        gamma_pnl=float(row["cum_gamma_pnl"]),
                        vega_pnl=float(row["cum_vega_pnl"]),
                        theta_pnl=float(row["cum_theta_pnl"]),
                        transaction_costs=float(row["cum_transaction_costs"]),
                    )
                )

        # Get final cumulative values for metrics
        cum_total = float(attr_df["cum_total_pnl"].iloc[-1]) if not attr_df.empty else 0.0
        cum_delta = float(attr_df["cum_delta_pnl"].iloc[-1]) if not attr_df.empty else 0.0
        cum_gamma = float(attr_df["cum_gamma_pnl"].iloc[-1]) if not attr_df.empty else 0.0
        cum_vega = float(attr_df["cum_vega_pnl"].iloc[-1]) if not attr_df.empty else 0.0
        cum_theta = float(attr_df["cum_theta_pnl"].iloc[-1]) if not attr_df.empty else 0.0

        # Calculate max drawdown from cumulative total
        if not attr_df.empty:
            cumulative = attr_df["cum_total_pnl"].values
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            max_drawdown = 0.0

        response_metrics = DeltaHedgedMetrics(
            total_pnl=cum_total,
            total_sharpe=metrics.total_sharpe,
            vega_gamma_pnl=cum_vega + cum_gamma,
            vega_gamma_sharpe=metrics.vega_gamma_sharpe,
            delta_pnl=cum_delta,
            theta_pnl=cum_theta,
            transaction_costs=metrics.total_transaction_costs,
            rebalance_count=metrics.rebalance_count,
            hedge_effectiveness=metrics.hedge_effectiveness,
            max_drawdown=max_drawdown,
        )

        computation_time = (time.time() - start_time) * 1000

        return DeltaHedgedResponse(
            metrics=response_metrics,
            attribution=attribution,
            computation_time_ms=computation_time,
        )
