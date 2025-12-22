"""
Delta-hedged backtest engine.

Runs backtests with P&L attribution to demonstrate that alpha comes from
volatility (Vega/Gamma) rather than directional exposure (Delta).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

from volatility_arbitrage.models.black_scholes import OptionType

from .hedger import DeltaHedger
from .types import HedgeConfig, PnLAttribution


@dataclass
class BacktestMetrics:
    """Metrics from a delta-hedged backtest."""

    total_pnl: float
    total_sharpe: float
    vega_gamma_sharpe: float
    hedge_effectiveness: float
    delta_pnl_variance_ratio: float
    total_transaction_costs: float
    rebalance_count: int
    num_periods: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_pnl": self.total_pnl,
            "total_sharpe": self.total_sharpe,
            "vega_gamma_sharpe": self.vega_gamma_sharpe,
            "hedge_effectiveness": self.hedge_effectiveness,
            "delta_pnl_variance_ratio": self.delta_pnl_variance_ratio,
            "total_transaction_costs": self.total_transaction_costs,
            "rebalance_count": self.rebalance_count,
            "num_periods": self.num_periods,
        }


class DeltaHedgedBacktest:
    """
    Delta-hedged backtest runner.

    Takes historical or simulated data and runs a delta-hedged options
    strategy, tracking P&L attribution throughout.
    """

    def __init__(
        self,
        config: HedgeConfig,
        option_position: int = 100,
        risk_free_rate: Decimal = Decimal("0.05"),
    ):
        """
        Initialize the backtest.

        Args:
            config: Hedge configuration
            option_position: Number of option contracts (positive = long)
            risk_free_rate: Annual risk-free rate
        """
        self.config = config
        self.option_position = option_position
        self.risk_free_rate = risk_free_rate
        self.hedger = DeltaHedger(config, risk_free_rate)

    def run(
        self,
        data: pd.DataFrame,
        strike: Decimal,
        expiry: datetime,
        option_type: OptionType = OptionType.CALL,
    ) -> BacktestMetrics:
        """
        Run the backtest.

        Args:
            data: DataFrame with columns ['timestamp', 'spot', 'iv']
            strike: Option strike price
            expiry: Option expiry datetime
            option_type: CALL or PUT

        Returns:
            BacktestMetrics with performance summary
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for backtest")

        # Reset hedger
        self.hedger.reset()

        # Initialize with first row
        first_row = data.iloc[0]
        self.hedger.initialize(
            timestamp=pd.Timestamp(first_row["timestamp"]).to_pydatetime(),
            spot=Decimal(str(first_row["spot"])),
            iv=Decimal(str(first_row["iv"])),
            option_position=Decimal(str(self.option_position)),
            strike=strike,
            expiry=expiry,
            option_type=option_type,
        )

        # Iterate through data
        for _, row in data.iloc[1:].iterrows():
            timestamp = pd.Timestamp(row["timestamp"]).to_pydatetime()

            # Stop if past expiry
            if timestamp >= expiry:
                break

            self.hedger.update(
                timestamp=timestamp,
                spot=Decimal(str(row["spot"])),
                iv=Decimal(str(row["iv"])),
            )

        return self.calculate_metrics()

    def get_attribution_df(self) -> pd.DataFrame:
        """
        Get attribution history as DataFrame.

        Returns:
            DataFrame with columns for each P&L component and cumulative versions
        """
        if not self.hedger.attribution_history:
            return pd.DataFrame()

        records = [attr.to_dict() for attr in self.hedger.attribution_history]
        df = pd.DataFrame(records)

        # Add cumulative columns
        pnl_cols = [
            "total_pnl",
            "delta_pnl",
            "gamma_pnl",
            "vega_pnl",
            "theta_pnl",
            "transaction_costs",
        ]
        for col in pnl_cols:
            if col in df.columns:
                df[f"cum_{col}"] = df[col].cumsum()

        return df

    def calculate_metrics(self) -> BacktestMetrics:
        """
        Calculate backtest metrics.

        Returns:
            BacktestMetrics with Sharpe ratios, hedge effectiveness, etc.
        """
        df = self.get_attribution_df()

        if df.empty:
            return BacktestMetrics(
                total_pnl=0.0,
                total_sharpe=0.0,
                vega_gamma_sharpe=0.0,
                hedge_effectiveness=1.0,
                delta_pnl_variance_ratio=0.0,
                total_transaction_costs=0.0,
                rebalance_count=0,
                num_periods=0,
            )

        # Calculate Sharpe ratios (annualized assuming daily data)
        total_sharpe = self._calculate_sharpe(df["total_pnl"])
        vega_gamma_pnl = df["gamma_pnl"] + df["vega_pnl"]
        vega_gamma_sharpe = self._calculate_sharpe(vega_gamma_pnl)

        # Hedge effectiveness: 1 - (delta variance / total variance)
        total_var = df["total_pnl"].var()
        delta_var = df["delta_pnl"].var()

        if total_var > 0:
            delta_pnl_variance_ratio = delta_var / total_var
            hedge_effectiveness = 1.0 - delta_pnl_variance_ratio
        else:
            delta_pnl_variance_ratio = 0.0
            hedge_effectiveness = 1.0

        return BacktestMetrics(
            total_pnl=df["total_pnl"].sum(),
            total_sharpe=total_sharpe,
            vega_gamma_sharpe=vega_gamma_sharpe,
            hedge_effectiveness=hedge_effectiveness,
            delta_pnl_variance_ratio=delta_pnl_variance_ratio,
            total_transaction_costs=df["transaction_costs"].sum(),
            rebalance_count=self.hedger.rebalance_count,
            num_periods=len(df),
        )

    def _calculate_sharpe(
        self, returns: pd.Series, periods_per_year: float = 252.0
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Series of returns (not cumulative)
            periods_per_year: Number of periods per year for annualization

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * np.sqrt(periods_per_year)

        return annualized_return / annualized_std


def generate_gbm_with_stochastic_vol(
    days: int = 30,
    initial_spot: float = 450.0,
    initial_iv: float = 0.20,
    vol_of_vol: float = 0.5,
    mean_reversion: float = 2.0,
    long_run_vol: float = 0.20,
    spot_vol_corr: float = -0.7,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic price data with stochastic volatility.

    Uses a simplified Heston-like model where IV follows a
    mean-reverting process correlated with spot returns.

    Args:
        days: Number of days to simulate
        initial_spot: Starting spot price
        initial_iv: Starting implied volatility
        vol_of_vol: Volatility of volatility (σ_v)
        mean_reversion: Speed of mean reversion (κ)
        long_run_vol: Long-run volatility level (θ)
        spot_vol_corr: Correlation between spot and vol (ρ)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns ['timestamp', 'spot', 'iv']
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate hourly data
    n_points = days * 8  # 8 hours of trading per day
    dt = 1 / 252 / 8  # Time step in years

    timestamps = []
    spots = []
    ivs = []

    spot = initial_spot
    iv = initial_iv
    start_time = datetime(2024, 1, 2, 9, 30)  # Market open

    for i in range(n_points):
        # Store current values
        hour_offset = i % 8
        day_offset = i // 8
        timestamp = start_time + timedelta(days=day_offset, hours=hour_offset)

        timestamps.append(timestamp)
        spots.append(spot)
        ivs.append(iv)

        # Generate correlated random shocks
        z1 = np.random.standard_normal()
        z2 = spot_vol_corr * z1 + np.sqrt(1 - spot_vol_corr**2) * np.random.standard_normal()

        # Update spot (GBM)
        spot_return = (0.05 - 0.5 * iv**2) * dt + iv * np.sqrt(dt) * z1
        spot = spot * np.exp(spot_return)

        # Update IV (mean-reverting process)
        iv_change = mean_reversion * (long_run_vol - iv) * dt + vol_of_vol * iv * np.sqrt(dt) * z2
        iv = max(0.05, iv + iv_change)  # Floor at 5%

    return pd.DataFrame({"timestamp": timestamps, "spot": spots, "iv": ivs})


def generate_vol_spike_scenario(
    days: int = 30,
    initial_spot: float = 450.0,
    initial_iv: float = 0.15,
    spike_day: int = 10,
    spike_magnitude: float = 0.15,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate data with a volatility spike.

    Useful for testing vega P&L attribution.

    Args:
        days: Number of days to simulate
        initial_spot: Starting spot price
        initial_iv: Starting implied volatility
        spike_day: Day on which IV spikes
        spike_magnitude: Size of IV increase
        seed: Random seed

    Returns:
        DataFrame with columns ['timestamp', 'spot', 'iv']
    """
    if seed is not None:
        np.random.seed(seed)

    n_points = days * 8
    dt = 1 / 252 / 8

    timestamps = []
    spots = []
    ivs = []

    spot = initial_spot
    iv = initial_iv
    start_time = datetime(2024, 1, 2, 9, 30)

    for i in range(n_points):
        hour_offset = i % 8
        day_offset = i // 8
        timestamp = start_time + timedelta(days=day_offset, hours=hour_offset)

        # IV spike on spike_day
        if day_offset == spike_day and hour_offset == 0:
            iv += spike_magnitude

        timestamps.append(timestamp)
        spots.append(spot)
        ivs.append(iv)

        # Update spot
        z = np.random.standard_normal()
        spot_return = -0.5 * iv**2 * dt + iv * np.sqrt(dt) * z
        spot = spot * np.exp(spot_return)

        # IV slowly decays back after spike
        if day_offset > spike_day:
            iv = max(initial_iv, iv - 0.01 * dt * 252)

    return pd.DataFrame({"timestamp": timestamps, "spot": spots, "iv": ivs})
