"""
Performance metrics for backtest analysis.

Calculates key risk-adjusted performance measures including Sharpe ratio,
maximum drawdown, Calmar ratio, and trade statistics.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

from volatility_arbitrage.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a backtest.

    Contains both return-based and risk-adjusted metrics.
    """

    # Return metrics
    total_return: Decimal
    total_return_pct: Decimal
    annualized_return: Decimal

    # Risk metrics
    volatility: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal

    # Risk-adjusted metrics
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal

    # Trade statistics
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: Decimal

    # Time metrics
    trading_days: int

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_return": float(self.total_return),
            "total_return_pct": float(self.total_return_pct),
            "annualized_return": float(self.annualized_return),
            "volatility": float(self.volatility),
            "max_drawdown": float(self.max_drawdown),
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "sharpe_ratio": float(self.sharpe_ratio),
            "sortino_ratio": float(self.sortino_ratio),
            "calmar_ratio": float(self.calmar_ratio),
            "num_trades": self.num_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": float(self.win_rate),
            "avg_win": float(self.avg_win),
            "avg_loss": float(self.avg_loss),
            "profit_factor": float(self.profit_factor),
            "trading_days": self.trading_days,
        }


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: Decimal = Decimal("0.05"),
    periods_per_year: int = 252,
    adjust_autocorrelation: bool = True,
) -> Decimal:
    """
    Calculate Sharpe ratio with optional Newey-West autocorrelation adjustment.

    Volatility arbitrage returns are often autocorrelated due to mean reversion
    in volatility. The standard Sharpe ratio assumes IID returns, which understates
    the true standard deviation when returns are correlated. The Newey-West
    adjustment accounts for this serial correlation.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)
        adjust_autocorrelation: Whether to apply Newey-West adjustment (default True)

    Returns:
        Sharpe ratio (adjusted for autocorrelation if enabled)
    """
    if len(returns) < 2:
        return Decimal("0")

    # Calculate excess returns
    daily_rf = float(risk_free_rate) / periods_per_year
    excess_returns = returns - daily_rf

    # Calculate mean of excess returns
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0 or np.isnan(std_excess):
        return Decimal("0")

    # Apply Newey-West adjustment for autocorrelation if enabled
    if adjust_autocorrelation and len(returns) > 10:
        try:
            # Calculate autocorrelation function
            n = len(excess_returns)
            # Optimal lag selection (Newey-West formula)
            max_lag = int(np.floor(4 * (n / 100) ** (2/9)))
            max_lag = max(1, min(max_lag, n // 4))  # Bound lag

            # Calculate autocorrelations
            acf_values = []
            for lag in range(1, max_lag + 1):
                if lag < len(excess_returns):
                    autocorr = excess_returns.autocorr(lag=lag)
                    if not np.isnan(autocorr):
                        acf_values.append(autocorr)
                    else:
                        acf_values.append(0.0)
                else:
                    acf_values.append(0.0)

            # Newey-West variance adjustment factor
            # adjustment = 1 + 2 * sum((1 - i/(lag+1)) * rho_i for i in 1..lag)
            adjustment = 1.0
            for i, rho in enumerate(acf_values, 1):
                weight = 1 - i / (max_lag + 1)  # Bartlett kernel
                adjustment += 2 * weight * rho

            # Apply adjustment (only if it increases std, i.e., positive autocorrelation)
            if adjustment > 1:
                std_excess = std_excess * np.sqrt(adjustment)
                logger.debug(
                    f"Newey-West adjustment applied",
                    extra={
                        "adjustment_factor": adjustment,
                        "original_std": float(excess_returns.std()),
                        "adjusted_std": float(std_excess),
                    }
                )
        except Exception as e:
            logger.warning(f"Newey-West adjustment failed, using standard Sharpe: {e}")

    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

    return Decimal(str(sharpe))


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: Decimal = Decimal("0.05"),
    periods_per_year: int = 252,
) -> Decimal:
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return Decimal("0")

    # Calculate excess returns
    daily_rf = float(risk_free_rate) / periods_per_year
    excess_returns = returns - daily_rf

    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return Decimal("999.99")  # Cap at very high value

    downside_std = downside_returns.std()

    if downside_std == 0 or np.isnan(downside_std):
        return Decimal("0")

    # Calculate Sortino
    mean_excess = excess_returns.mean()
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)

    return Decimal(str(sortino))


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple[Decimal, Decimal]:
    """
    Calculate maximum drawdown in both absolute and percentage terms.

    Args:
        equity_curve: Series of equity values over time

    Returns:
        Tuple of (max_drawdown_absolute, max_drawdown_percentage)
    """
    if len(equity_curve) < 2:
        return Decimal("0"), Decimal("0")

    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = equity_curve - running_max
    drawdown_pct = (drawdown / running_max) * 100

    # Get maximum drawdown
    max_dd = abs(drawdown.min())
    max_dd_pct = abs(drawdown_pct.min())

    return Decimal(str(max_dd)), Decimal(str(max_dd_pct))


def calculate_calmar_ratio(
    annualized_return: Decimal,
    max_drawdown_pct: Decimal,
) -> Decimal:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        annualized_return: Annualized return (as percentage)
        max_drawdown_pct: Maximum drawdown (as percentage)

    Returns:
        Calmar ratio
    """
    if max_drawdown_pct == 0:
        return Decimal("999.99")  # Cap at very high value

    return annualized_return / max_drawdown_pct


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate returns from equity curve.

    Args:
        equity_curve: Series of equity values

    Returns:
        Series of returns
    """
    return equity_curve.pct_change().fillna(0)


def calculate_comprehensive_metrics(
    equity_curve: pd.DataFrame,
    initial_capital: Decimal,
    final_capital: Decimal,
    num_trades: int = 0,
    winning_trades: int = 0,
    losing_trades: int = 0,
    total_wins: Decimal = Decimal("0"),
    total_losses: Decimal = Decimal("0"),
    risk_free_rate: Decimal = Decimal("0.05"),
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: DataFrame with equity over time
        initial_capital: Starting capital
        final_capital: Ending capital
        num_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        total_wins: Total profit from winning trades
        total_losses: Total loss from losing trades
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics instance
    """
    if equity_curve.empty:
        logger.warning("Empty equity curve, returning zero metrics")
        return PerformanceMetrics(
            total_return=Decimal("0"),
            total_return_pct=Decimal("0"),
            annualized_return=Decimal("0"),
            volatility=Decimal("0"),
            max_drawdown=Decimal("0"),
            max_drawdown_pct=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            sortino_ratio=Decimal("0"),
            calmar_ratio=Decimal("0"),
            num_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=Decimal("0"),
            avg_win=Decimal("0"),
            avg_loss=Decimal("0"),
            profit_factor=Decimal("0"),
            trading_days=0,
        )

    # Extract equity series
    equity_series = equity_curve["total_equity"]
    trading_days = len(equity_curve)

    # Calculate returns
    total_return = final_capital - initial_capital
    total_return_pct = (total_return / initial_capital) * Decimal("100")

    # Annualize return
    years = trading_days / 252
    if years > 0:
        annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * Decimal("100")
    else:
        annualized_return = Decimal("0")

    # Calculate return series
    returns = calculate_returns(equity_series)

    # Volatility (annualized)
    if len(returns) > 1:
        volatility = Decimal(str(returns.std() * np.sqrt(252) * 100))
    else:
        volatility = Decimal("0")

    # Max drawdown
    max_dd, max_dd_pct = calculate_max_drawdown(equity_series)

    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)

    # Sortino ratio
    sortino = calculate_sortino_ratio(returns, risk_free_rate)

    # Calmar ratio
    if max_dd_pct > 0:
        calmar = calculate_calmar_ratio(annualized_return, max_dd_pct)
    else:
        calmar = Decimal("999.99")

    # Trade statistics
    win_rate = Decimal(str(winning_trades / num_trades * 100)) if num_trades > 0 else Decimal("0")

    avg_win = total_wins / Decimal(winning_trades) if winning_trades > 0 else Decimal("0")
    avg_loss = total_losses / Decimal(losing_trades) if losing_trades > 0 else Decimal("0")

    # Profit factor
    if abs(total_losses) > 0:
        profit_factor = abs(total_wins / total_losses)
    else:
        profit_factor = Decimal("999.99") if total_wins > 0 else Decimal("0")

    return PerformanceMetrics(
        total_return=total_return,
        total_return_pct=total_return_pct,
        annualized_return=annualized_return,
        volatility=volatility,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        num_trades=num_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        trading_days=trading_days,
    )


def print_metrics(metrics: PerformanceMetrics) -> None:
    """
    Pretty-print performance metrics.

    Args:
        metrics: PerformanceMetrics to display
    """
    print("\n" + "=" * 60)
    print("BACKTEST PERFORMANCE METRICS")
    print("=" * 60)

    print("\nReturn Metrics:")
    print(f"  Total Return:        ${metrics.total_return:>12,.2f} ({metrics.total_return_pct:>6.2f}%)")
    print(f"  Annualized Return:   {metrics.annualized_return:>6.2f}%")

    print("\nRisk Metrics:")
    print(f"  Volatility (ann.):   {metrics.volatility:>6.2f}%")
    print(f"  Max Drawdown:        ${metrics.max_drawdown:>12,.2f} ({metrics.max_drawdown_pct:>6.2f}%)")

    print("\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:>6.2f}")
    print(f"  Sortino Ratio:       {metrics.sortino_ratio:>6.2f}")
    print(f"  Calmar Ratio:        {metrics.calmar_ratio:>6.2f}")

    print("\nTrade Statistics:")
    print(f"  Total Trades:        {metrics.num_trades:>6}")
    print(f"  Winning Trades:      {metrics.winning_trades:>6}")
    print(f"  Losing Trades:       {metrics.losing_trades:>6}")
    print(f"  Win Rate:            {metrics.win_rate:>6.2f}%")
    print(f"  Avg Win:             ${metrics.avg_win:>12,.2f}")
    print(f"  Avg Loss:            ${metrics.avg_loss:>12,.2f}")
    print(f"  Profit Factor:       {metrics.profit_factor:>6.2f}")

    print("\nTime Metrics:")
    print(f"  Trading Days:        {metrics.trading_days:>6}")

    print("=" * 60 + "\n")


@dataclass(frozen=True)
class RegimeConditionalMetrics:
    """
    Performance metrics split by market regime.

    Allows analyzing how strategy performs in different volatility regimes.
    """

    regime_id: int
    observations: int
    total_return: Decimal
    annualized_return: Decimal
    sharpe_ratio: Decimal
    volatility: Decimal
    max_drawdown_pct: Decimal
    num_trades: int
    win_rate: Decimal

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "regime_id": self.regime_id,
            "observations": self.observations,
            "total_return": float(self.total_return),
            "annualized_return": float(self.annualized_return),
            "sharpe_ratio": float(self.sharpe_ratio),
            "volatility": float(self.volatility),
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "num_trades": self.num_trades,
            "win_rate": float(self.win_rate),
        }


@dataclass(frozen=True)
class GreeksAttribution:
    """
    P&L attribution to individual Greeks.

    Decomposes portfolio returns into delta, gamma, vega, and theta contributions.
    """

    total_pnl: Decimal
    delta_pnl: Decimal
    gamma_pnl: Decimal
    vega_pnl: Decimal
    theta_pnl: Decimal
    other_pnl: Decimal  # Residual

    delta_pct: Decimal
    gamma_pct: Decimal
    vega_pct: Decimal
    theta_pct: Decimal

    def to_dict(self) -> dict:
        """Convert attribution to dictionary."""
        return {
            "total_pnl": float(self.total_pnl),
            "delta_pnl": float(self.delta_pnl),
            "gamma_pnl": float(self.gamma_pnl),
            "vega_pnl": float(self.vega_pnl),
            "theta_pnl": float(self.theta_pnl),
            "other_pnl": float(self.other_pnl),
            "delta_pct": float(self.delta_pct),
            "gamma_pct": float(self.gamma_pct),
            "vega_pct": float(self.vega_pct),
            "theta_pct": float(self.theta_pct),
        }


def calculate_regime_conditional_metrics(
    equity_curve: pd.DataFrame,
    regime_labels: pd.Series,
    trades_df: Optional[pd.DataFrame] = None,
    risk_free_rate: Decimal = Decimal("0.05"),
) -> list[RegimeConditionalMetrics]:
    """
    Calculate performance metrics conditional on market regime.

    Args:
        equity_curve: DataFrame with equity over time (must have datetime index)
        regime_labels: Series of regime labels (same index as equity_curve)
        trades_df: Optional DataFrame of trades with regime information
        risk_free_rate: Annual risk-free rate

    Returns:
        List of RegimeConditionalMetrics, one per regime
    """
    if equity_curve.empty or regime_labels.empty:
        logger.warning("Empty data for regime conditional metrics")
        return []

    # Align regime labels with equity curve
    aligned_regimes = regime_labels.reindex(equity_curve.index, method="ffill")

    # Calculate returns
    equity_series = equity_curve["total_equity"]
    returns = calculate_returns(equity_series)

    results = []

    for regime_id in sorted(aligned_regimes.unique()):
        if pd.isna(regime_id):
            continue

        regime_id_int = int(regime_id)

        # Filter data for this regime
        regime_mask = aligned_regimes == regime_id
        regime_returns = returns[regime_mask]
        regime_equity = equity_series[regime_mask]

        if len(regime_returns) < 2:
            continue

        # Calculate metrics
        total_return = regime_equity.iloc[-1] - regime_equity.iloc[0]
        years = len(regime_returns) / 252
        annualized_return = (
            ((regime_equity.iloc[-1] / regime_equity.iloc[0]) ** (1 / years) - 1) * Decimal("100")
            if years > 0 and regime_equity.iloc[0] > 0
            else Decimal("0")
        )

        sharpe = calculate_sharpe_ratio(regime_returns, risk_free_rate)
        volatility = Decimal(str(regime_returns.std() * np.sqrt(252) * 100)) if len(regime_returns) > 1 else Decimal("0")

        _, max_dd_pct = calculate_max_drawdown(regime_equity)

        # Trade statistics
        num_trades = 0
        wins = 0
        if trades_df is not None and "regime" in trades_df.columns:
            regime_trades = trades_df[trades_df["regime"] == regime_id_int]
            num_trades = len(regime_trades)
            if "pnl" in regime_trades.columns:
                wins = len(regime_trades[regime_trades["pnl"] > 0])

        win_rate = Decimal(str(wins / num_trades * 100)) if num_trades > 0 else Decimal("0")

        metrics = RegimeConditionalMetrics(
            regime_id=regime_id_int,
            observations=len(regime_returns),
            total_return=Decimal(str(total_return)),
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            volatility=volatility,
            max_drawdown_pct=max_dd_pct,
            num_trades=num_trades,
            win_rate=win_rate,
        )

        results.append(metrics)

    return results


def calculate_greeks_attribution(
    portfolio_history: pd.DataFrame,
) -> GreeksAttribution:
    """
    Calculate P&L attribution to individual Greeks.

    Approximates daily P&L contributions from delta, gamma, vega, and theta.

    Args:
        portfolio_history: DataFrame with columns:
            - total_equity: Portfolio value
            - total_delta: Portfolio delta
            - total_gamma: Portfolio gamma
            - total_vega: Portfolio vega
            - total_theta: Portfolio theta
            - underlying_price: Underlying price (for delta/gamma calculation)
            - implied_volatility: Implied volatility (for vega calculation)

    Returns:
        GreeksAttribution with P&L decomposition
    """
    if portfolio_history.empty or len(portfolio_history) < 2:
        logger.warning("Insufficient data for Greeks attribution")
        return GreeksAttribution(
            total_pnl=Decimal("0"),
            delta_pnl=Decimal("0"),
            gamma_pnl=Decimal("0"),
            vega_pnl=Decimal("0"),
            theta_pnl=Decimal("0"),
            other_pnl=Decimal("0"),
            delta_pct=Decimal("0"),
            gamma_pct=Decimal("0"),
            vega_pct=Decimal("0"),
            theta_pct=Decimal("0"),
        )

    # Calculate total P&L
    total_pnl = portfolio_history["total_equity"].iloc[-1] - portfolio_history["total_equity"].iloc[0]

    # Calculate daily changes
    price_change = portfolio_history["underlying_price"].diff().fillna(0) if "underlying_price" in portfolio_history.columns else pd.Series(0, index=portfolio_history.index)
    vol_change = portfolio_history["implied_volatility"].diff().fillna(0) if "implied_volatility" in portfolio_history.columns else pd.Series(0, index=portfolio_history.index)

    # Calculate P&L contributions
    # Delta P&L: Delta * ΔS
    if "total_delta" in portfolio_history.columns:
        delta_contribution = (portfolio_history["total_delta"].shift(1) * price_change).sum()
    else:
        delta_contribution = 0

    # Gamma P&L: 0.5 * Gamma * (ΔS)²
    if "total_gamma" in portfolio_history.columns:
        gamma_contribution = (0.5 * portfolio_history["total_gamma"].shift(1) * price_change ** 2).sum()
    else:
        gamma_contribution = 0

    # Vega P&L: Vega * Δσ
    if "total_vega" in portfolio_history.columns:
        vega_contribution = (portfolio_history["total_vega"].shift(1) * vol_change).sum()
    else:
        vega_contribution = 0

    # Theta P&L: Theta * Δt (assuming 1 day)
    if "total_theta" in portfolio_history.columns:
        theta_contribution = portfolio_history["total_theta"].sum()
    else:
        theta_contribution = 0

    # Convert to Decimal
    delta_pnl = Decimal(str(delta_contribution))
    gamma_pnl = Decimal(str(gamma_contribution))
    vega_pnl = Decimal(str(vega_contribution))
    theta_pnl = Decimal(str(theta_contribution))

    # Residual (unexplained P&L)
    explained_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    other_pnl = Decimal(str(total_pnl)) - explained_pnl

    # Calculate percentages
    total_abs = abs(Decimal(str(total_pnl)))
    if total_abs > 0:
        delta_pct = (delta_pnl / total_abs) * Decimal("100")
        gamma_pct = (gamma_pnl / total_abs) * Decimal("100")
        vega_pct = (vega_pnl / total_abs) * Decimal("100")
        theta_pct = (theta_pnl / total_abs) * Decimal("100")
    else:
        delta_pct = gamma_pct = vega_pct = theta_pct = Decimal("0")

    return GreeksAttribution(
        total_pnl=Decimal(str(total_pnl)),
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        other_pnl=other_pnl,
        delta_pct=delta_pct,
        gamma_pct=gamma_pct,
        vega_pct=vega_pct,
        theta_pct=theta_pct,
    )
