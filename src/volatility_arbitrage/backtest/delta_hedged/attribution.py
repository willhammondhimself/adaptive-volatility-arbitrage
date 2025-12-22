"""
P&L attribution using Taylor expansion.

Decomposes option portfolio returns into Greek components:
- Delta P&L: First-order sensitivity to spot
- Gamma P&L: Second-order sensitivity (convexity)
- Vega P&L: Sensitivity to implied volatility
- Theta P&L: Time decay

For a delta-hedged portfolio, Delta P&L should be near zero,
and alpha comes from Gamma + Vega.
"""

from datetime import datetime
from decimal import Decimal

from .types import HedgeState, PnLAttribution


def calculate_delta_pnl(delta: Decimal, spot_change: Decimal) -> Decimal:
    """
    Calculate P&L from delta exposure.

    Formula: Δ_pnl = Δ · ΔS

    For a hedged portfolio, delta should be ~0, so delta_pnl ~= 0.

    Args:
        delta: Portfolio delta (net of option and hedge)
        spot_change: Change in spot price

    Returns:
        P&L from delta exposure
    """
    return delta * spot_change


def calculate_gamma_pnl(
    gamma: Decimal,
    spot: Decimal,
    spot_return: Decimal,
) -> Decimal:
    """
    Calculate P&L from gamma (convexity).

    Formula: Γ_pnl = ½ · Γ · S² · r²

    Where r = ΔS/S is the return. This captures the "long volatility"
    profit from price moves in either direction.

    Args:
        gamma: Portfolio gamma
        spot: Spot price at start of period
        spot_return: Spot return (ΔS/S)

    Returns:
        P&L from gamma (always positive for long gamma)
    """
    return Decimal("0.5") * gamma * (spot ** 2) * (spot_return ** 2)


def calculate_vega_pnl(vega: Decimal, iv_change: Decimal) -> Decimal:
    """
    Calculate P&L from vega exposure.

    Formula: ν_pnl = ν · Δσ

    Where Δσ is the change in implied volatility. Note that vega
    in our model is per 1% change in vol, so iv_change should be
    in percentage points (e.g., 0.01 = 1% increase).

    Args:
        vega: Portfolio vega (sensitivity per 1% vol change)
        iv_change: Change in implied volatility (absolute, e.g., 0.20 → 0.22 is 0.02)

    Returns:
        P&L from IV changes
    """
    # Convert iv_change to percentage points for vega
    # If IV goes from 0.20 to 0.22, that's +2 percentage points
    iv_change_pct = iv_change * Decimal("100")
    return vega * iv_change_pct


def calculate_theta_pnl(theta: Decimal, dt_days: Decimal) -> Decimal:
    """
    Calculate P&L from theta (time decay).

    Formula: θ_pnl = θ · Δt

    Where theta is per day and dt is in days. For long options,
    theta is negative (time decay).

    Args:
        theta: Portfolio theta (P&L per day)
        dt_days: Time elapsed in days

    Returns:
        P&L from time decay (typically negative for long options)
    """
    return theta * dt_days


def calculate_total_attribution(
    prev_state: HedgeState,
    curr_state: HedgeState,
    transaction_cost: Decimal,
    rebalanced: bool,
) -> PnLAttribution:
    """
    Calculate full P&L attribution between two states.

    Uses Taylor expansion to decompose P&L into:
    dV = Δ·dS + ½Γ·(dS)² + ν·dσ + θ·dt

    The actual P&L is calculated from mark-to-market option value changes
    plus the hedge P&L. The residual captures higher-order terms and
    cross-Greeks.

    Args:
        prev_state: Portfolio state at start of period
        curr_state: Portfolio state at end of period
        transaction_cost: Cost of any rebalancing trades
        rebalanced: Whether the hedge was rebalanced

    Returns:
        PnLAttribution with all components
    """
    # Calculate changes
    spot_change = curr_state.spot - prev_state.spot
    spot_return = spot_change / prev_state.spot if prev_state.spot != 0 else Decimal("0")
    iv_change = curr_state.iv - prev_state.iv
    dt_seconds = (curr_state.timestamp - prev_state.timestamp).total_seconds()
    dt_days = Decimal(str(dt_seconds / 86400))

    # Calculate attributed P&L components using Greeks at start of period
    delta_pnl = calculate_delta_pnl(prev_state.portfolio_delta, spot_change)
    gamma_pnl = calculate_gamma_pnl(prev_state.portfolio_gamma, prev_state.spot, spot_return)
    vega_pnl = calculate_vega_pnl(prev_state.portfolio_vega, iv_change)
    theta_pnl = calculate_theta_pnl(prev_state.portfolio_theta, dt_days)

    # Calculate actual P&L from mark-to-market
    # Option P&L (with multiplier factored into position)
    option_pnl = prev_state.option_position * (curr_state.option_price - prev_state.option_price)

    # Hedge P&L (short position gains when spot falls)
    hedge_pnl = prev_state.hedge_shares * spot_change

    # Total actual P&L (before transaction costs)
    actual_pnl = option_pnl + hedge_pnl

    # Attributed P&L (what Taylor expansion predicts)
    attributed_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

    # Residual is the difference (higher-order terms, cross-Greeks)
    residual = actual_pnl - attributed_pnl

    # Total P&L after transaction costs
    total_pnl = actual_pnl - transaction_cost

    return PnLAttribution(
        timestamp=curr_state.timestamp,
        total_pnl=total_pnl,
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        transaction_costs=transaction_cost,
        residual=residual,
        rebalanced=rebalanced,
    )
