"""
Delta hedger for options portfolios.

Manages delta-neutral hedging with configurable rebalancing.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from volatility_arbitrage.execution.costs import SquareRootImpactModel, TransactionCostModel
from volatility_arbitrage.models.black_scholes import BlackScholesModel, Greeks, OptionType

from .attribution import calculate_total_attribution
from .types import HedgeConfig, HedgeState, PnLAttribution, RebalanceFrequency


class DeltaHedger:
    """
    Delta hedger for options portfolios.

    Maintains a delta-neutral position by holding stock against options.
    Tracks P&L attribution to demonstrate that returns come from Vega/Gamma
    rather than directional exposure.
    """

    def __init__(
        self,
        config: HedgeConfig,
        risk_free_rate: Decimal = Decimal("0.05"),
    ):
        """
        Initialize the delta hedger.

        Args:
            config: Hedge configuration
            risk_free_rate: Annual risk-free rate
        """
        self.config = config
        self.risk_free_rate = risk_free_rate

        # Use default cost model if not provided
        if config.cost_model is None:
            self.cost_model: TransactionCostModel = SquareRootImpactModel()
        else:
            self.cost_model = config.cost_model

        # State
        self.state: Optional[HedgeState] = None
        self.attribution_history: list[PnLAttribution] = []
        self._last_rebalance: Optional[datetime] = None
        self._rebalance_count: int = 0

        # Option parameters (set on initialize)
        self._strike: Optional[Decimal] = None
        self._expiry: Optional[datetime] = None
        self._option_type: Optional[OptionType] = None

    def initialize(
        self,
        timestamp: datetime,
        spot: Decimal,
        iv: Decimal,
        option_position: Decimal,
        strike: Decimal,
        expiry: datetime,
        option_type: OptionType = OptionType.CALL,
    ) -> HedgeState:
        """
        Initialize the hedger with an option position.

        Sets up the initial delta hedge.

        Args:
            timestamp: Current timestamp
            spot: Spot price
            iv: Implied volatility
            option_position: Number of option contracts (positive = long)
            strike: Option strike price
            expiry: Option expiry datetime
            option_type: CALL or PUT

        Returns:
            Initial HedgeState
        """
        self._strike = strike
        self._expiry = expiry
        self._option_type = option_type
        self._last_rebalance = timestamp

        # Calculate time to expiry
        T = self._time_to_expiry(timestamp)

        # Calculate option price and Greeks
        option_price = BlackScholesModel.price(
            S=spot, K=strike, T=T, r=self.risk_free_rate, sigma=iv, option_type=option_type
        )
        greeks = BlackScholesModel.greeks(
            S=spot, K=strike, T=T, r=self.risk_free_rate, sigma=iv, option_type=option_type
        )

        # Scale by position size and multiplier
        position_multiplier = option_position * Decimal(self.config.option_multiplier)

        # Initial hedge: short delta shares to neutralize
        hedge_shares = -position_multiplier * greeks.delta

        self.state = HedgeState(
            timestamp=timestamp,
            spot=spot,
            iv=iv,
            option_position=position_multiplier,
            option_price=option_price,
            hedge_shares=hedge_shares,
            portfolio_delta=Decimal("0"),  # Exactly hedged at start
            portfolio_gamma=position_multiplier * greeks.gamma,
            portfolio_vega=position_multiplier * greeks.vega,
            portfolio_theta=position_multiplier * greeks.theta,
        )

        return self.state

    def update(
        self,
        timestamp: datetime,
        spot: Decimal,
        iv: Decimal,
    ) -> tuple[HedgeState, PnLAttribution, Decimal]:
        """
        Update state with new market data.

        Calculates P&L attribution and rebalances if needed.

        Args:
            timestamp: Current timestamp
            spot: Current spot price
            iv: Current implied volatility

        Returns:
            Tuple of (new_state, attribution, transaction_cost)

        Raises:
            RuntimeError: If hedger not initialized
        """
        if self.state is None:
            raise RuntimeError("Hedger not initialized. Call initialize() first.")

        # Calculate time to expiry
        T = self._time_to_expiry(timestamp)

        # Handle expired options
        if T <= Decimal("0"):
            return self._handle_expiry(timestamp, spot)

        # Calculate new option price and Greeks
        option_price = BlackScholesModel.price(
            S=spot,
            K=self._strike,
            T=T,
            r=self.risk_free_rate,
            sigma=iv,
            option_type=self._option_type,
        )
        greeks = BlackScholesModel.greeks(
            S=spot,
            K=self._strike,
            T=T,
            r=self.risk_free_rate,
            sigma=iv,
            option_type=self._option_type,
        )

        # Current portfolio delta before rebalancing
        # option_position is already scaled by multiplier, so no division needed
        current_delta = self.state.option_position * greeks.delta + self.state.hedge_shares

        # Check if rebalance needed
        transaction_cost = Decimal("0")
        new_hedge_shares = self.state.hedge_shares
        rebalanced = False

        if self._should_rebalance(timestamp, current_delta):
            # Target hedge to make delta = 0
            target_hedge = -self.state.option_position * greeks.delta
            trade_qty = abs(target_hedge - self.state.hedge_shares)

            # Calculate transaction cost
            if trade_qty > 0:
                transaction_cost = Decimal(
                    str(
                        self.cost_model.calculate_cost(
                            order_size=float(trade_qty),
                            price=float(spot),
                            volatility=float(iv),
                            daily_volume=self.config.daily_volume,
                        )
                    )
                )

            new_hedge_shares = target_hedge
            self._last_rebalance = timestamp
            self._rebalance_count += 1
            rebalanced = True

        # Calculate new portfolio delta after rebalancing
        # option_position is already scaled by multiplier
        new_portfolio_delta = self.state.option_position * greeks.delta + new_hedge_shares

        # Build new state
        new_state = HedgeState(
            timestamp=timestamp,
            spot=spot,
            iv=iv,
            option_position=self.state.option_position,
            option_price=option_price,
            hedge_shares=new_hedge_shares,
            portfolio_delta=new_portfolio_delta,
            portfolio_gamma=self.state.option_position * greeks.gamma,
            portfolio_vega=self.state.option_position * greeks.vega,
            portfolio_theta=self.state.option_position * greeks.theta,
        )

        # Calculate P&L attribution
        attribution = calculate_total_attribution(
            self.state, new_state, transaction_cost, rebalanced
        )
        self.attribution_history.append(attribution)

        # Update state
        self.state = new_state

        return new_state, attribution, transaction_cost

    def _should_rebalance(self, timestamp: datetime, delta: Decimal) -> bool:
        """
        Check if a rebalance is needed.

        Rebalances based on either:
        1. Delta exceeds threshold, OR
        2. Time-based frequency

        Args:
            timestamp: Current timestamp
            delta: Current portfolio delta

        Returns:
            True if should rebalance
        """
        # Threshold-based: rebalance if delta too large
        if abs(delta) > self.config.delta_threshold:
            return True

        # Time-based check
        if self._last_rebalance is None:
            return True

        elapsed = timestamp - self._last_rebalance

        if self.config.rebalance_frequency == RebalanceFrequency.CONTINUOUS:
            return True
        elif self.config.rebalance_frequency == RebalanceFrequency.HOURLY:
            return elapsed >= timedelta(hours=1)
        elif self.config.rebalance_frequency == RebalanceFrequency.FOUR_HOUR:
            return elapsed >= timedelta(hours=4)
        elif self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            return elapsed >= timedelta(days=1)

        return False

    def _time_to_expiry(self, timestamp: datetime) -> Decimal:
        """Calculate time to expiry in years."""
        if self._expiry is None:
            return Decimal("0")

        seconds_to_expiry = (self._expiry - timestamp).total_seconds()
        years = seconds_to_expiry / (365.25 * 24 * 3600)
        return Decimal(str(max(0, years)))

    def _handle_expiry(
        self, timestamp: datetime, spot: Decimal
    ) -> tuple[HedgeState, PnLAttribution, Decimal]:
        """Handle option expiry."""
        # Option expires worthless or ITM
        if self._option_type == OptionType.CALL:
            intrinsic = max(Decimal("0"), spot - self._strike)
        else:
            intrinsic = max(Decimal("0"), self._strike - spot)

        # Final settlement
        new_state = HedgeState(
            timestamp=timestamp,
            spot=spot,
            iv=Decimal("0"),
            option_position=self.state.option_position,
            option_price=intrinsic,
            hedge_shares=Decimal("0"),
            portfolio_delta=Decimal("0"),
            portfolio_gamma=Decimal("0"),
            portfolio_vega=Decimal("0"),
            portfolio_theta=Decimal("0"),
        )

        attribution = calculate_total_attribution(self.state, new_state, Decimal("0"), True)
        self.attribution_history.append(attribution)
        self.state = new_state

        return new_state, attribution, Decimal("0")

    @property
    def rebalance_count(self) -> int:
        """Number of rebalances executed."""
        return self._rebalance_count

    def reset(self) -> None:
        """Reset the hedger state."""
        self.state = None
        self.attribution_history.clear()
        self._last_rebalance = None
        self._rebalance_count = 0
        self._strike = None
        self._expiry = None
        self._option_type = None
