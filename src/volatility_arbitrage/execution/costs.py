"""
Transaction cost models for execution simulation.

Implements market impact and spread costs for realistic backtesting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models."""

    @abstractmethod
    def calculate_cost(
        self,
        order_size: float,
        price: float,
        volatility: float,
        daily_volume: float,
    ) -> float:
        """
        Calculate total transaction cost for an order.

        Args:
            order_size: Number of shares/contracts
            price: Current price per unit
            volatility: Annualized volatility (e.g., 0.20 = 20%)
            daily_volume: Average daily volume in shares/contracts

        Returns:
            Total cost in currency units (always positive)
        """
        pass


@dataclass
class SquareRootImpactModel(TransactionCostModel):
    """
    Square Root Law of Market Impact.

    Cost = spread_cost + impact_cost

    Where:
        spread_cost = order_value * half_spread_bps / 10000
        impact_cost = order_value * impact_coeff * volatility * sqrt(order_size / daily_volume)

    The square-root relationship reflects the empirical observation that
    market impact grows sub-linearly with order size.

    Reference: Almgren & Chriss (2000), Kyle (1985)
    """

    half_spread_bps: float = 5.0
    """Half the bid-ask spread in basis points. Default 5 bps for liquid equities."""

    impact_coeff: float = 0.1
    """
    Impact coefficient calibrated such that:
    - Order = 1% of ADV
    - Volatility = 20%
    - Results in ~10 bps market impact
    """

    def calculate_cost(
        self,
        order_size: float,
        price: float,
        volatility: float,
        daily_volume: float,
    ) -> float:
        """
        Calculate total transaction cost.

        Args:
            order_size: Number of shares (absolute value used)
            price: Price per share
            volatility: Annualized volatility (decimal, e.g., 0.20)
            daily_volume: Average daily trading volume

        Returns:
            Total cost in currency units
        """
        order_size = abs(order_size)
        if order_size == 0:
            return 0.0

        order_value = order_size * price

        # Spread cost: fixed cost per unit traded
        spread_cost = order_value * (self.half_spread_bps / 10_000)

        # Market impact: scales with sqrt of participation rate
        if daily_volume > 0:
            participation_rate = order_size / daily_volume
            impact_bps = self.impact_coeff * volatility * math.sqrt(participation_rate)
            impact_cost = order_value * impact_bps
        else:
            # No volume data: use spread only
            impact_cost = 0.0

        return spread_cost + impact_cost

    def estimate_impact_bps(
        self,
        order_size: float,
        volatility: float,
        daily_volume: float,
    ) -> float:
        """
        Estimate market impact in basis points (excluding spread).

        Useful for pre-trade cost estimation.

        Args:
            order_size: Number of shares
            volatility: Annualized volatility
            daily_volume: Average daily volume

        Returns:
            Estimated impact in basis points
        """
        if daily_volume <= 0 or order_size == 0:
            return 0.0

        participation_rate = abs(order_size) / daily_volume
        return self.impact_coeff * volatility * math.sqrt(participation_rate) * 10_000
