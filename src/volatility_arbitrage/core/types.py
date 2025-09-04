"""
Core data types for the volatility arbitrage backtesting engine.

All types use Pydantic for validation and are immutable for safe functional transformations.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class OptionType(str, Enum):
    """Option contract type."""

    CALL = "call"
    PUT = "put"


class TradeType(str, Enum):
    """Trade direction."""

    BUY = "buy"
    SELL = "sell"


class TickData(BaseModel):
    """
    Market tick data for underlying asset.

    Represents a single point-in-time snapshot of market data.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    timestamp: datetime = Field(..., description="Time of the tick")
    symbol: str = Field(..., description="Asset symbol", min_length=1)
    price: Decimal = Field(..., description="Asset price", gt=0)
    volume: int = Field(..., description="Trading volume", ge=0)
    bid: Optional[Decimal] = Field(None, description="Bid price", gt=0)
    ask: Optional[Decimal] = Field(None, description="Ask price", gt=0)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    @field_validator("ask")
    @classmethod
    def validate_ask_bid_spread(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        """Ensure ask >= bid if both are present."""
        if v is not None and info.data.get("bid") is not None:
            if v < info.data["bid"]:
                raise ValueError("Ask price must be >= bid price")
        return v

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price from bid/ask or use last price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / Decimal("2")
        return self.price

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


class OptionContract(BaseModel):
    """
    Option contract specification.

    Represents a single option contract with its characteristics.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    symbol: str = Field(..., description="Underlying symbol", min_length=1)
    option_type: OptionType = Field(..., description="Call or Put")
    strike: Decimal = Field(..., description="Strike price", gt=0)
    expiry: datetime = Field(..., description="Expiration date")
    price: Decimal = Field(..., description="Option price", ge=0)
    bid: Optional[Decimal] = Field(None, description="Bid price", ge=0)
    ask: Optional[Decimal] = Field(None, description="Ask price", ge=0)
    volume: int = Field(default=0, description="Trading volume", ge=0)
    open_interest: int = Field(default=0, description="Open interest", ge=0)
    implied_volatility: Optional[Decimal] = Field(
        None, description="Implied volatility", ge=0, le=10
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price from bid/ask or use last price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / Decimal("2")
        return self.price

    @property
    def moneyness(self) -> str:
        """
        Calculate moneyness relative to underlying price.

        Note: Requires underlying price to be meaningful.
        """
        # This is a placeholder - actual moneyness requires underlying price
        return "ATM"  # At-The-Money placeholder


class OptionChain(BaseModel):
    """
    Complete option chain for a given underlying and expiration.

    Contains all calls and puts for a specific expiry date.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Underlying symbol", min_length=1)
    timestamp: datetime = Field(..., description="Chain snapshot time")
    expiry: datetime = Field(..., description="Options expiration date")
    underlying_price: Decimal = Field(..., description="Current underlying price", gt=0)
    calls: list[OptionContract] = Field(default_factory=list, description="Call options")
    puts: list[OptionContract] = Field(default_factory=list, description="Put options")
    risk_free_rate: Decimal = Field(
        default=Decimal("0.05"), description="Risk-free rate", ge=0, le=1
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    @field_validator("expiry")
    @classmethod
    def validate_expiry(cls, v: datetime, info) -> datetime:
        """Ensure expiry is in the future."""
        if "timestamp" in info.data and v <= info.data["timestamp"]:
            raise ValueError("Expiry must be in the future")
        return v

    @property
    def time_to_expiry(self) -> Decimal:
        """Calculate time to expiration in years."""
        delta = self.expiry - self.timestamp
        return Decimal(delta.total_seconds()) / Decimal(365.25 * 24 * 3600)

    def get_atm_strike(self) -> Optional[Decimal]:
        """Find the strike closest to current underlying price."""
        all_strikes = [opt.strike for opt in self.calls + self.puts]
        if not all_strikes:
            return None
        return min(all_strikes, key=lambda x: abs(x - self.underlying_price))


class Trade(BaseModel):
    """
    Executed trade record.

    Immutable record of a single trade execution.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(..., description="Execution time")
    symbol: str = Field(..., description="Instrument symbol", min_length=1)
    trade_type: TradeType = Field(..., description="Buy or Sell")
    quantity: int = Field(..., description="Number of contracts/shares", gt=0)
    price: Decimal = Field(..., description="Execution price", ge=0)
    commission: Decimal = Field(default=Decimal("0"), description="Trading commission", ge=0)
    trade_id: Optional[str] = Field(None, description="Unique trade identifier")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    @property
    def total_cost(self) -> Decimal:
        """Calculate total trade cost including commission."""
        base_cost = self.price * Decimal(self.quantity)
        if self.trade_type == TradeType.BUY:
            return base_cost + self.commission
        else:  # SELL
            return base_cost - self.commission

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value (before commission)."""
        return self.price * Decimal(self.quantity)


class Position(BaseModel):
    """
    Current position in an instrument.

    Tracks quantity, entry price, and unrealized P&L.
    """

    model_config = ConfigDict(frozen=False)  # Mutable for position updates

    symbol: str = Field(..., description="Instrument symbol", min_length=1)
    quantity: int = Field(..., description="Current position size (+ long, - short)")
    avg_entry_price: Decimal = Field(..., description="Average entry price", ge=0)
    current_price: Decimal = Field(..., description="Current market price", ge=0)
    last_update: datetime = Field(..., description="Last update timestamp")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of position."""
        return Decimal(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> Decimal:
        """Calculate original cost basis."""
        return Decimal(abs(self.quantity)) * self.avg_entry_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized profit/loss."""
        if self.quantity > 0:  # Long position
            return (self.current_price - self.avg_entry_price) * Decimal(self.quantity)
        elif self.quantity < 0:  # Short position
            return (self.avg_entry_price - self.current_price) * Decimal(abs(self.quantity))
        return Decimal("0")

    @property
    def unrealized_pnl_pct(self) -> Decimal:
        """Calculate unrealized P&L as percentage of cost basis."""
        if self.cost_basis == 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.cost_basis) * Decimal("100")

    def update_price(self, new_price: Decimal, timestamp: datetime) -> None:
        """Update current price and timestamp."""
        self.current_price = new_price
        self.last_update = timestamp
