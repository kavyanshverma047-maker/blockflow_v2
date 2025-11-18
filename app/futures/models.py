# app/futures/models.py
from sqlalchemy import (
    Column, Integer, String, Numeric, Enum as SQLEnum,
    Boolean, DateTime, ForeignKey, Text, func, Index
)
from sqlalchemy.orm import relationship
from enum import Enum
from decimal import Decimal
from app.database import Base


# -------------------------
# ENUMS
# -------------------------
class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class FuturesOrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class FuturesOrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class FuturesOrderStatus(str, Enum):
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"


class FuturesLedgerType(str, Enum):
    MARGIN_RELEASED = "margin_released"
    REALIZED_PNL = "realized_pnl"
    LIQUIDATION_FEE = "liquidation_fee"
    TRADING_FEE = "trading_fee"


# -------------------------
# FUTURES ORDERS
# -------------------------
class FuturesOrder(Base):
    __tablename__ = "futures_orders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)

    side = Column(SQLEnum(FuturesOrderSide), nullable=False)
    order_type = Column(SQLEnum(FuturesOrderType), nullable=False)
    status = Column(SQLEnum(FuturesOrderStatus), nullable=False, default=FuturesOrderStatus.OPEN)

    price = Column(Numeric(30, 10), nullable=True)
    quantity = Column(Numeric(30, 10), nullable=False)
    filled_quantity = Column(Numeric(30, 10), default=Decimal("0"))

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # optional relation placeholders (if you want)
    # trades = relationship("FuturesTrade", back_populates="order")


# -------------------------
# FUTURES TRADES
# -------------------------
class FuturesTrade(Base):
    __tablename__ = "futures_trades"

    id = Column(Integer, primary_key=True, index=True)
    buy_order_id = Column(Integer, ForeignKey("futures_orders.id"), nullable=True)
    sell_order_id = Column(Integer, ForeignKey("futures_orders.id"), nullable=True)

    symbol = Column(String, nullable=False, index=True)
    price = Column(Numeric(30, 10), nullable=False)
    quantity = Column(Numeric(30, 10), nullable=False)

    buyer_id = Column(Integer, nullable=True)
    seller_id = Column(Integer, nullable=True)

    buyer_fee = Column(Numeric(30, 10), default=Decimal("0"))
    seller_fee = Column(Numeric(30, 10), default=Decimal("0"))

    tds_amount_inr = Column(String, nullable=True)

    executed_at = Column(DateTime, server_default=func.now())


# -------------------------
# FUTURES POSITIONS
# -------------------------
class FuturesPosition(Base):
    __tablename__ = "futures_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)

    symbol = Column(String, nullable=False, index=True)
    side = Column(SQLEnum(PositionSide), nullable=False)

    leverage = Column(Integer, default=1)
    entry_price = Column(Numeric(30, 10), nullable=False)
    quantity = Column(Numeric(30, 10), nullable=False)

    # margins/fees
    initial_margin = Column(Numeric(30, 10), default=Decimal("0"))
    maintenance_margin = Column(Numeric(30, 10), default=Decimal("0"))
    funding_fees_paid = Column(Numeric(30, 10), default=Decimal("0"))
    open_fee = Column(Numeric(30, 10), default=Decimal("0"))
    close_fee = Column(Numeric(30, 10), default=Decimal("0"))

    # PNL
    unrealized_pnl = Column(Numeric(30, 10), default=Decimal("0"))
    realized_pnl = Column(Numeric(30, 10), default=Decimal("0"))

    # TP/SL fields (stored as strings to avoid Decimal issues when null)
    take_profit_price = Column(String, nullable=True)
    stop_loss_price = Column(String, nullable=True)

    liquidation_price = Column(Numeric(30, 10), nullable=True)

    status = Column(SQLEnum(PositionStatus), default=PositionStatus.OPEN)
    is_demo = Column(Boolean, default=True)

    created_at = Column(DateTime, server_default=func.now())
    closed_at = Column(DateTime, nullable=True)


# -------------------------
# FUTURES LIQUIDATIONS
# -------------------------
class FuturesLiquidation(Base):
    __tablename__ = "futures_liquidations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    position_id = Column(Integer, nullable=False, index=True)

    symbol = Column(String, nullable=False)
    side = Column(String)

    entry_price = Column(String)
    liquidation_price = Column(String)
    quantity = Column(String)

    loss_amount = Column(String)
    liquidation_fee = Column(String)
    trigger_price = Column(String)

    is_demo = Column(Boolean, default=True)
    liquidated_at = Column(DateTime, server_default=func.now())


# -------------------------
# FUTURES LEDGER
# -------------------------
class FuturesLedger(Base):
    __tablename__ = "futures_ledger"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    position_id = Column(Integer, nullable=False, index=True)

    entry_type = Column(SQLEnum(FuturesLedgerType), nullable=False)

    asset = Column(String, default="USDT")
    amount = Column(Numeric(30, 10))
    balance_after = Column(Numeric(30, 10))

    related_price = Column(Numeric(30, 10))

    # metadata is a reserved attr name in Declarative; expose DB column as "metadata" but Python attr as details
    details = Column("metadata", Text)

    created_at = Column(DateTime, server_default=func.now())

# -------------------------
# FUTURES WAL (Write-Ahead Log)
# -------------------------
class FuturesWAL(Base):
    __tablename__ = "futures_wal"

    id = Column(Integer, primary_key=True, index=True)

    idempotency_key = Column(String, unique=True, nullable=False, index=True)
    operation = Column(String, nullable=False)
    payload = Column(Text, nullable=False)

    status = Column(String, default="PENDING")  # PENDING | DONE | FAILED
    retry_count = Column(Integer, default=0)

    created_at = Column(DateTime, server_default=func.now())

# Indexes helpers (optional)
Index("ix_futures_positions_user_symbol", FuturesPosition.user_id, FuturesPosition.symbol)
Index("ix_futures_orders_user_symbol", FuturesOrder.user_id, FuturesOrder.symbol)
Index("ix_futures_trades_symbol", FuturesTrade.symbol)
