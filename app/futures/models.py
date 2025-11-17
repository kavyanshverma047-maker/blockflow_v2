"""
Futures Trading Models - Database Schema
========================================
SQLAlchemy models for USDM Futures trading system.
"""

from sqlalchemy import Column, Integer, String, Numeric, Boolean, DateTime, Enum as SQLEnum, Index, ForeignKey, Text
from datetime import datetime, timezone
from enum import Enum
from app.db.models import Base

# Enums
class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"

class FuturesOrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

class FuturesOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class FuturesOrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TRIGGERED = "TRIGGERED"

class FuturesLedgerType(str, Enum):
    MARGIN_LOCKED = "MARGIN_LOCKED"
    MARGIN_RELEASED = "MARGIN_RELEASED"
    REALIZED_PNL = "REALIZED_PNL"
    FUNDING_FEE = "FUNDING_FEE"
    TRADING_FEE = "TRADING_FEE"
    LIQUIDATION_FEE = "LIQUIDATION_FEE"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"

# Models
class FuturesPosition(Base):
    """Open futures position"""
    __tablename__ = "futures_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)  # BTCUSDT, ETHUSDT
    side = Column(SQLEnum(PositionSide), nullable=False)
    status = Column(SQLEnum(PositionStatus), nullable=False, default=PositionStatus.OPEN, index=True)
    
    # Position details
    entry_price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)  # Number of contracts
    leverage = Column(Integer, nullable=False)  # 1-125x
    
    # Margin
    initial_margin = Column(Numeric(20, 8), nullable=False)  # Locked collateral
    maintenance_margin = Column(Numeric(20, 8), nullable=False)  # Min margin to avoid liquidation
    
    # PnL tracking
    unrealized_pnl = Column(Numeric(20, 8), nullable=False, default=0)
    realized_pnl = Column(Numeric(20, 8), nullable=False, default=0)
    
    # Risk management
    liquidation_price = Column(Numeric(20, 8), nullable=False)
    take_profit_price = Column(Numeric(20, 8), nullable=True)
    stop_loss_price = Column(Numeric(20, 8), nullable=True)
    
    # Fees
    open_fee = Column(Numeric(20, 8), nullable=False, default=0)
    close_fee = Column(Numeric(20, 8), nullable=False, default=0)
    funding_fees_paid = Column(Numeric(20, 8), nullable=False, default=0)
    
    # Metadata / flags
    is_reduce_only = Column(Boolean, default=False)
    is_demo = Column(Boolean, default=False, index=True)
    
    # Timestamps
    opened_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    closed_at = Column(DateTime(timezone=True), nullable=True)
    last_updated = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index('idx_user_status', 'user_id', 'status'),
        Index('idx_symbol_status', 'symbol', 'status'),
        Index('idx_demo_status', 'is_demo', 'status'),
    )

class FuturesOrder(Base):
    """Futures order (including TP/SL orders)"""
    __tablename__ = "futures_orders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    position_id = Column(Integer, ForeignKey("futures_positions.id"), nullable=True, index=True)
    
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLEnum(FuturesOrderSide), nullable=False)
    order_type = Column(SQLEnum(FuturesOrderType), nullable=False)
    status = Column(SQLEnum(FuturesOrderStatus), nullable=False, default=FuturesOrderStatus.PENDING, index=True)
    
    # Order details
    price = Column(Numeric(20, 8), nullable=True)  # Null for market orders
    trigger_price = Column(Numeric(20, 8), nullable=True)  # For stop/TP orders
    quantity = Column(Numeric(20, 8), nullable=False)
    filled_quantity = Column(Numeric(20, 8), nullable=False, default=0)
    
    # Flags
    is_reduce_only = Column(Boolean, default=False)
    is_tp_order = Column(Boolean, default=False)
    is_sl_order = Column(Boolean, default=False)
    is_demo = Column(Boolean, default=False, index=True)
    
    # Execution
    filled_price = Column(Numeric(20, 8), nullable=True)
    fee = Column(Numeric(20, 8), nullable=False, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('idx_user_symbol', 'user_id', 'symbol'),
        Index('idx_position_status', 'position_id', 'status'),
    )

class FuturesLedger(Base):
    """Futures ledger for all margin/PnL movements"""
    __tablename__ = "futures_ledger"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    position_id = Column(Integer, ForeignKey("futures_positions.id"), nullable=True, index=True)
    order_id = Column(Integer, ForeignKey("futures_orders.id"), nullable=True)
    
    entry_type = Column(SQLEnum(FuturesLedgerType), nullable=False, index=True)
    asset = Column(String(10), nullable=False, default="USDT")
    
    # Amount (+ or -)
    amount = Column(Numeric(20, 8), nullable=False)
    balance_after = Column(Numeric(20, 8), nullable=False)
    
    # Context
    related_price = Column(Numeric(20, 8), nullable=True)
    meta_info = Column(Text, nullable=True)  # JSON / extra context (renamed from 'metadata')
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    
    __table_args__ = (
        Index('idx_user_type_created', 'user_id', 'entry_type', 'created_at'),
    )

class FuturesLiquidation(Base):
    """Liquidation events"""
    __tablename__ = "futures_liquidations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    position_id = Column(Integer, ForeignKey("futures_positions.id"), nullable=False, index=True)
    
    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum(PositionSide), nullable=False)
    
    # Liquidation details
    entry_price = Column(Numeric(20, 8), nullable=False)
    liquidation_price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    
    # Loss
    loss_amount = Column(Numeric(20, 8), nullable=False)
    liquidation_fee = Column(Numeric(20, 8), nullable=False)
    
    # Trigger
    trigger_price = Column(Numeric(20, 8), nullable=False)  # Actual mark price that triggered liquidation
    
    is_demo = Column(Boolean, default=False)
    liquidated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)

class FuturesWAL(Base):
    """Write-Ahead Log for rate-limited DB writes"""
    __tablename__ = "futures_wal"
    
    id = Column(Integer, primary_key=True, index=True)
    idempotency_key = Column(String(64), nullable=False, unique=True, index=True)
    
    operation = Column(String(50), nullable=False)  # create_position, update_pnl, close_position, etc.
    payload = Column(Text, nullable=False)  # JSON
    
    status = Column(String(20), nullable=False, default="PENDING", index=True)  # PENDING, APPLIED, FAILED
    retry_count = Column(Integer, nullable=False, default=0)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    applied_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
