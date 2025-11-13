# app/models.py
"""
Blockflow Exchange - Production-grade SQLAlchemy models
Version: 3.6 (production-ready schema)
Postgres-first (uses JSONB, Numeric)
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Boolean,
    DateTime,
    Text,
    Index,
    UniqueConstraint,
    ForeignKey,
    Numeric,
    CheckConstraint,
    func,
    text,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# -------------------------
# Enums
# -------------------------
class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    COMPLIANCE = "compliance"

class KYCStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"

class OrderStatus(str, Enum):
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"

# -------------------------
# Helper constants
# -------------------------
DECIMAL_MONEY = Numeric(36, 18, asdecimal=True)  # high precision for crypto amounts
DECIMAL_FX = Numeric(24, 8, asdecimal=True)

# -------------------------
# User model
# -------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    role = Column(SQLEnum(UserRole, name="user_role_enum"), nullable=False, server_default=text(f"'{UserRole.USER.value}'"))

    # KYC fields
    kyc_status = Column(SQLEnum(KYCStatus, name="kyc_status_enum"), nullable=False, server_default=text(f"'{KYCStatus.PENDING.value}'"))
    pan_number = Column(String(10), unique=True, nullable=True, index=True)
    aadhaar_last4 = Column(String(4), nullable=True)
    full_name = Column(String(255), nullable=True)
    kyc_verified_at = Column(DateTime(timezone=True), nullable=True)

    # Balances (Numeric for correct arithmetic)
    balance_inr = Column(DECIMAL_MONEY, nullable=False, server_default="0")
    balance_usdt = Column(DECIMAL_MONEY, nullable=False, server_default="0")
    balance_btc = Column(DECIMAL_MONEY, nullable=False, server_default="0")
    balance_eth = Column(DECIMAL_MONEY, nullable=False, server_default="0")

    # Locked balances for open orders
    locked_inr = Column(DECIMAL_MONEY, nullable=False, server_default="0")
    locked_usdt = Column(DECIMAL_MONEY, nullable=False, server_default="0")
    locked_btc = Column(DECIMAL_MONEY, nullable=False, server_default="0")
    locked_eth = Column(DECIMAL_MONEY, nullable=False, server_default="0")

    # Flags
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    is_banned = Column(Boolean, nullable=False, server_default=text("false"))
    is_demo = Column(Boolean, nullable=False, server_default=text("false"))

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationships
    orders = relationship("Order", back_populates="user", cascade="save-update, merge")
    trades_as_buyer = relationship("Trade", back_populates="buyer", foreign_keys="Trade.buyer_id")
    trades_as_seller = relationship("Trade", back_populates="seller", foreign_keys="Trade.seller_id")
    ledger_entries = relationship("Ledger", back_populates="user", cascade="all, delete-orphan")
    tax_entries = relationship("TaxEntry", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_user_email_active", "email", "is_active"),
        Index("idx_user_username", "username"),
    )

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"

# -------------------------
# Order model
# -------------------------
class Order(Base):
    __tablename__ = "orders"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    symbol = Column(String(32), nullable=False, index=True)  # e.g., BTCUSDT
    side = Column(SQLEnum(OrderSide, name="order_side_enum"), nullable=False)
    order_type = Column(SQLEnum(OrderType, name="order_type_enum"), nullable=False)
    status = Column(SQLEnum(OrderStatus, name="order_status_enum"), nullable=False, server_default=text(f"'{OrderStatus.OPEN.value}'"), index=True)

    price = Column(DECIMAL_MONEY, nullable=True)   # NULL for market orders
    amount = Column(DECIMAL_MONEY, nullable=False)
    filled_amount = Column(DECIMAL_MONEY, nullable=False, server_default="0")
    remaining_amount = Column(DECIMAL_MONEY, nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="orders")
    trades = relationship("Trade", back_populates="order", primaryjoin="Order.id==Trade.buy_order_id or Order.id==Trade.sell_order_id", viewonly=True)

    __table_args__ = (
        Index("idx_order_symbol_status", "symbol", "status"),
        Index("idx_order_user_status", "user_id", "status"),
        Index("idx_order_created", "created_at"),
        CheckConstraint("amount >= 0", name="check_order_amount_nonnegative"),
    )

    def __repr__(self):
        return f"<Order(id={self.id}, symbol={self.symbol}, side={self.side}, status={self.status})>"

# -------------------------
# Trade model
# -------------------------
class Trade(Base):
    __tablename__ = "trades"

    id = Column(BigInteger, primary_key=True)
    buy_order_id = Column(BigInteger, ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True)
    sell_order_id = Column(BigInteger, ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True)

    symbol = Column(String(32), nullable=False, index=True)
    price = Column(DECIMAL_MONEY, nullable=False)
    amount = Column(DECIMAL_MONEY, nullable=False)

    buyer_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    seller_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    buyer_fee = Column(DECIMAL_MONEY, nullable=False, server_default="0")
    seller_fee = Column(DECIMAL_MONEY, nullable=False, server_default="0")
    tds_amount_inr = Column(DECIMAL_MONEY, nullable=False, server_default="0")

    executed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    buyer = relationship("User", back_populates="trades_as_buyer", foreign_keys=[buyer_id])
    seller = relationship("User", back_populates="trades_as_seller", foreign_keys=[seller_id])
    order = relationship("Order", primaryjoin="or_(Trade.buy_order_id==Order.id, Trade.sell_order_id==Order.id)", viewonly=True)

    __table_args__ = (
        Index("idx_trade_symbol_time", "symbol", "executed_at"),
        Index("idx_trade_buyer", "buyer_id", "executed_at"),
        Index("idx_trade_seller", "seller_id", "executed_at"),
        CheckConstraint("price >= 0", name="check_trade_price_nonnegative"),
        CheckConstraint("amount >= 0", name="check_trade_amount_nonnegative"),
    )

    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, amount={self.amount} @ {self.price})>"

# -------------------------
# Ledger (immutable audit)
# -------------------------
class Ledger(Base):
    __tablename__ = "ledger"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    entry_type = Column(String(30), nullable=False)  # deposit, withdrawal, trade, fee, tds
    asset = Column(String(16), nullable=False)  # INR, USDT, BTC, ETH

    amount = Column(DECIMAL_MONEY, nullable=False)
    balance_after = Column(DECIMAL_MONEY, nullable=False)

    related_id = Column(BigInteger, nullable=True)  # trade_id, order_id, etc.
    meta_info = Column(JSONB, nullable=True)  # JSONB metadata (was 'metadata' before causing conflicts)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    user = relationship("User", back_populates="ledger_entries")

    __table_args__ = (
        Index("idx_ledger_user_time", "user_id", "created_at"),
        Index("idx_ledger_type", "entry_type", "created_at"),
    )

    def __repr__(self):
        return f"<Ledger(id={self.id}, user_id={self.user_id}, type={self.entry_type}, amount={self.amount})>"

# -------------------------
# TaxEntry (TDS tracking)
# -------------------------
class TaxEntry(Base):
    __tablename__ = "tax_entries"

    id = Column(BigInteger, primary_key=True)
    trade_id = Column(BigInteger, ForeignKey("trades.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    symbol = Column(String(32), nullable=False)
    gross_value_crypto = Column(DECIMAL_MONEY, nullable=False)
    gross_value_inr = Column(DECIMAL_MONEY, nullable=False)
    fx_rate = Column(DECIMAL_FX, nullable=False)

    tds_rate = Column(Numeric(6, 4), nullable=False, server_default="0.0100")  # 1% default
    tds_amount_inr = Column(DECIMAL_MONEY, nullable=False)
    net_amount_inr = Column(DECIMAL_MONEY, nullable=False)

    quarter = Column(String(16), nullable=False, index=True)  # e.g., Q3-FY25
    form_26qe_filed = Column(Boolean, nullable=False, server_default=text("false"))
    form_26qe_date = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user = relationship("User", back_populates="tax_entries")

    __table_args__ = (
        Index("idx_tax_quarter_user", "quarter", "user_id"),
        Index("idx_tax_trade", "trade_id"),
    )

    def __repr__(self):
        return f"<TaxEntry(id={self.id}, quarter={self.quarter}, tds_inr={self.tds_amount_inr})>"

# -------------------------
# AuditLog
# -------------------------
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    request_id = Column(String(64), nullable=True, index=True)  # correlation id
    event_type = Column(String(64), nullable=False, index=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(512), nullable=True)
    details = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("idx_audit_user_time", "user_id", "created_at"),
        Index("idx_audit_event", "event_type", "created_at"),
        Index("idx_audit_request", "request_id"),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, event={self.event_type}, user_id={self.user_id})>"

# -------------------------
# Utilities
# -------------------------
def init_db(engine):
    """Create all tables using the single Base metadata root."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")

def drop_all_tables(engine):
    """Drop all tables (development only)."""
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All database tables dropped")

def model_to_dict(instance):
    """Convert an ORM instance to dict (shallow)."""
    return {c.name: getattr(instance, c.name) for c in instance.__table__.columns}

# attach to_dict helpers
User.to_dict = lambda self: model_to_dict(self)
Order.to_dict = lambda self: model_to_dict(self)
Trade.to_dict = lambda self: model_to_dict(self)
Ledger.to_dict = lambda self: model_to_dict(self)
TaxEntry.to_dict = lambda self: model_to_dict(self)
AuditLog.to_dict = lambda self: model_to_dict(self)

# Exports
__all__ = [
    "Base",
    "User",
    "Order",
    "Trade",
    "Ledger",
    "TaxEntry",
    "AuditLog",
    "init_db",
    "drop_all_tables",
]
