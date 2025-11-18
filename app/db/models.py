from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Float,
    Text,
    Enum as SQLEnum,
    ForeignKey,
    Numeric,
    func,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from decimal import Decimal
from enum import Enum
from app.database import Base


# ============================================================
# ENUMS
# ============================================================

class UserRole(Enum):
    USER = "user"
    ADMIN = "admin"
    COMPLIANCE = "compliance"


class KYCStatus(Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


# ============================================================
# USER MODEL
# ============================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    is_active = Column(Boolean, default=True)
    is_demo = Column(Boolean, default=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    kyc_status = Column(SQLEnum(KYCStatus), default=KYCStatus.PENDING)

    balance_inr = Column(Numeric(20, 8), default=Decimal("0"))
    locked_inr = Column(Numeric(20, 8), default=Decimal("0"))

    balance_usdt = Column(Numeric(20, 8), default=Decimal("0"))
    locked_usdt = Column(Numeric(20, 8), default=Decimal("0"))

    balance_btc = Column(Numeric(20, 8), default=Decimal("0"))
    locked_btc = Column(Numeric(20, 8), default=Decimal("0"))

    balance_eth = Column(Numeric(20, 8), default=Decimal("0"))
    locked_eth = Column(Numeric(20, 8), default=Decimal("0"))

    created_at = Column(DateTime, server_default=func.now())


# ============================================================
# ORDER MODEL
# ============================================================

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User")

    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum(OrderSide), nullable=False)
    order_type = Column(SQLEnum(OrderType), nullable=False)

    price = Column(Numeric(20, 8), nullable=True)
    amount = Column(Numeric(20, 8), nullable=False)

    filled_amount = Column(Numeric(20, 8), default=Decimal("0"))
    remaining_amount = Column(Numeric(20, 8), nullable=False)

    status = Column(SQLEnum(OrderStatus), default=OrderStatus.OPEN)

    created_at = Column(DateTime, server_default=func.now())


# ============================================================
# TRADE MODEL
# ============================================================

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)

    buy_order_id = Column(Integer)
    sell_order_id = Column(Integer)
    buyer_id = Column(Integer)
    seller_id = Column(Integer)

    symbol = Column(String(20))
    price = Column(Numeric(20, 8))
    amount = Column(Numeric(20, 8))

    buyer_fee = Column(Numeric(20, 8), default=Decimal("0"))
    seller_fee = Column(Numeric(20, 8), default=Decimal("0"))
    tds_amount_inr = Column(Numeric(20, 8), default=Decimal("0"))

    executed_at = Column(DateTime, server_default=func.now())


# ============================================================
# LEDGER MODEL
# ============================================================

class Ledger(Base):
    __tablename__ = "ledger"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    entry_type = Column(String(50))

    asset = Column(String(10))
    amount = Column(Numeric(20, 8))
    balance_after = Column(Numeric(20, 8))

    meta_json = Column(Text)  # FIXED: renamed from 'metadata'
    created_at = Column(DateTime, server_default=func.now())


# ============================================================
# TAX ENTRY
# ============================================================

class TaxEntry(Base):
    __tablename__ = "tax_entries"

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer)
    user_id = Column(Integer)

    symbol = Column(String(20))
    gross_value_crypto = Column(Numeric(20, 8))
    gross_value_inr = Column(Numeric(20, 8))

    fx_rate = Column(Numeric(20, 8))
    tds_rate = Column(Numeric(20, 8))
    tds_amount_inr = Column(Numeric(20, 8))
    net_amount_inr = Column(Numeric(20, 8))

    quarter = Column(String(10))

    created_at = Column(DateTime, server_default=func.now())


# ============================================================
# AUDIT LOG
# ============================================================

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)

    request_id = Column(String(255))
    event_type = Column(String(50))

    details = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
