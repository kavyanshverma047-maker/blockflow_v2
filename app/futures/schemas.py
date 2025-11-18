from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


# ======================
# ENUMS
# ======================

class FuturesOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class FuturesOrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"


# ======================
# REQUEST SCHEMAS (router.py EXACT NAMES)
# ======================

class CreateFuturesOrderRequest(BaseModel):
    symbol: str
    side: FuturesOrderSide
    order_type: FuturesOrderType
    quantity: float = Field(gt=0)
    leverage: int = Field(gt=0, le=125)
    price: Optional[float] = None


class CancelOrderRequest(BaseModel):
    order_id: int


class ClosePositionRequest(BaseModel):
    position_id: int


class ModifyTPSLRequest(BaseModel):
    position_id: int
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


# ======================
# RESPONSE SCHEMAS
# ======================

class FuturesOrderResponse(BaseModel):
    id: int
    symbol: str
    side: FuturesOrderSide
    order_type: FuturesOrderType
    quantity: float
    price: Optional[float]
    status: str

    class Config:
        from_attributes = True


class FuturesPositionResponse(BaseModel):
    id: int
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    leverage: int
    status: PositionStatus
    unrealized_pnl: float

    class Config:
        from_attributes = True
