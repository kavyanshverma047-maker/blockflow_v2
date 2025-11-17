# app/futures/schemas.py

from pydantic import BaseModel, EmailStr
from datetime import datetime
from enum import Enum
from typing import Optional


# ------------------------------
# ENUM DEFINITIONS
# ------------------------------

class OrderSide(str, Enum):
    buy = "buy"
    sell = "sell"


class OrderType(str, Enum):
    market = "market"
    limit = "limit"


# ------------------------------
# ORDER CREATION REQUEST
# ------------------------------

class CreateFuturesOrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.market
    quantity: float
    leverage: int = 20
    price: Optional[float] = None        # for LIMIT orders
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    reduce_only: bool = False
    is_demo: bool = True


# ------------------------------
# CANCEL ORDER REQUEST
# ------------------------------

class CancelOrderRequest(BaseModel):
    order_id: int
    symbol: str
    is_demo: bool = True


# ------------------------------
# CLOSE POSITION REQUEST
# ------------------------------

class ClosePositionRequest(BaseModel):
    position_id: int
    symbol: str
    is_demo: bool = True


# ------------------------------
# MODIFY TP/SL REQUEST
# ------------------------------

class ModifyTPSLRequest(BaseModel):
    position_id: int
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    is_demo: bool = True


# ------------------------------
# POSITION RESPONSE
# ------------------------------

class FuturesPositionResponse(BaseModel):
    id: int
    symbol: str
    side: str
    quantity: float
    entry_price: float
    leverage: int
    unrealized_pnl: float
    take_profit: Optional[float]
    stop_loss: Optional[float]

    class Config:
        orm_mode = True
