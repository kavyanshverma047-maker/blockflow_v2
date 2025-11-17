# app/futures/router.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.futures.engine import FuturesEngine
from app.futures.schemas import (
    CreateFuturesOrderRequest,
    CancelOrderRequest,
    ClosePositionRequest,
    ModifyTPSLRequest
)

router = APIRouter(prefix="/futures", tags=["Futures"])


@router.post("/order")
def create_futures_order(
    payload: CreateFuturesOrderRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    engine = FuturesEngine(db)
    return engine.create_order(user_id=user.id, data=payload)


@router.post("/order/cancel")
def cancel_order(
    payload: CancelOrderRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    engine = FuturesEngine(db)
    return engine.cancel_order(user_id=user.id, order_id=payload.order_id)


@router.post("/position/close")
def close_position(
    payload: ClosePositionRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    engine = FuturesEngine(db)
    return engine.close_position(user_id=user.id, position_id=payload.position_id)


@router.post("/position/modify-tpsl")
def modify_tp_sl(
    payload: ModifyTPSLRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    engine = FuturesEngine(db)
    return engine.modify_tp_sl(
        user_id=user.id,
        position_id=payload.position_id,
        take_profit=payload.take_profit,
        stop_loss=payload.stop_loss,
    )


@router.get("/positions")
def get_positions(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    engine = FuturesEngine(db)
    return engine.get_positions(user_id=user.id)


@router.get("/orders")
def get_orders(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    engine = FuturesEngine(db)
    return engine.get_orders(user_id=user.id)
