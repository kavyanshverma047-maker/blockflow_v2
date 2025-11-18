from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.main import get_db, get_current_user
from app.main import get_asset_balance_field, get_locked_balance_field, lock_balance, unlock_balance
from pydantic import BaseModel
from app.db.models import User

router = APIRouter(prefix="/api/wallet", tags=["wallet"])

class DepositReq(BaseModel):
    asset: str
    amount: float

@router.get("/balances")
def balances(current_user: User = Depends(get_current_user)):
    return {
        "INR": {"available": current_user.balance_inr, "locked": current_user.locked_inr},
        "USDT": {"available": current_user.balance_usdt, "locked": current_user.locked_usdt},
        "BTC": {"available": current_user.balance_btc, "locked": current_user.locked_btc},
        "ETH": {"available": current_user.balance_eth, "locked": current_user.locked_eth},
        "is_demo": current_user.is_demo
    }

@router.post("/deposit")
def deposit(req: DepositReq, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not hasattr(current_user, get_asset_balance_field(req.asset)):
        raise HTTPException(status_code=400, detail="Unknown asset")
    # demo-only deposit
    balance_field = get_asset_balance_field(req.asset)
    new = float(getattr(current_user, balance_field)) + float(req.amount)
    setattr(current_user, balance_field, str(new))
    db.commit()
    return {"success": True, "asset": req.asset, "new_balance": str(new)}
