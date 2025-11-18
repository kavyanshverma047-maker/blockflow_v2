from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import User
from app.core.auth import hash_password, create_access_token
from pydantic import BaseModel

router = APIRouter(prefix="/api/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


@router.post("/register")
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()

    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password)
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.email})

    return {"message": "User registered successfully", "access_token": token}
