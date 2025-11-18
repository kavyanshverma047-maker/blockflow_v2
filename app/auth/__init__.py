# package shim so `from app.auth import auth_router` imports a module-like object
from . import auth_router as auth_router
