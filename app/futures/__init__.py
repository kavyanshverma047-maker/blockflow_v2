"""
Blockflow Futures Trading Module
================================
USDM Perpetual Futures with demo/live mode toggle.
"""

from app.futures.engine import FuturesEngine
from app.futures.price_feed import price_feed
from app.futures.demo_engine import demo_engine
from app.futures.router import router as futures_router   # <-- IMPORTANT

__all__ = [
    "FuturesEngine",
    "price_feed",
    "demo_engine",
    "futures_router",   # <-- EXPORT THE ROUTER
]
