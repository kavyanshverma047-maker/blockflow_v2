"""
Futures Main Engine - Hybrid Demo/Live Orchestrator
==================================================
Routes operations to demo or live engine based on APP_MODE.
"""

import os
from decimal import Decimal
from typing import Dict, List, Optional, Union
from sqlalchemy.orm import Session
from loguru import logger

from app.futures.demo_engine import demo_engine, DemoPosition
from app.futures.live_engine import LiveEngine
from app.futures.models import FuturesPosition, PositionStatus, FuturesOrderSide
from app.futures.price_feed import price_feed

# Global mode toggle
APP_MODE = os.getenv("APP_MODE", "demo")  # "demo" or "live"

class FuturesEngine:
    """Hybrid futures engine that routes to demo/live"""
    
    def __init__(self, db: Session):
        self.db = db
        self.mode = APP_MODE
        self.live_engine = LiveEngine(db) if self.mode == "live" else None
    
    def is_demo(self) -> bool:
        """Check if running in demo mode"""
        return self.mode == "demo"
    
    async def create_position(
        self,
        user_id: int,
        symbol: str,
        side: FuturesOrderSide,
        quantity: Decimal,
        price: Decimal,
        leverage: int,
        take_profit_price: Optional[Decimal] = None,
        stop_loss_price: Optional[Decimal] = None
    ) -> Union[DemoPosition, FuturesPosition]:
        """Create position (demo or live)"""
        
        if self.is_demo():
            return await demo_engine.create_position(
                user_id, symbol, side, quantity, price, leverage,
                take_profit_price, stop_loss_price
            )
        else:
            return await self.live_engine.create_position(
                user_id, symbol, side, quantity, price, leverage,
                take_profit_price, stop_loss_price
            )
    
    async def close_position(
        self,
        position_id: int,
        exit_price: Decimal,
        quantity: Optional[Decimal] = None
    ) -> Dict:
        """Close position (demo or live)"""
        
        if self.is_demo():
            return await demo_engine.close_position(position_id, exit_price, quantity)
        else:
            return await self.live_engine.close_position(position_id, exit_price, quantity)
    
    def modify_tpsl(
        self,
        position_id: int,
        take_profit_price: Optional[Decimal] = None,
        stop_loss_price: Optional[Decimal] = None
    ) -> bool:
        """Modify TP/SL (demo or live)"""
        
        if self.is_demo():
            return demo_engine.modify_tpsl(position_id, take_profit_price, stop_loss_price)
        else:
            return self.live_engine.modify_tpsl(position_id, take_profit_price, stop_loss_price)
    
    def get_user_positions(
        self,
        user_id: int,
        status: Optional[PositionStatus] = None
    ) -> List[Union[DemoPosition, FuturesPosition]]:
        """Get user positions (demo or live)"""
        
        if self.is_demo():
            return demo_engine.get_user_positions(user_id, status)
        else:
            return self.live_engine.get_user_positions(user_id, status)
    
    def get_position(self, position_id: int) -> Optional[Union[DemoPosition, FuturesPosition]]:
        """Get specific position (demo or live)"""
        
        if self.is_demo():
            return demo_engine.get_position(position_id)
        else:
            return self.live_engine.get_position(position_id)
    
    def get_account_summary(self, user_id: int) -> Dict:
        """Get account summary (demo or live)"""
        
        if self.is_demo():
            return demo_engine.get_account_summary(user_id)
        else:
            return self.live_engine.get_account_summary(user_id)
    
    async def process_price_tick(self, symbol: str, mark_price: Decimal):
        """Process price tick - update PnL, check TP/SL, check liquidations"""
        
        try:
            # Update all open positions
            if self.is_demo():
                positions = [p for p in demo_engine.positions.values() 
                           if p.symbol == symbol and p.status == PositionStatus.OPEN]
                
                for position in positions:
                    demo_engine.update_position_pnl(position, mark_price)
            else:
                positions = self.db.query(FuturesPosition).filter(
                    FuturesPosition.symbol == symbol,
                    FuturesPosition.status == PositionStatus.OPEN,
                    FuturesPosition.is_demo == False
                ).all()
                
                for position in positions:
                    self.live_engine.update_position_pnl(position, mark_price)
            
            # Check TP/SL triggers
            if self.is_demo():
                tpsl_events = await demo_engine.check_tpsl_triggers(symbol, mark_price)
            else:
                tpsl_events = await self.live_engine.check_tpsl_triggers(symbol, mark_price)
            
            # Check liquidations
            if self.is_demo():
                liq_events = await demo_engine.check_liquidations(symbol, mark_price)
            else:
                liq_events = await self.live_engine.check_liquidations(symbol, mark_price)
            
            # Return events for WebSocket broadcast
            return {
                "tpsl_triggers": tpsl_events,
                "liquidations": liq_events
            }
            
        except Exception as e:
            logger.error(f"Error processing price tick for {symbol}: {e}")
            return {"tpsl_triggers": [], "liquidations": []}
    
    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current mark price from price feed"""
        return price_feed.get_current_price(symbol)
    
    def switch_mode(self, new_mode: str):
        """Switch between demo and live mode (admin only)"""
        if new_mode not in ["demo", "live"]:
            raise ValueError("Mode must be 'demo' or 'live'")
        
        self.mode = new_mode
        
        if new_mode == "live" and not self.live_engine:
            self.live_engine = LiveEngine(self.db)
        
        logger.warning(f"Futures engine mode switched to: {new_mode}")