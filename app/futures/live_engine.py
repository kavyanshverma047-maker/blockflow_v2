"""
Futures Live Engine - Production DB Persistence
==============================================
Handles all futures operations with real DB writes and WAL fallback.
"""

from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from loguru import logger
import json

from app.futures.models import (
    FuturesPosition, FuturesOrder, FuturesLedger, FuturesLedgerType,
    PositionSide, PositionStatus, FuturesOrderType, FuturesOrderSide, FuturesOrderStatus
)
from app.futures.margin import MarginCalculator
from app.futures.pnl import PnLCalculator
from app.futures.liquidation import LiquidationEngine
from app.futures.tpsl import TPSLManager
from app.futures.wal import WALSystem
from app.db.models import User

class LiveEngine:
    """Production futures engine with DB persistence"""
    
    def __init__(self, db: Session):
        self.db = db
        self.wal = WALSystem(db)
        self.liquidation_engine = LiquidationEngine(db)
        self.tpsl_manager = TPSLManager(db)
    
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
    ) -> FuturesPosition:
        """Create new position with DB persistence"""
        
        try:
            # Get user with lock
            user = self.db.query(User).filter(User.id == user_id).with_for_update().first()
            
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            # Calculate margins
            initial_margin = MarginCalculator.calculate_initial_margin(price, quantity, leverage)
            maintenance_margin = MarginCalculator.calculate_maintenance_margin(price, quantity)
            
            # Calculate open fee
            open_fee = PnLCalculator.calculate_trading_fee(price, quantity, is_maker=False)
            
            # Calculate liquidation price
            position_side = PositionSide.LONG if side == FuturesOrderSide.BUY else PositionSide.SHORT
            liquidation_price = MarginCalculator.calculate_liquidation_price(
                position_side,
                price,
                quantity,
                leverage,
                initial_margin
            )
            
            # Check balance
            required_margin = initial_margin + open_fee
            user_balance = Decimal(user.balance_usdt)
            
            if user_balance < required_margin:
                raise ValueError(f"Insufficient balance. Required: {required_margin} USDT")
            
            # Create position
            position = FuturesPosition(
                user_id=user_id,
                symbol=symbol,
                side=position_side,
                entry_price=str(price),
                quantity=str(quantity),
                leverage=leverage,
                initial_margin=str(initial_margin),
                maintenance_margin=str(maintenance_margin),
                liquidation_price=str(liquidation_price),
                take_profit_price=str(take_profit_price) if take_profit_price else None,
                stop_loss_price=str(stop_loss_price) if stop_loss_price else None,
                open_fee=str(open_fee),
                is_demo=False
            )
            
            self.db.add(position)
            self.db.flush()
            
            # Deduct margin from balance
            user.balance_usdt = str(user_balance - required_margin)
            
            # Create ledger entry
            self.db.add(FuturesLedger(
                user_id=user_id,
                position_id=position.id,
                entry_type=FuturesLedgerType.MARGIN_LOCKED,
                asset="USDT",
                amount=str(-required_margin),
                balance_after=user.balance_usdt,
                related_price=str(price),
                metadata=json.dumps({
                    "initial_margin": str(initial_margin),
                    "open_fee": str(open_fee)
                })
            ))
            
            self.db.commit()
            self.db.refresh(position)
            
            logger.info(
                f"Position created: #{position.id} | User {user_id} | "
                f"{symbol} {position_side.value} {quantity} @ {price} | {leverage}x"
            )
            
            return position
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create position: {e}")
            raise
    
    async def close_position(
        self,
        position_id: int,
        exit_price: Decimal,
        quantity: Optional[Decimal] = None
    ) -> Dict:
        """Close position with DB persistence"""
        
        try:
            position = self.db.query(FuturesPosition).filter(
                FuturesPosition.id == position_id
            ).with_for_update().first()
            
            if not position:
                raise ValueError(f"Position {position_id} not found")
            
            if position.status != PositionStatus.OPEN:
                raise ValueError(f"Position {position_id} is not open")
            
            user = self.db.query(User).filter(User.id == position.user_id).with_for_update().first()
            
            # Full or partial close
            position_quantity = Decimal(position.quantity)
            close_quantity = quantity if quantity else position_quantity
            
            if close_quantity > position_quantity:
                raise ValueError("Close quantity exceeds position size")
            
            # Calculate close fee
            close_fee = PnLCalculator.calculate_trading_fee(exit_price, close_quantity, is_maker=False)
            
            # Calculate realized PnL
            entry_price = Decimal(position.entry_price)
            initial_margin = Decimal(position.initial_margin)
            open_fee = Decimal(position.open_fee)
            funding_fees = Decimal(position.funding_fees_paid)
            
            realized_pnl = PnLCalculator.calculate_realized_pnl(
                position.side,
                entry_price,
                exit_price,
                close_quantity,
                open_fee,
                close_fee,
                funding_fees
            )
            
            user_balance = Decimal(user.balance_usdt)
            
            # Partial close
            if close_quantity < position_quantity:
                remaining_quantity = position_quantity - close_quantity
                remaining_margin_ratio = remaining_quantity / position_quantity
                
                # Update position
                position.quantity = str(remaining_quantity)
                position.initial_margin = str(initial_margin * remaining_margin_ratio)
                position.realized_pnl = str(Decimal(position.realized_pnl) + realized_pnl)
                
                # Return partial margin + PnL
                margin_returned = initial_margin * (Decimal("1") - remaining_margin_ratio)
                balance_change = margin_returned + realized_pnl - close_fee
                user.balance_usdt = str(user_balance + balance_change)
                
            else:
                # Full close
                position.status = PositionStatus.CLOSED
                position.realized_pnl = str(realized_pnl)
                position.close_fee = str(close_fee)
                position.closed_at = datetime.now(timezone.utc)
                
                # Return full margin + PnL - close fee
                balance_change = initial_margin + realized_pnl - close_fee
                user.balance_usdt = str(user_balance + balance_change)
            
            # Ledger entries
            self.db.add(FuturesLedger(
                user_id=user.id,
                position_id=position.id,
                entry_type=FuturesLedgerType.MARGIN_RELEASED,
                asset="USDT",
                amount=str(initial_margin if close_quantity == position_quantity else initial_margin * (Decimal("1") - (remaining_quantity / position_quantity))),
                balance_after=user.balance_usdt,
                related_price=str(exit_price)
            ))
            
            self.db.add(FuturesLedger(
                user_id=user.id,
                position_id=position.id,
                entry_type=FuturesLedgerType.REALIZED_PNL,
                asset="USDT",
                amount=str(realized_pnl),
                balance_after=user.balance_usdt,
                related_price=str(exit_price)
            ))
            
            self.db.add(FuturesLedger(
                user_id=user.id,
                position_id=position.id,
                entry_type=FuturesLedgerType.TRADING_FEE,
                asset="USDT",
                amount=str(-close_fee),
                balance_after=user.balance_usdt,
                related_price=str(exit_price)
            ))
            
            self.db.commit()
            
            logger.info(
                f"Position closed: #{position_id} | "
                f"Exit: {exit_price} | Qty: {close_quantity} | PnL: {realized_pnl} USDT"
            )
            
            return {
                "position_id": position_id,
                "closed_quantity": str(close_quantity),
                "exit_price": str(exit_price),
                "realized_pnl": str(realized_pnl),
                "close_fee": str(close_fee)
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to close position: {e}")
            raise
    
    def modify_tpsl(
        self,
        position_id: int,
        take_profit_price: Optional[Decimal] = None,
        stop_loss_price: Optional[Decimal] = None
    ) -> bool:
        """Modify TP/SL with DB persistence"""
        
        try:
            position = self.db.query(FuturesPosition).filter(
                FuturesPosition.id == position_id
            ).with_for_update().first()
            
            if not position:
                raise ValueError(f"Position {position_id} not found")
            
            success, message = self.tpsl_manager.set_tpsl(
                position,
                take_profit_price,
                stop_loss_price
            )
            
            if not success:
                raise ValueError(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to modify TP/SL: {e}")
            raise
    
    def get_user_positions(
        self,
        user_id: int,
        status: Optional[PositionStatus] = None
    ) -> List[FuturesPosition]:
        """Get all positions for user"""
        
        query = self.db.query(FuturesPosition).filter(
            FuturesPosition.user_id == user_id,
            FuturesPosition.is_demo == False
        )
        
        if status:
            query = query.filter(FuturesPosition.status == status)
        
        return query.order_by(FuturesPosition.opened_at.desc()).all()
    
    def get_position(self, position_id: int) -> Optional[FuturesPosition]:
        """Get specific position"""
        return self.db.query(FuturesPosition).filter(
            FuturesPosition.id == position_id
        ).first()
    
    def update_position_pnl(self, position: FuturesPosition, mark_price: Decimal):
        """Update position's unrealized PnL"""
        
        try:
            position.unrealized_pnl = str(PnLCalculator.calculate_unrealized_pnl(
                position.side,
                Decimal(position.entry_price),
                mark_price,
                Decimal(position.quantity)
            ))
            position.last_updated = datetime.now(timezone.utc)
            
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update PnL for position {position.id}: {e}")
    
    async def check_tpsl_triggers(self, symbol: str, mark_price: Decimal) -> List[Dict]:
        """Check all positions for TP/SL triggers"""
        return self.tpsl_manager.check_all_positions(symbol, mark_price)
    
    async def check_liquidations(self, symbol: str, mark_price: Decimal) -> List[Dict]:
        """Check all positions for liquidation"""
        return self.liquidation_engine.check_all_positions(symbol, mark_price)
    
    def get_account_summary(self, user_id: int) -> Dict:
        """Get account summary"""
        
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            positions = self.get_user_positions(user_id, PositionStatus.OPEN)
            
            total_margin = sum(Decimal(p.initial_margin) for p in positions)
            total_unrealized_pnl = sum(Decimal(p.unrealized_pnl) for p in positions)
            balance = Decimal(user.balance_usdt)
            
            account_value = balance + total_unrealized_pnl
            
            return {
                "total_balance": str(balance),
                "available_balance": str(balance - total_margin),
                "total_margin_used": str(total_margin),
                "total_unrealized_pnl": str(total_unrealized_pnl),
                "account_value": str(account_value),
                "margin_ratio": str((total_margin / account_value * 100).quantize(Decimal("0.01"))) if account_value > 0 else "0.00",
                "open_positions_count": len(positions)
            }
            
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            raise