"""
Futures Liquidation Engine - Risk Management
===========================================
Automatic liquidation when mark price crosses liquidation price.
"""

from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from loguru import logger

from app.futures.models import (
    FuturesPosition, PositionStatus, PositionSide,
    FuturesLiquidation, FuturesLedger, FuturesLedgerType
)
from app.futures.pnl import PnLCalculator
from app.db.models import User
import json

class LiquidationEngine:
    """Handles position liquidations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def check_liquidation(
        self,
        position: FuturesPosition,
        mark_price: Decimal
    ) -> bool:
        """
        Check if position should be liquidated
        
        Liquidation conditions:
        - Long: Mark Price <= Liquidation Price
        - Short: Mark Price >= Liquidation Price
        
        Returns:
            True if liquidation triggered
        """
        liquidation_price = Decimal(position.liquidation_price)
        
        if position.side == PositionSide.LONG:
            return mark_price <= liquidation_price
        else:  # SHORT
            return mark_price >= liquidation_price
    
    def liquidate_position(
        self,
        position: FuturesPosition,
        mark_price: Decimal
    ) -> Dict[str, Any]:
        """
        Execute liquidation
        
        Process:
        1. Close position at liquidation price
        2. Calculate loss
        3. Deduct liquidation fee
        4. Update user balance
        5. Create liquidation record
        6. Create ledger entries
        
        Returns:
            Liquidation details
        """
        try:
            user = self.db.query(User).filter(User.id == position.user_id).with_for_update().first()
            
            if not user:
                raise ValueError(f"User {position.user_id} not found")
            
            # Calculate loss
            entry_price = Decimal(position.entry_price)
            quantity = Decimal(position.quantity)
            initial_margin = Decimal(position.initial_margin)
            open_fee = Decimal(position.open_fee)
            funding_fees = Decimal(position.funding_fees_paid)
            
            # Loss is approximately equal to initial margin (total collateral)
            if position.side == PositionSide.LONG:
                price_loss = (entry_price - mark_price) * quantity
            else:  # SHORT
                price_loss = (mark_price - entry_price) * quantity
            
            # Calculate liquidation fee
            liquidation_fee = PnLCalculator.calculate_liquidation_fee(mark_price, quantity)
            
            # Total loss = margin + fees
            total_loss = initial_margin + liquidation_fee
            
            # Update position
            position.status = PositionStatus.LIQUIDATED
            position.unrealized_pnl = str(-total_loss)
            position.realized_pnl = str(-total_loss)
            position.close_fee = str(liquidation_fee)
            position.closed_at = datetime.now(timezone.utc)
            
            # Release margin (but user loses it)
            if not position.is_demo:
                # User balance doesn't increase - they lost the margin
                # Just unlock it from the locked state
                pass
            
            # Create liquidation record
            liquidation = FuturesLiquidation(
                user_id=position.user_id,
                position_id=position.id,
                symbol=position.symbol,
                side=position.side,
                entry_price=str(entry_price),
                liquidation_price=str(position.liquidation_price),
                quantity=str(quantity),
                loss_amount=str(total_loss),
                liquidation_fee=str(liquidation_fee),
                trigger_price=str(mark_price),
                is_demo=position.is_demo
            )
            self.db.add(liquidation)
            
            # Ledger entries
            if not position.is_demo:
                # Margin released (but lost)
                self.db.add(FuturesLedger(
                    user_id=user.id,
                    position_id=position.id,
                    entry_type=FuturesLedgerType.MARGIN_RELEASED,
                    asset="USDT",
                    amount=str(-initial_margin),
                    balance_after=user.balance_usdt,
                    related_price=str(mark_price),
                    metadata=json.dumps({"reason": "liquidation"})
                ))
                
                # Realized loss
                self.db.add(FuturesLedger(
                    user_id=user.id,
                    position_id=position.id,
                    entry_type=FuturesLedgerType.REALIZED_PNL,
                    asset="USDT",
                    amount=str(-total_loss),
                    balance_after=user.balance_usdt,
                    related_price=str(mark_price)
                ))
                
                # Liquidation fee
                self.db.add(FuturesLedger(
                    user_id=user.id,
                    position_id=position.id,
                    entry_type=FuturesLedgerType.LIQUIDATION_FEE,
                    asset="USDT",
                    amount=str(-liquidation_fee),
                    balance_after=user.balance_usdt,
                    related_price=str(mark_price)
                ))
            
            self.db.commit()
            
            logger.warning(
                f"LIQUIDATION: User {user.username} | Position #{position.id} | "
                f"{position.symbol} {position.side.value} | "
                f"Entry: {entry_price} | Liq: {mark_price} | Loss: {total_loss} USDT"
            )
            
            return {
                "liquidation_id": liquidation.id,
                "position_id": position.id,
                "symbol": position.symbol,
                "side": position.side.value,
                "entry_price": str(entry_price),
                "liquidation_price": str(mark_price),
                "quantity": str(quantity),
                "loss_amount": str(total_loss),
                "liquidation_fee": str(liquidation_fee),
                "timestamp": liquidation.liquidated_at.isoformat()
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Liquidation failed for position {position.id}: {e}")
            raise
    
    def check_all_positions(self, symbol: str, mark_price: Decimal) -> list:
        """
        Check all open positions for liquidation
        
        Called by price feed on each tick
        
        Returns:
            List of liquidation events
        """
        liquidations = []
        
        try:
            # Get all open positions for this symbol
            positions = self.db.query(FuturesPosition).filter(
                FuturesPosition.symbol == symbol,
                FuturesPosition.status == PositionStatus.OPEN
            ).all()
            
            for position in positions:
                if self.check_liquidation(position, mark_price):
                    try:
                        liq_event = self.liquidate_position(position, mark_price)
                        liquidations.append(liq_event)
                    except Exception as e:
                        logger.error(f"Failed to liquidate position {position.id}: {e}")
            
            return liquidations
            
        except Exception as e:
            logger.error(f"Error checking liquidations for {symbol}: {e}")
            return []