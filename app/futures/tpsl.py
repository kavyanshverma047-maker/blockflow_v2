"""
Futures TP/SL Manager - Take Profit & Stop Loss
==============================================
Handles TP/SL orders and automatic position closing.
"""

from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from loguru import logger
import json

from app.futures.models import (
    FuturesPosition, PositionStatus, PositionSide,
    FuturesOrder, FuturesOrderType, FuturesOrderSide, FuturesOrderStatus,
    FuturesLedger, FuturesLedgerType
)
from app.futures.pnl import PnLCalculator
from app.db.models import User

class TPSLManager:
    """Manage Take Profit and Stop Loss orders"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def set_tpsl(
        self,
        position: FuturesPosition,
        take_profit_price: Optional[Decimal] = None,
        stop_loss_price: Optional[Decimal] = None
    ) -> Tuple[bool, str]:
        """
        Set or update TP/SL for position
        
        Validation:
        - Long TP must be > entry price, SL must be < entry price
        - Short TP must be < entry price, SL must be > entry price
        
        Returns:
            (success, message)
        """
        try:
            entry_price = Decimal(position.entry_price)
            
            # Validate TP
            if take_profit_price is not None:
                if position.side == PositionSide.LONG:
                    if take_profit_price <= entry_price:
                        return False, "Take profit must be above entry price for long positions"
                else:  # SHORT
                    if take_profit_price >= entry_price:
                        return False, "Take profit must be below entry price for short positions"
                
                position.take_profit_price = str(take_profit_price)
            
            # Validate SL
            if stop_loss_price is not None:
                if position.side == PositionSide.LONG:
                    if stop_loss_price >= entry_price:
                        return False, "Stop loss must be below entry price for long positions"
                else:  # SHORT
                    if stop_loss_price <= entry_price:
                        return False, "Stop loss must be above entry price for short positions"
                
                # Check SL is not beyond liquidation price
                liq_price = Decimal(position.liquidation_price)
                if position.side == PositionSide.LONG:
                    if stop_loss_price < liq_price:
                        return False, f"Stop loss too close to liquidation price ({liq_price})"
                else:  # SHORT
                    if stop_loss_price > liq_price:
                        return False, f"Stop loss too close to liquidation price ({liq_price})"
                
                position.stop_loss_price = str(stop_loss_price)
            
            self.db.commit()
            
            logger.info(
                f"TP/SL updated for position #{position.id} | "
                f"TP: {take_profit_price or 'None'} | SL: {stop_loss_price or 'None'}"
            )
            
            return True, "TP/SL updated successfully"
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to set TP/SL for position {position.id}: {e}")
            return False, str(e)
    
    def check_tpsl_triggers(
        self,
        position: FuturesPosition,
        mark_price: Decimal
    ) -> Optional[Dict[str, Any]]:
        """
        Check if TP or SL should trigger
        
        Returns:
            Trigger event details if triggered, else None
        """
        if position.status != PositionStatus.OPEN:
            return None
        
        triggered_type = None
        trigger_price = None
        
        # Check Take Profit
        if position.take_profit_price:
            tp_price = Decimal(position.take_profit_price)
            
            if position.side == PositionSide.LONG:
                if mark_price >= tp_price:
                    triggered_type = "take_profit"
                    trigger_price = tp_price
            else:  # SHORT
                if mark_price <= tp_price:
                    triggered_type = "take_profit"
                    trigger_price = tp_price
        
        # Check Stop Loss (only if TP not triggered)
        if not triggered_type and position.stop_loss_price:
            sl_price = Decimal(position.stop_loss_price)
            
            if position.side == PositionSide.LONG:
                if mark_price <= sl_price:
                    triggered_type = "stop_loss"
                    trigger_price = sl_price
            else:  # SHORT
                if mark_price >= sl_price:
                    triggered_type = "stop_loss"
                    trigger_price = sl_price
        
        if triggered_type:
            return {
                "type": triggered_type,
                "trigger_price": trigger_price,
                "mark_price": mark_price
            }
        
        return None
    
    def execute_tpsl_close(
        self,
        position: FuturesPosition,
        trigger_type: str,
        exit_price: Decimal
    ) -> Dict[str, Any]:
        """
        Close position due to TP/SL trigger
        
        Process:
        1. Calculate realized PnL
        2. Calculate close fee
        3. Update user balance
        4. Update position status
        5. Create ledger entries
        
        Returns:
            Closure details
        """
        try:
            user = self.db.query(User).filter(User.id == position.user_id).with_for_update().first()
            
            if not user:
                raise ValueError(f"User {position.user_id} not found")
            
            # Calculate PnL
            entry_price = Decimal(position.entry_price)
            quantity = Decimal(position.quantity)
            initial_margin = Decimal(position.initial_margin)
            open_fee = Decimal(position.open_fee)
            funding_fees = Decimal(position.funding_fees_paid)
            
            # Close fee
            close_fee = PnLCalculator.calculate_trading_fee(exit_price, quantity, is_maker=False)
            
            # Realized PnL
            realized_pnl = PnLCalculator.calculate_realized_pnl(
                position.side,
                entry_price,
                exit_price,
                quantity,
                open_fee,
                close_fee,
                funding_fees
            )
            
            # Update position
            position.status = PositionStatus.CLOSED
            position.realized_pnl = str(realized_pnl)
            position.unrealized_pnl = "0"
            position.close_fee = str(close_fee)
            position.closed_at = datetime.now(timezone.utc)
            
            # Update user balance (if not demo)
            if not position.is_demo:
                user_balance = Decimal(user.balance_usdt)
                
                # Return initial margin + realized PnL - close fee
                balance_change = initial_margin + realized_pnl - close_fee
                new_balance = user_balance + balance_change
                
                user.balance_usdt = str(new_balance)
                
                # Ledger entries
                self.db.add(FuturesLedger(
                    user_id=user.id,
                    position_id=position.id,
                    entry_type=FuturesLedgerType.MARGIN_RELEASED,
                    asset="USDT",
                    amount=str(initial_margin),
                    balance_after=str(new_balance),
                    related_price=str(exit_price),
                    metadata=json.dumps({"reason": trigger_type})
                ))
                
                self.db.add(FuturesLedger(
                    user_id=user.id,
                    position_id=position.id,
                    entry_type=FuturesLedgerType.REALIZED_PNL,
                    asset="USDT",
                    amount=str(realized_pnl),
                    balance_after=str(new_balance),
                    related_price=str(exit_price)
                ))
                
                self.db.add(FuturesLedger(
                    user_id=user.id,
                    position_id=position.id,
                    entry_type=FuturesLedgerType.TRADING_FEE,
                    asset="USDT",
                    amount=str(-close_fee),
                    balance_after=str(new_balance),
                    related_price=str(exit_price)
                ))
            
            self.db.commit()
            
            logger.info(
                f"{trigger_type.upper()} TRIGGERED: User {user.username} | Position #{position.id} | "
                f"{position.symbol} {position.side.value} | "
                f"Entry: {entry_price} | Exit: {exit_price} | PnL: {realized_pnl} USDT"
            )
            
            return {
                "position_id": position.id,
                "trigger_type": trigger_type,
                "symbol": position.symbol,
                "side": position.side.value,
                "entry_price": str(entry_price),
                "exit_price": str(exit_price),
                "quantity": str(quantity),
                "realized_pnl": str(realized_pnl),
                "close_fee": str(close_fee),
                "timestamp": position.closed_at.isoformat()
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"TP/SL close failed for position {position.id}: {e}")
            raise
    
    def check_all_positions(self, symbol: str, mark_price: Decimal) -> list:
        """
        Check all open positions for TP/SL triggers
        
        Called by price feed on each tick
        
        Returns:
            List of TP/SL trigger events
        """
        triggers = []
        
        try:
            positions = self.db.query(FuturesPosition).filter(
                FuturesPosition.symbol == symbol,
                FuturesPosition.status == PositionStatus.OPEN
            ).all()
            
            for position in positions:
                trigger_event = self.check_tpsl_triggers(position, mark_price)
                
                if trigger_event:
                    try:
                        close_event = self.execute_tpsl_close(
                            position,
                            trigger_event["type"],
                            trigger_event["trigger_price"]
                        )
                        triggers.append(close_event)
                    except Exception as e:
                        logger.error(f"Failed to execute TP/SL for position {position.id}: {e}")
            
            return triggers
            
        except Exception as e:
            logger.error(f"Error checking TP/SL for {symbol}: {e}")
            return []