"""
Futures Demo Engine - In-Memory Simulation
=========================================
Handles all futures operations in demo mode without DB writes.
"""

from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger
import asyncio

from app.futures.models import PositionSide, PositionStatus, FuturesOrderType, FuturesOrderSide
from app.futures.margin import MarginCalculator
from app.futures.pnl import PnLCalculator
from app.futures.liquidation import LiquidationEngine
from app.futures.tpsl import TPSLManager

class DemoPosition:
    """In-memory position for demo mode"""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get("id")
        self.user_id = kwargs.get("user_id")
        self.symbol = kwargs.get("symbol")
        self.side = kwargs.get("side")
        self.status = kwargs.get("status", PositionStatus.OPEN)
        
        self.entry_price = Decimal(str(kwargs.get("entry_price")))
        self.quantity = Decimal(str(kwargs.get("quantity")))
        self.leverage = kwargs.get("leverage")
        
        self.initial_margin = Decimal(str(kwargs.get("initial_margin")))
        self.maintenance_margin = Decimal(str(kwargs.get("maintenance_margin")))
        
        self.unrealized_pnl = Decimal("0")
        self.realized_pnl = Decimal("0")
        
        self.liquidation_price = Decimal(str(kwargs.get("liquidation_price")))
        self.take_profit_price = Decimal(str(kwargs.get("take_profit_price", "0"))) if kwargs.get("take_profit_price") else None
        self.stop_loss_price = Decimal(str(kwargs.get("stop_loss_price", "0"))) if kwargs.get("stop_loss_price") else None
        
        self.open_fee = Decimal(str(kwargs.get("open_fee", "0")))
        self.close_fee = Decimal("0")
        self.funding_fees_paid = Decimal("0")
        
        self.is_demo = True
        self.opened_at = kwargs.get("opened_at", datetime.now(timezone.utc))
        self.closed_at = None
        self.last_updated = datetime.now(timezone.utc)
    
    def to_dict(self):
        """Convert to dict for API response"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, PositionSide) else self.side,
            "status": self.status.value if isinstance(self.status, PositionStatus) else self.status,
            "entry_price": str(self.entry_price),
            "quantity": str(self.quantity),
            "leverage": self.leverage,
            "initial_margin": str(self.initial_margin),
            "maintenance_margin": str(self.maintenance_margin),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "liquidation_price": str(self.liquidation_price),
            "take_profit_price": str(self.take_profit_price) if self.take_profit_price else None,
            "stop_loss_price": str(self.stop_loss_price) if self.stop_loss_price else None,
            "open_fee": str(self.open_fee),
            "funding_fees_paid": str(self.funding_fees_paid),
            "is_demo": True,
            "opened_at": self.opened_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }

class DemoEngine:
    """In-memory futures engine for demo mode"""
    
    def __init__(self):
        self.positions: Dict[int, DemoPosition] = {}  # position_id -> DemoPosition
        self.user_positions: Dict[int, List[int]] = {}  # user_id -> [position_ids]
        self.next_position_id = 1
        
        # Demo balances (separate from DB)
        self.demo_balances: Dict[int, Decimal] = {}  # user_id -> USDT balance
        
        logger.info("Demo engine initialized")
    
    def get_user_balance(self, user_id: int) -> Decimal:
        """Get user's demo balance"""
        if user_id not in self.demo_balances:
            self.demo_balances[user_id] = Decimal("10000.00")  # Default demo balance
        return self.demo_balances[user_id]
    
    def update_user_balance(self, user_id: int, amount: Decimal):
        """Update user's demo balance"""
        current = self.get_user_balance(user_id)
        self.demo_balances[user_id] = current + amount
    
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
    ) -> DemoPosition:
        """Create new demo position"""
        
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
        user_balance = self.get_user_balance(user_id)
        
        if user_balance < required_margin:
            raise ValueError(f"Insufficient balance. Required: {required_margin} USDT")
        
        # Deduct margin
        self.update_user_balance(user_id, -required_margin)
        
        # Create position
        position = DemoPosition(
            id=self.next_position_id,
            user_id=user_id,
            symbol=symbol,
            side=position_side,
            entry_price=price,
            quantity=quantity,
            leverage=leverage,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            liquidation_price=liquidation_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            open_fee=open_fee
        )
        
        self.positions[self.next_position_id] = position
        
        if user_id not in self.user_positions:
            self.user_positions[user_id] = []
        self.user_positions[user_id].append(self.next_position_id)
        
        self.next_position_id += 1
        
        logger.info(
            f"Demo position created: #{position.id} | User {user_id} | "
            f"{symbol} {position_side.value} {quantity} @ {price} | {leverage}x"
        )
        
        return position
    
    def update_position_pnl(self, position: DemoPosition, mark_price: Decimal):
        """Update position's unrealized PnL"""
        position.unrealized_pnl = PnLCalculator.calculate_unrealized_pnl(
            position.side,
            position.entry_price,
            mark_price,
            position.quantity
        )
        position.last_updated = datetime.now(timezone.utc)
    
    async def close_position(
        self,
        position_id: int,
        exit_price: Decimal,
        quantity: Optional[Decimal] = None
    ) -> Dict:
        """Close demo position"""
        
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        if position.status != PositionStatus.OPEN:
            raise ValueError(f"Position {position_id} is not open")
        
        # Full or partial close
        close_quantity = quantity if quantity else position.quantity
        
        if close_quantity > position.quantity:
            raise ValueError("Close quantity exceeds position size")
        
        # Calculate close fee
        close_fee = PnLCalculator.calculate_trading_fee(exit_price, close_quantity, is_maker=False)
        
        # Calculate realized PnL
        realized_pnl = PnLCalculator.calculate_realized_pnl(
            position.side,
            position.entry_price,
            exit_price,
            close_quantity,
            position.open_fee,
            close_fee,
            position.funding_fees_paid
        )
        
        # Partial close
        if close_quantity < position.quantity:
            # Reduce position size
            remaining_quantity = position.quantity - close_quantity
            remaining_margin_ratio = remaining_quantity / position.quantity
            
            position.quantity = remaining_quantity
            position.initial_margin = position.initial_margin * remaining_margin_ratio
            position.realized_pnl += realized_pnl
            
            # Return margin + PnL
            self.update_user_balance(
                position.user_id,
                position.initial_margin * (Decimal("1") - remaining_margin_ratio) + realized_pnl
            )
            
        else:
            # Full close
            position.status = PositionStatus.CLOSED
            position.realized_pnl = realized_pnl
            position.close_fee = close_fee
            position.closed_at = datetime.now(timezone.utc)
            
            # Return margin + PnL - close fee
            self.update_user_balance(
                position.user_id,
                position.initial_margin + realized_pnl - close_fee
            )
        
        logger.info(
            f"Demo position closed: #{position_id} | "
            f"Exit: {exit_price} | PnL: {realized_pnl} USDT"
        )
        
        return {
            "position_id": position_id,
            "closed_quantity": str(close_quantity),
            "exit_price": str(exit_price),
            "realized_pnl": str(realized_pnl),
            "close_fee": str(close_fee)
        }
    
    def modify_tpsl(
        self,
        position_id: int,
        take_profit_price: Optional[Decimal] = None,
        stop_loss_price: Optional[Decimal] = None
    ) -> bool:
        """Modify TP/SL for demo position"""
        
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        if take_profit_price is not None:
            position.take_profit_price = take_profit_price
        
        if stop_loss_price is not None:
            position.stop_loss_price = stop_loss_price
        
        logger.info(f"Demo TP/SL modified: #{position_id} | TP: {take_profit_price} | SL: {stop_loss_price}")
        
        return True
    
    def get_user_positions(self, user_id: int, status: Optional[PositionStatus] = None) -> List[DemoPosition]:
        """Get all positions for user"""
        position_ids = self.user_positions.get(user_id, [])
        positions = [self.positions[pid] for pid in position_ids if pid in self.positions]
        
        if status:
            positions = [p for p in positions if p.status == status]
        
        return positions
    
    def get_position(self, position_id: int) -> Optional[DemoPosition]:
        """Get specific position"""
        return self.positions.get(position_id)
    
    async def check_tpsl_triggers(self, symbol: str, mark_price: Decimal) -> List[Dict]:
        """Check all positions for TP/SL triggers"""
        triggers = []
        
        for position in self.positions.values():
            if position.symbol != symbol or position.status != PositionStatus.OPEN:
                continue
            
            triggered = False
            trigger_type = None
            
            # Check TP
            if position.take_profit_price:
                if position.side == PositionSide.LONG and mark_price >= position.take_profit_price:
                    triggered = True
                    trigger_type = "take_profit"
                elif position.side == PositionSide.SHORT and mark_price <= position.take_profit_price:
                    triggered = True
                    trigger_type = "take_profit"
            
            # Check SL
            if not triggered and position.stop_loss_price:
                if position.side == PositionSide.LONG and mark_price <= position.stop_loss_price:
                    triggered = True
                    trigger_type = "stop_loss"
                elif position.side == PositionSide.SHORT and mark_price >= position.stop_loss_price:
                    triggered = True
                    trigger_type = "stop_loss"
            
            if triggered:
                result = await self.close_position(position.id, mark_price)
                triggers.append({
                    "type": trigger_type,
                    "position_id": position.id,
                    **result
                })
        
        return triggers
    
    async def check_liquidations(self, symbol: str, mark_price: Decimal) -> List[Dict]:
        """Check all positions for liquidation"""
        liquidations = []
        
        for position in self.positions.values():
            if position.symbol != symbol or position.status != PositionStatus.OPEN:
                continue
            
            should_liquidate = False
            
            if position.side == PositionSide.LONG:
                should_liquidate = mark_price <= position.liquidation_price
            else:  # SHORT
                should_liquidate = mark_price >= position.liquidation_price
            
            if should_liquidate:
                # Liquidate
                position.status = PositionStatus.LIQUIDATED
                position.closed_at = datetime.now(timezone.utc)
                
                # User loses margin
                loss = position.initial_margin
                liquidation_fee = PnLCalculator.calculate_liquidation_fee(mark_price, position.quantity)
                
                liquidations.append({
                    "position_id": position.id,
                    "symbol": position.symbol,
                    "liquidation_price": str(mark_price),
                    "loss_amount": str(loss),
                    "liquidation_fee": str(liquidation_fee)
                })
                
                logger.warning(f"Demo liquidation: #{position.id} | Loss: {loss} USDT")
        
        return liquidations
    
    def get_account_summary(self, user_id: int) -> Dict:
        """Get account summary"""
        positions = self.get_user_positions(user_id, PositionStatus.OPEN)
        
        total_margin = sum(p.initial_margin for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        balance = self.get_user_balance(user_id)
        
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

# Global demo engine instance
demo_engine = DemoEngine()