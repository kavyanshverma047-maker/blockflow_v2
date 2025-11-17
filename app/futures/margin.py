"""
Futures Margin Calculator - Cross Margin System
==============================================
Calculate initial margin, maintenance margin, and liquidation prices.

Margin System: Cross Margin (shared across all positions)
- Initial Margin Rate = 1 / Leverage
- Maintenance Margin Rate = 0.5% (for most assets)
"""

from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, Any, Tuple
from app.futures.models import PositionSide

class MarginCalculator:
    """USDM Cross Margin Calculator"""
    
    # Maintenance margin rates (varies by notional value tiers)
    MAINTENANCE_MARGIN_RATE = Decimal("0.005")  # 0.5% base rate
    
    # Liquidation fee
    LIQUIDATION_FEE_RATE = Decimal("0.004")  # 0.4%
    
    @staticmethod
    def calculate_initial_margin(
        price: Decimal,
        quantity: Decimal,
        leverage: int
    ) -> Decimal:
        """
        Calculate initial margin required to open position
        
        Formula: Initial Margin = (Price × Quantity) / Leverage
        
        Args:
            price: Entry price
            quantity: Position size
            leverage: Leverage (1-125x)
            
        Returns:
            Required initial margin in USDT
        """
        notional_value = price * quantity
        initial_margin = notional_value / Decimal(str(leverage))
        
        return initial_margin.quantize(Decimal("0.00000001"), rounding=ROUND_UP)
    
    @staticmethod
    def calculate_maintenance_margin(
        price: Decimal,
        quantity: Decimal
    ) -> Decimal:
        """
        Calculate maintenance margin (minimum to avoid liquidation)
        
        Formula: Maintenance Margin = Notional Value × Maintenance Rate
        """
        notional_value = price * quantity
        maintenance_margin = notional_value * MarginCalculator.MAINTENANCE_MARGIN_RATE
        
        return maintenance_margin.quantize(Decimal("0.00000001"), rounding=ROUND_UP)
    
    @staticmethod
    def calculate_liquidation_price(
        side: PositionSide,
        entry_price: Decimal,
        quantity: Decimal,
        leverage: int,
        initial_margin: Decimal
    ) -> Decimal:
        """
        Calculate liquidation price
        
        Long: Liq Price = Entry Price × (1 - 1/Leverage + MM Rate + Liq Fee Rate)
        Short: Liq Price = Entry Price × (1 + 1/Leverage - MM Rate - Liq Fee Rate)
        
        Simplified formula that accounts for:
        - Initial margin
        - Maintenance margin requirement
        - Liquidation fee
        """
        notional_value = entry_price * quantity
        maintenance_margin = MarginCalculator.calculate_maintenance_margin(entry_price, quantity)
        liquidation_fee = notional_value * MarginCalculator.LIQUIDATION_FEE_RATE
        
        if side == PositionSide.LONG:
            # Loss that triggers liquidation
            max_loss = initial_margin - maintenance_margin - liquidation_fee
            liquidation_price = entry_price - (max_loss / quantity)
        else:  # SHORT
            max_loss = initial_margin - maintenance_margin - liquidation_fee
            liquidation_price = entry_price + (max_loss / quantity)
        
        # Ensure liquidation price is positive
        liquidation_price = max(liquidation_price, Decimal("0.01"))
        
        return liquidation_price.quantize(Decimal("0.01"), rounding=ROUND_DOWN if side == PositionSide.LONG else ROUND_UP)
    
    @staticmethod
    def calculate_max_position_size(
        price: Decimal,
        available_balance: Decimal,
        leverage: int
    ) -> Decimal:
        """
        Calculate maximum position size user can open
        
        Max Quantity = (Available Balance × Leverage) / Price
        """
        max_notional = available_balance * Decimal(str(leverage))
        max_quantity = max_notional / price
        
        return max_quantity.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    
    @staticmethod
    def check_margin_ratio(
        total_balance: Decimal,
        total_margin_used: Decimal,
        total_unrealized_pnl: Decimal
    ) -> Dict[str, Any]:
        """
        Check account margin ratio and risk status
        
        Margin Ratio = (Margin Used) / (Balance + Unrealized PnL)
        
        Risk Levels:
        - < 50%: Safe
        - 50-80%: Warning
        - 80-100%: Danger (approaching liquidation)
        - >= 100%: Liquidation triggered
        
        Returns:
            {
                "margin_ratio": Decimal,
                "risk_level": str,
                "available_margin": Decimal
            }
        """
        account_value = total_balance + total_unrealized_pnl
        
        if account_value <= 0:
            return {
                "margin_ratio": Decimal("100"),
                "risk_level": "LIQUIDATION",
                "available_margin": Decimal("0")
            }
        
        margin_ratio = (total_margin_used / account_value * Decimal("100")).quantize(Decimal("0.01"))
        available_margin = (account_value - total_margin_used).quantize(Decimal("0.00000001"))
        
        # Determine risk level
        if margin_ratio >= 100:
            risk_level = "LIQUIDATION"
        elif margin_ratio >= 80:
            risk_level = "DANGER"
        elif margin_ratio >= 50:
            risk_level = "WARNING"
        else:
            risk_level = "SAFE"
        
        return {
            "margin_ratio": margin_ratio,
            "risk_level": risk_level,
            "available_margin": max(available_margin, Decimal("0"))
        }
    
    @staticmethod
    def validate_order_margin(
        user_balance: Decimal,
        existing_margin_used: Decimal,
        new_order_margin: Decimal
    ) -> Tuple[bool, str]:
        """
        Validate if user has sufficient margin for new order
        
        Returns:
            (is_valid, error_message)
        """
        required_margin = existing_margin_used + new_order_margin
        
        if required_margin > user_balance:
            shortage = required_margin - user_balance
            return False, f"Insufficient margin. Need {shortage} USDT more."
        
        # Check if new order would put account at risk
        margin_ratio = (required_margin / user_balance * Decimal("100")).quantize(Decimal("0.01"))
        
        if margin_ratio > 95:
            return False, "Order would exceed safe margin ratio (>95%)."
        
        return True, ""