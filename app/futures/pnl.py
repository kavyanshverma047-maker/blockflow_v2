"""
Futures PnL Calculator - Professional Grade
==========================================
Accurate PnL calculations for USDM (linear) perpetual futures.

Formulas based on Binance/Bybit standards:
- Long PnL = (Mark Price - Entry Price) × Quantity
- Short PnL = (Entry Price - Mark Price) × Quantity
- ROE = PnL / Initial Margin × 100%
"""

from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any
from app.futures.models import PositionSide

class PnLCalculator:
    """USDM Perpetual Futures PnL Calculator"""
    
    # Fee rates
    MAKER_FEE_RATE = Decimal("0.0002")  # 0.02%
    TAKER_FEE_RATE = Decimal("0.0005")  # 0.05%
    LIQUIDATION_FEE_RATE = Decimal("0.004")  # 0.4%
    
    @staticmethod
    def calculate_unrealized_pnl(
        side: PositionSide,
        entry_price: Decimal,
        mark_price: Decimal,
        quantity: Decimal
    ) -> Decimal:
        """
        Calculate unrealized PnL
        
        Args:
            side: LONG or SHORT
            entry_price: Position entry price
            mark_price: Current mark price
            quantity: Position size
            
        Returns:
            Unrealized PnL in USDT
        """
        if side == PositionSide.LONG:
            pnl = (mark_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - mark_price) * quantity
        
        return pnl.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    
    @staticmethod
    def calculate_realized_pnl(
        side: PositionSide,
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
        open_fee: Decimal,
        close_fee: Decimal,
        funding_fees: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Calculate realized PnL after closing position
        
        Formula: PnL = Price Difference × Quantity - Open Fee - Close Fee - Funding Fees
        """
        if side == PositionSide.LONG:
            price_pnl = (exit_price - entry_price) * quantity
        else:  # SHORT
            price_pnl = (entry_price - exit_price) * quantity
        
        realized_pnl = price_pnl - open_fee - close_fee - funding_fees
        
        return realized_pnl.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    
    @staticmethod
    def calculate_roe(pnl: Decimal, initial_margin: Decimal) -> Decimal:
        """
        Calculate Return on Equity (ROE)
        
        ROE = (PnL / Initial Margin) × 100%
        """
        if initial_margin == 0:
            return Decimal("0")
        
        roe = (pnl / initial_margin) * Decimal("100")
        return roe.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    
    @staticmethod
    def calculate_trading_fee(
        price: Decimal,
        quantity: Decimal,
        is_maker: bool = False
    ) -> Decimal:
        """Calculate trading fee (open or close)"""
        notional_value = price * quantity
        fee_rate = PnLCalculator.MAKER_FEE_RATE if is_maker else PnLCalculator.TAKER_FEE_RATE
        fee = notional_value * fee_rate
        
        return fee.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    
    @staticmethod
    def calculate_liquidation_fee(
        price: Decimal,
        quantity: Decimal
    ) -> Decimal:
        """Calculate liquidation fee"""
        notional_value = price * quantity
        fee = notional_value * PnLCalculator.LIQUIDATION_FEE_RATE
        
        return fee.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    
    @staticmethod
    def calculate_position_metrics(
        side: PositionSide,
        entry_price: Decimal,
        mark_price: Decimal,
        quantity: Decimal,
        initial_margin: Decimal,
        open_fee: Decimal = Decimal("0"),
        funding_fees: Decimal = Decimal("0")
    ) -> Dict[str, Any]:
        """
        Calculate all position metrics in one call
        
        Returns:
            {
                "unrealized_pnl": Decimal,
                "roe_percent": Decimal,
                "total_cost": Decimal,
                "breakeven_price": Decimal
            }
        """
        unrealized_pnl = PnLCalculator.calculate_unrealized_pnl(
            side, entry_price, mark_price, quantity
        )
        
        roe = PnLCalculator.calculate_roe(unrealized_pnl, initial_margin)
        
        total_cost = initial_margin + open_fee + funding_fees
        
        # Breakeven price calculation
        if side == PositionSide.LONG:
            breakeven_price = entry_price + (total_cost / quantity)
        else:  # SHORT
            breakeven_price = entry_price - (total_cost / quantity)
        
        return {
            "unrealized_pnl": unrealized_pnl,
            "roe_percent": roe,
            "total_cost": total_cost,
            "breakeven_price": breakeven_price.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        }
    
    @staticmethod
    def calculate_notional_value(price: Decimal, quantity: Decimal) -> Decimal:
        """Calculate notional value of position"""
        return (price * quantity).quantize(Decimal("0.01"), rounding=ROUND_DOWN)