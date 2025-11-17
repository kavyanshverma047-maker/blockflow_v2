"""
Futures Price Feed - Real-time Market Data
=========================================
Simulates realistic price movements for demo mode.
"""

import asyncio
import random
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Optional
from loguru import logger

class PriceFeed:
    """Real-time price feed simulator"""
    
    def __init__(self):
        # Base prices
        self.prices = {
            "BTCUSDT": Decimal("95000.00"),
            "ETHUSDT": Decimal("3200.00")
        }
        
        # Price volatility (% per tick)
        self.volatility = {
            "BTCUSDT": Decimal("0.0005"),  # 0.05% per tick
            "ETHUSDT": Decimal("0.0008")   # 0.08% per tick
        }
        
        # Funding rates (8h rate)
        self.funding_rates = {
            "BTCUSDT": Decimal("0.0001"),  # 0.01%
            "ETHUSDT": Decimal("0.0001")
        }
        
        # Tick interval (milliseconds)
        self.tick_interval = 200  # 200ms = 5 ticks/second
        
        # Subscribers
        self.subscribers = []
    
    def subscribe(self, callback):
        """Subscribe to price updates"""
        self.subscribers.append(callback)
    
    async def emit(self, data: Dict):
        """Emit price update to all subscribers"""
        for callback in self.subscribers:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Price feed emit error: {e}")
    
    def generate_price_tick(self, symbol: str) -> Decimal:
        """
        Generate next price using random walk
        
        Formula: New Price = Current Price × (1 + Random(-volatility, +volatility))
        """
        current_price = self.prices[symbol]
        volatility = self.volatility[symbol]
        
        # Random walk: -volatility to +volatility
        change_percent = Decimal(str(random.uniform(
            -float(volatility),
            float(volatility)
        )))
        
        new_price = current_price * (1 + change_percent)
        
        # Ensure price stays positive and within reasonable bounds
        new_price = max(new_price, current_price * Decimal("0.95"))  # Max 5% down per tick
        new_price = min(new_price, current_price * Decimal("1.05"))  # Max 5% up per tick
        
        self.prices[symbol] = new_price
        return new_price.quantize(Decimal("0.01"))
    
    async def start(self):
        """Start price feed loop"""
        logger.info("Price feed started")
        
        while True:
            try:
                for symbol in self.prices.keys():
                    mark_price = self.generate_price_tick(symbol)
                    
                    # Emit price update
                    await self.emit({
                        "type": "price_tick",
                        "symbol": symbol,
                        "mark_price": str(mark_price),
                        "funding_rate": str(self.funding_rates[symbol]),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                # Wait for next tick
                await asyncio.sleep(self.tick_interval / 1000)
                
            except Exception as e:
                logger.error(f"Price feed error: {e}")
                await asyncio.sleep(1)
    
    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current mark price"""
        return self.prices.get(symbol)
    
    def set_price(self, symbol: str, price: Decimal):
        """Manually set price (for testing)"""
        self.prices[symbol] = price
        logger.info(f"Price manually set: {symbol} = {price}")

# Global price feed instance
price_feed = PriceFeed()