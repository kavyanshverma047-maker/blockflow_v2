# demo_trading.py
"""
Blockflow Exchange - Automated Trading Demo
Run: python demo_trading.py
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

class BlockflowDemo:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.buyer_token = None
        self.seller_token = None
        
    def print_section(self, title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/api/auth/register", json={
            "username": username,
            "email": email,
            "password": password
        })
        return response.json()
    
    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/api/auth/login", json={
            "username": username,
            "password": password
        })
        return response.json()
    
    def get_balances(self, token: str) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/wallet/balances",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
    
    def place_order(self, token: str, symbol: str, side: str, price: float, amount: float) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/trading/order",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "symbol": symbol,
                "side": side,
                "order_type": "limit",
                "price": price,
                "amount": amount
            }
        )
        return response.json()
    
    def get_tax_summary(self, token: str) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/tax/summary",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
    
    def get_public_stats(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/public/stats")
        return response.json()
    
    def run_demo(self):
        self.print_section("🚀 Blockflow v3.5.1 - Automated Trading Demo")
        
        # Check demo status
        try:
            demo_status = requests.get(f"{self.base_url}/api/demo-status").json()
            if demo_status.get("demo_mode"):
                print("✅ Demo mode: ACTIVE")
            else:
                print("⚠️  Warning: Live trading mode detected!")
        except Exception as e:
            print(f"❌ Could not connect to API: {e}")
            return
        
        # Create users
        self.print_section("1️⃣  Creating Users")
        timestamp = int(time.time())
        
        buyer = self.register_user(
            f"buyer_{timestamp}",
            f"buyer_{timestamp}@demo.test",
            "password123"
        )
        self.buyer_token = buyer["token"]
        print(f"✅ Buyer: {buyer['user']['username']}")
        
        seller = self.register_user(
            f"seller_{timestamp}",
            f"seller_{timestamp}@demo.test",
            "password123"
        )
        self.seller_token = seller["token"]
        print(f"✅ Seller: {seller['user']['username']}")
        
        # Initial balances
        self.print_section("2️⃣  Initial Balances")
        buyer_bal = self.get_balances(self.buyer_token)
        print(f"Buyer:")
        print(f"  USDT: {buyer_bal['balances']['USDT']['available']} (available)")
        print(f"  BTC:  {buyer_bal['balances']['BTC']['available']}")
        
        seller_bal = self.get_balances(self.seller_token)
        print(f"Seller:")
        print(f"  USDT: {seller_bal['balances']['USDT']['available']}")
        print(f"  BTC:  {seller_bal['balances']['BTC']['available']} (available)")
        
        # Seller places sell order
        self.print_section("3️⃣  Seller Places SELL Order")
        sell_order = self.place_order(self.seller_token, "BTCUSDT", "sell", 95000, 0.01)
        print(f"✅ Order #{sell_order['order']['id']}")
        print(f"   Symbol: {sell_order['order']['symbol']}")
        print(f"   Side: {sell_order['order']['side']}")
        print(f"   Price: ${sell_order['order']['price']}")
        print(f"   Amount: {sell_order['order']['amount']} BTC")
        print(f"   Status: {sell_order['order']['status']}")
        
        time.sleep(2)
        
        # Buyer places buy order (matching)
        self.print_section("4️⃣  Buyer Places BUY Order (Matching)")
        buy_order = self.place_order(self.buyer_token, "BTCUSDT", "buy", 95000, 0.01)
        print(f"✅ Order #{buy_order['order']['id']}")
        print(f"   Status: {buy_order['order']['status']}")
        print(f"   Filled: {buy_order['order']['filled']} BTC")
        
        if buy_order.get('trades'):
            print(f"\n🎉 TRADES EXECUTED: {len(buy_order['trades'])}")
            for trade in buy_order['trades']:
                print(f"   Trade #{trade['id']}: {trade['amount']} BTC @ ${trade['price']}")
        else:
            print("⚠️  No trades executed (orders may be in book)")
        
        time.sleep(1)
        
        # Updated balances
        self.print_section("5️⃣  Updated Balances (After Trade)")
        buyer_bal_after = self.get_balances(self.buyer_token)
        print(f"Buyer:")
        print(f"  USDT: {buyer_bal['balances']['USDT']['available']} → {buyer_bal_after['balances']['USDT']['available']} (decreased)")
        print(f"  BTC:  {buyer_bal['balances']['BTC']['available']} → {buyer_bal_after['balances']['BTC']['available']} (increased)")
        
        seller_bal_after = self.get_balances(self.seller_token)
        print(f"Seller:")
        print(f"  USDT: {seller_bal['balances']['USDT']['available']} → {seller_bal_after['balances']['USDT']['available']} (increased, after fees)")
        print(f"  BTC:  {seller_bal['balances']['BTC']['available']} → {seller_bal_after['balances']['BTC']['available']} (decreased)")
        
        # TDS Summary
        self.print_section("6️⃣  TDS Summary (Seller)")
        try:
            tax = self.get_tax_summary(self.seller_token)
            print(f"Quarter: {tax['quarter']}")
            print(f"Total Trades: {tax['summary']['total_trades']}")
            print(f"Gross Value: ₹{tax['summary']['gross_value_inr']}")
            print(f"TDS Deducted: ₹{tax['summary']['total_tds_deducted']}")
            print(f"Net Value: ₹{tax['summary']['net_value_inr']}")
        except Exception as e:
            print(f"⚠️  Tax summary not available: {e}")
        
        # Public Stats
        self.print_section("7️⃣  Public Platform Statistics")
        try:
            stats = self.get_public_stats()
            print(f"Total Users: {stats['users']}")
            print(f"Total Trades: {stats['trades']}")
            print(f"24h Volume: ${stats['volume_24h_usdt']} USDT")
            print(f"TDS Collected: ₹{stats['tds_collected_inr']}")
            print(f"Timestamp: {stats['timestamp']}")
        except Exception as e:
            print(f"⚠️  Stats not available: {e}")
        
        self.print_section("✅ Demo Completed Successfully!")
        print("\n💡 Next Steps:")
        print("   - Check WebSocket feed: ws://localhost:8000/ws/market/BTCUSDT")
        print("   - View admin stats: GET /api/admin/stats")
        print("   - Download tax report: GET /api/tax/download")
        print("")

if __name__ == "__main__":
    demo = BlockflowDemo()
    try:
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()