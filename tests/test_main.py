# tests/test_main.py
"""
Blockflow Exchange - Unit Tests
Run: pytest tests/test_main.py -v
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

# Import from main.py
try:
    from main import (
        calculate_tds, get_quarter_code, settings,
        hash_password, verify_password, create_access_token, decode_token
    )
except ImportError:
    from app.main import (
        calculate_tds, get_quarter_code, settings,
        hash_password, verify_password, create_access_token, decode_token
    )

class TestTDSCalculation:
    """Test TDS calculation logic"""
    
    def test_tds_below_threshold(self):
        """TDS should not apply below threshold"""
        result = calculate_tds(
            trade_value_crypto=Decimal("0.001"),
            price_inr=Decimal("1000000"),  # ₹10L per BTC
            user_ytd_trades=Decimal("0"),
            is_business=False
        )
        
        assert result["tds_applicable"] == False
        assert result["tds_inr"] == "0.00"
        assert result["threshold_exceeded"] == False
    
    def test_tds_above_threshold_individual(self):
        """TDS should apply above ₹50K threshold for individuals"""
        result = calculate_tds(
            trade_value_crypto=Decimal("0.01"),
            price_inr=Decimal("8000000"),  # ₹80L per BTC
            user_ytd_trades=Decimal("0"),
            is_business=False
        )
        
        assert result["tds_applicable"] == True
        assert Decimal(result["tds_inr"]) == Decimal("800.00")  # 1% of 80,000
        assert result["threshold_exceeded"] == True
    
    def test_tds_business_threshold(self):
        """Business entities have ₹10K threshold"""
        result = calculate_tds(
            trade_value_crypto=Decimal("0.002"),
            price_inr=Decimal("6000000"),  # ₹60L per BTC
            user_ytd_trades=Decimal("0"),
            is_business=True
        )
        
        assert result["tds_applicable"] == True
        # 0.002 * 6000000 = 12000, which is > 10000 threshold
    
    def test_tds_ytd_accumulation(self):
        """YTD accumulation should trigger TDS"""
        result = calculate_tds(
            trade_value_crypto=Decimal("0.0005"),
            price_inr=Decimal("10000000"),  # ₹1Cr per BTC
            user_ytd_trades=Decimal("49000"),  # Already at ₹49K
            is_business=False
        )
        
        # New trade: 0.0005 * 10000000 = 5000
        # Total: 49000 + 5000 = 54000 > 50000 threshold
        assert result["tds_applicable"] == True
        assert Decimal(result["tds_inr"]) == Decimal("50.00")  # 1% of 5000
    
    def test_tds_rate_correct(self):
        """Verify TDS rate is exactly 1%"""
        result = calculate_tds(
            trade_value_crypto=Decimal("1.0"),
            price_inr=Decimal("8000000"),
            user_ytd_trades=Decimal("0"),
            is_business=False
        )
        
        gross = Decimal("8000000")
        tds = Decimal(result["tds_inr"])
        rate = (tds / gross * 100).quantize(Decimal("0.01"))
        
        assert rate == Decimal("1.00")

class TestQuarterCalculation:
    """Test fiscal quarter calculation"""
    
    def test_quarter_format(self):
        """Quarter code should be in correct format"""
        quarter = get_quarter_code()
        
        assert quarter.startswith("Q")
        assert "FY" in quarter
        assert len(quarter) == 7  # e.g., "Q3-FY25"
    
    def test_quarter_valid_q(self):
        """Quarter should be Q1, Q2, Q3, or Q4"""
        quarter = get_quarter_code()
        q_part = quarter.split("-")[0]
        
        assert q_part in ["Q1", "Q2", "Q3", "Q4"]

class TestPasswordHashing:
    """Test password security"""
    
    def test_password_hashing(self):
        """Password should hash correctly"""
        password = "test_password_123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long
        assert hashed.startswith("$2b$")  # bcrypt prefix
    
    def test_password_verification(self):
        """Password verification should work"""
        password = "correct_password"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) == True
        assert verify_password("wrong_password", hashed) == False

class TestJWTTokens:
    """Test JWT token creation and validation"""
    
    def test_token_creation(self):
        """JWT token should be created"""
        data = {
            "user_id": 123,
            "username": "testuser",
            "role": "user"
        }
        
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 50
    
    def test_token_decoding(self):
        """JWT token should decode correctly"""
        data = {
            "user_id": 456,
            "username": "testuser2",
            "role": "admin"
        }
        
        token = create_access_token(data)
        decoded = decode_token(token)
        
        assert decoded.user_id == 456
        assert decoded.username == "testuser2"
        assert decoded.role == "admin"

class TestSettings:
    """Test application settings"""
    
    def test_tds_settings(self):
        """TDS settings should be correct"""
        assert settings.TDS_RATE == Decimal("0.01")
        assert settings.TDS_THRESHOLD_INDIVIDUAL == Decimal("50000")
        assert settings.TDS_THRESHOLD_BUSINESS == Decimal("10000")
    
    def test_fee_settings(self):
        """Trading fee settings should be correct"""
        assert settings.MAKER_FEE == Decimal("0.0004")
        assert settings.TAKER_FEE == Decimal("0.001")
    
    def test_jwt_settings(self):
        """JWT settings should be configured"""
        assert settings.JWT_ALGORITHM == "HS256"
        assert settings.JWT_EXPIRY_MINUTES > 0
        assert len(settings.JWT_SECRET) >= 32

# Run with: pytest tests/test_main.py -v --cov=. --cov-report=html