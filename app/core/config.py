from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    ENV: str = Field(default="production")
    DEMO_MODE: bool = Field(default=True)
    DEBUG: bool = Field(default=False)

    DATABASE_URL: str

    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60

    ALLOWED_ORIGINS: str  # keep string, parse later

    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40

    TDS_RATE: float = 0.01
    TDS_THRESHOLD_INDIVIDUAL: int = 50000
    TDS_THRESHOLD_BUSINESS: int = 10000

    MAKER_FEE: float = 0.0004
    TAKER_FEE: float = 0.001

    PORT: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"   # <-- IMPORTANT (fixes your error)

settings = Settings()
