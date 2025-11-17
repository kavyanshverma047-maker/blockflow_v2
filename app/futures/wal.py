# app/futures/wal.py

import json
import hashlib
import asyncio
from typing import Dict, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from loguru import logger

from app.futures.models import FuturesWAL


class WALSystem:
    """Write-Ahead Log for rate-limited writes"""

    def __init__(self, db: Session):
        self.db = db
        self.max_retries = 3

    @staticmethod
    def generate_idempotency_key(operation: str, payload: Dict[str, Any]) -> str:
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(f"{operation}:{payload_str}".encode()).hexdigest()

    def write_to_wal(self, operation: str, payload: Dict[str, Any]) -> str:
        key = self.generate_idempotency_key(operation, payload)

        try:
            existing = self.db.query(FuturesWAL).filter(
                FuturesWAL.idempotency_key == key
            ).first()

            if existing:
                return key

            wal = FuturesWAL(
                idempotency_key=key,
                operation=operation,
                payload=json.dumps(payload),
                status="PENDING"
            )
            self.db.add(wal)
            self.db.commit()
            return key

        except Exception as e:
            self.db.rollback()
            logger.error(f"WAL write failed: {e}")
            raise


class WALReplayWorker:
    """Async background worker"""

    def __init__(self, db: Session):
        self.db = db
        self.wal = WALSystem(db)

    async def start(self):
        logger.info("WAL Replay Worker Started")

        while True:
            try:
                pending = self.db.query(FuturesWAL).filter(
                    FuturesWAL.status == "PENDING",
                    FuturesWAL.retry_count < 3
                ).order_by(FuturesWAL.created_at).limit(10).all()

                for entry in pending:
                    try:
                        self.wal.replay_wal_entry(entry)
                    except Exception as e:
                        logger.error(f"Replay failed: {e}")
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"WAL worker main loop error: {e}")

            await asyncio.sleep(20)


# ❗ MAIN.PY WILL CREATE THE INSTANCE (not here)
# Keep this exactly as is:
wal_replay_worker = WALReplayWorker
