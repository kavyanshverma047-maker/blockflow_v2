from logging.config import fileConfig
from alembic import context
from sqlalchemy import create_engine, pool
import os
import sys

# Add base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Load project environment variables
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------
# USE ALEMBIC.INI URL ALWAYS (DO NOT OVERRIDE IT)
# -------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Read URL from alembic.ini
DATABASE_URL = config.get_main_option("sqlalchemy.url")

# If running locally and DB is sqlite → fix path
if DATABASE_URL.startswith("sqlite:///") and not DATABASE_URL.startswith("sqlite:///./"):
    DATABASE_URL = DATABASE_URL.replace("sqlite:///", "sqlite:///./")

print(f"🔧 Alembic using DB → {DATABASE_URL}")

# Import metadata AFTER path fixed
from app.db.base import Base
target_metadata = Base.metadata


# -------------------------------------------------------------
# OFFLINE MODE
# -------------------------------------------------------------
def run_migrations_offline():
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# -------------------------------------------------------------
# ONLINE MODE
# -------------------------------------------------------------
def run_migrations_online():
    connectable = create_engine(
        DATABASE_URL,
        poolclass=pool.NullPool,
        connect_args={"check_same_thread": False}
            if DATABASE_URL.startswith("sqlite") else {},
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


# -------------------------------------------------------------
# ENTRY
# -------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
