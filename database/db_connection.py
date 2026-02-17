"""
Database Connection Manager
============================
Single point of access for the SQLAlchemy engine + session factory.

Usage
-----
    from database.db_connection import get_session, engine

    with get_session() as session:
        session.execute(...)

Integration point
-----------------
Imported by every module that touches the DB:
    pipeline/ingestion.py      – writes raw records
    pipeline/preprocessing.py  – reads raw, writes cleaned
    pipeline/retrain.py        – reads cleaned for training
    models/model_registry.py   – writes model metadata
    api/main.py                – reads predictions for the dashboard
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATABASE_URL, USE_SQLITE_FALLBACK, SQLITE_URL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Choose the correct connection string
# ---------------------------------------------------------------------------
_url = DATABASE_URL
if USE_SQLITE_FALLBACK:
    _url = SQLITE_URL
    logger.info("Using SQLite fallback: %s", _url)

# For SQLite, connect_args keeps the same connection across threads (Streamlit
# spins up multiple threads).  Ignored silently for PostgreSQL.
_connect_args = {"check_same_thread": False} if _url.startswith("sqlite") else {}

engine = create_engine(
    _url,
    pool_pre_ping=True,         # auto-reconnect stale connections
    echo=False,                 # set True for SQL debugging
    connect_args=_connect_args,
)

# Enable WAL mode for SQLite so readers don't block the writer
if _url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Base class for all ORM models – imported by database/schema.py
Base = declarative_base()


@contextmanager
def get_session():
    """Yield a transactional session that auto-closes and rolls back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """
    Create all tables that don't exist yet.
    Called once at app startup (api/main.py) or by the scheduler on first run.
    """
    # Import schema so all models are registered on Base.metadata
    from database import schema  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified / created.")
