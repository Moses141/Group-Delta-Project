"""
MedSupply AI - Centralized Configuration
=========================================
All environment-dependent settings live here. Values are loaded from .env
file via python-dotenv and fall back to sensible defaults for local dev.

Integration point: Every module (pipeline, database, API, models) imports
from this single source of truth so there's no scattered hardcoding.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (safe no-op if file doesn't exist)
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"           # ingested raw files land here
CLEAN_DATA_DIR = BASE_DIR / "data" / "cleaned"
MODEL_DIR = BASE_DIR / "trained_models"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for d in [DATA_DIR, CLEAN_DATA_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database (PostgreSQL)
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "medsupply")
DB_USER = os.getenv("DB_USER", "medsupply_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "medsupply_pass")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

# SQLite fallback for quick local testing when Postgres is not available
USE_SQLITE_FALLBACK = os.getenv("USE_SQLITE_FALLBACK", "true").lower() == "true"
SQLITE_URL = f"sqlite:///{BASE_DIR / 'medsupply.db'}"

# ---------------------------------------------------------------------------
# Google Sheets API
# ---------------------------------------------------------------------------
GOOGLE_SHEETS_CREDENTIALS_FILE = os.getenv(
    "GOOGLE_SHEETS_CREDENTIALS_FILE", str(BASE_DIR / "credentials" / "google_service_account.json")
)
# Comma-separated list of Google Sheet IDs to ingest
GOOGLE_SHEET_IDS = [
    sid.strip()
    for sid in os.getenv("GOOGLE_SHEET_IDS", "").split(",")
    if sid.strip()
]

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------
# Scheduler: cron-style expression for the weekly pipeline run
PIPELINE_CRON_DAY_OF_WEEK = os.getenv("PIPELINE_CRON_DAY", "mon")   # monday
PIPELINE_CRON_HOUR = int(os.getenv("PIPELINE_CRON_HOUR", "2"))       # 2 AM
PIPELINE_CRON_MINUTE = int(os.getenv("PIPELINE_CRON_MINUTE", "0"))

# Minimum new weekly records needed to trigger retraining
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))

# How many historical model versions to keep on disk
MODEL_RETENTION_COUNT = int(os.getenv("MODEL_RETENTION_COUNT", "5"))

# ---------------------------------------------------------------------------
# CSV / File ingestion
# ---------------------------------------------------------------------------
# Directory watched for new CSV/Excel uploads
INGEST_WATCH_DIR = os.getenv("INGEST_WATCH_DIR", str(DATA_DIR))

# The original dataset shipped with the project (used as seed data)
SEED_CSV = os.getenv("SEED_CSV", str(BASE_DIR / "uganda_drug_supply_synthetic.csv"))

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8090"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = str(LOG_DIR / "pipeline.log")
