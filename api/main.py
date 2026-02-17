"""
FastAPI Application – MedSupply AI Backend
============================================
Provides REST endpoints for:
  - Stock predictions (consumed by the Streamlit dashboard)
  - Pipeline control (trigger ingestion / retrain manually)
  - Model registry inspection
  - Health check

Startup hooks
-------------
- Initialise database tables.
- Seed original CSV data on first run.
- Start the APScheduler weekly pipeline.

Integration points
------------------
- The Streamlit dashboard calls ``/api/predictions`` to get live data.
- Pipeline admin endpoints let operators trigger runs outside the cron.
- This file does NOT replace the dashboard.  Both run concurrently:
    * FastAPI on port 8000  (data API)
    * Streamlit on port 8501 (UI)

Run
---
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import API_HOST, API_PORT, LOG_FILE, LOG_LEVEL

# Logging setup (before importing anything that logs)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.db_connection import init_db
from pipeline.ingestion import seed_from_original_csv
from pipeline.scheduler import start_scheduler, stop_scheduler

from api.routes import router as api_router


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run on startup, yield for the app lifetime, then run on shutdown."""
    logger.info("Starting MedSupply AI API...")

    # 1. Create / verify DB tables
    init_db()

    # 2. Seed original CSV if DB is empty (idempotent)
    seed_from_original_csv()

    # 3. Start background scheduler
    start_scheduler()

    logger.info("API startup complete.")
    yield

    # Shutdown
    stop_scheduler()
    logger.info("API shutdown complete.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MedSupply AI – Uganda",
    description=(
        "Backend API for the MedSupply AI pharmaceutical stock prediction system. "
        "Serves predictions, manages the data pipeline, and exposes model metadata."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow the Streamlit dashboard and any local dev tooling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes
app.include_router(api_router)


# ---------------------------------------------------------------------------
# Run with uvicorn when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
