"""
Pipeline Scheduler
===================
Orchestrates the full weekly data pipeline using APScheduler.

Pipeline stages (executed sequentially):
1. **Ingestion** – pull CSV / Excel / Google Sheets into raw_stock_data.
2. **Preprocessing** – clean and validate raw rows → cleaned_stock_data.
3. **Feature engineering** – aggregate to weekly + lag/rolling features.
4. **Retraining** – if enough new data, retrain models and refresh predictions.

Each stage is logged to the ``pipeline_runs`` audit table.

Integration points
------------------
- Started from ``api/main.py`` on app startup (background scheduler).
- Can also be started standalone: ``python -m pipeline.scheduler``
- Uses APScheduler CronTrigger configured via ``config.py``.
"""

import logging
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    PIPELINE_CRON_DAY_OF_WEEK,
    PIPELINE_CRON_HOUR,
    PIPELINE_CRON_MINUTE,
    LOG_FILE,
    LOG_LEVEL,
)

# Configure logging early
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

from database.db_connection import init_db, get_session
from database.schema import PipelineRun
from pipeline.ingestion import run_full_ingestion
from pipeline.preprocessing import run_preprocessing
from pipeline.feature_engineering import run_feature_engineering
from pipeline.retrain import run_retraining


def _log_stage(run_id: str, stage: str, status: str, records: int = 0, error: str = None):
    """
    Write or update an audit row in pipeline_runs.
    On 'started' → insert a new row.
    On 'success'/'failed'/'skipped' → update the existing row for this run_id+stage.
    """
    composite_id = f"{run_id}-{stage}"
    with get_session() as session:
        if status == "started":
            session.add(PipelineRun(
                run_id=composite_id,
                stage=stage,
                status=status,
                records_processed=0,
                started_at=datetime.utcnow(),
            ))
        else:
            existing = session.query(PipelineRun).filter_by(run_id=composite_id).first()
            if existing:
                existing.status = status
                existing.records_processed = records
                existing.error_message = error
                existing.finished_at = datetime.utcnow()
            else:
                # Fallback: insert if the "started" row was never created
                session.add(PipelineRun(
                    run_id=composite_id,
                    stage=stage,
                    status=status,
                    records_processed=records,
                    error_message=error,
                    started_at=datetime.utcnow(),
                    finished_at=datetime.utcnow(),
                ))


def run_full_pipeline(force_retrain: bool = False) -> dict:
    """
    Execute the complete pipeline end-to-end.

    Parameters
    ----------
    force_retrain : bool
        If True, retrain models regardless of new-data threshold.

    Returns
    -------
    dict summarising what happened at each stage.
    """
    run_id = str(uuid.uuid4())[:8]
    summary = {"run_id": run_id, "stages": {}}

    logger.info("========== Pipeline run %s started ==========", run_id)

    # Ensure tables exist
    init_db()

    # --- Stage 1: Ingestion ---
    _log_stage(run_id, "ingestion", "started")
    try:
        ingested = run_full_ingestion()
        _log_stage(run_id, "ingestion", "success", records=ingested)
        summary["stages"]["ingestion"] = {"status": "success", "rows": ingested}
        logger.info("Ingestion complete: %d rows", ingested)
    except Exception as e:
        _log_stage(run_id, "ingestion", "failed", error=str(e))
        summary["stages"]["ingestion"] = {"status": "failed", "error": str(e)}
        logger.error("Ingestion failed: %s", e)

    # --- Stage 2: Preprocessing ---
    _log_stage(run_id, "preprocessing", "started")
    try:
        cleaned = run_preprocessing()
        _log_stage(run_id, "preprocessing", "success", records=cleaned)
        summary["stages"]["preprocessing"] = {"status": "success", "rows": cleaned}
        logger.info("Preprocessing complete: %d rows", cleaned)
    except Exception as e:
        _log_stage(run_id, "preprocessing", "failed", error=str(e))
        summary["stages"]["preprocessing"] = {"status": "failed", "error": str(e)}
        logger.error("Preprocessing failed: %s", e)

    # --- Stage 3: Feature engineering ---
    _log_stage(run_id, "feature_eng", "started")
    try:
        weekly = run_feature_engineering()
        _log_stage(run_id, "feature_eng", "success", records=weekly)
        summary["stages"]["feature_engineering"] = {"status": "success", "rows": weekly}
        logger.info("Feature engineering complete: %d rows", weekly)
    except Exception as e:
        _log_stage(run_id, "feature_eng", "failed", error=str(e))
        summary["stages"]["feature_engineering"] = {"status": "failed", "error": str(e)}
        logger.error("Feature engineering failed: %s", e)

    # --- Stage 4: Retraining ---
    _log_stage(run_id, "retrain", "started")
    try:
        retrain_result = run_retraining(force=force_retrain)
        status = "success" if retrain_result.get("retrained") else "skipped"
        _log_stage(run_id, "retrain", status,
                   records=retrain_result.get("predictions_count", 0))
        summary["stages"]["retraining"] = {"status": status, **retrain_result}
        logger.info("Retraining complete: %s", retrain_result)
    except Exception as e:
        _log_stage(run_id, "retrain", "failed", error=str(e))
        summary["stages"]["retraining"] = {"status": "failed", "error": str(e)}
        logger.error("Retraining failed: %s", e)

    logger.info("========== Pipeline run %s finished ==========", run_id)
    return summary


# ---------------------------------------------------------------------------
# APScheduler setup
# ---------------------------------------------------------------------------
_scheduler = None


def start_scheduler():
    """
    Start the APScheduler background scheduler with a weekly cron trigger.
    Safe to call multiple times – only starts once.
    """
    global _scheduler
    if _scheduler is not None:
        logger.info("Scheduler already running.")
        return

    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        run_full_pipeline,
        trigger=CronTrigger(
            day_of_week=PIPELINE_CRON_DAY_OF_WEEK,
            hour=PIPELINE_CRON_HOUR,
            minute=PIPELINE_CRON_MINUTE,
        ),
        id="weekly_pipeline",
        name="Weekly data pipeline",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "Scheduler started – pipeline runs every %s at %02d:%02d",
        PIPELINE_CRON_DAY_OF_WEEK, PIPELINE_CRON_HOUR, PIPELINE_CRON_MINUTE,
    )


def stop_scheduler():
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped.")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedSupply AI Pipeline")
    parser.add_argument("--run-now", action="store_true", help="Run the full pipeline immediately")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--daemon", action="store_true", help="Start scheduler daemon (blocks)")
    args = parser.parse_args()

    if args.run_now:
        result = run_full_pipeline(force_retrain=args.force_retrain)
        print("Pipeline result:", result)
    elif args.daemon:
        start_scheduler()
        print("Scheduler running. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            stop_scheduler()
    else:
        parser.print_help()
