"""
API Routes
===========
All REST endpoints for MedSupply AI, grouped by concern.

Route groups
------------
1. ``/api/health``        – liveness / readiness check
2. ``/api/predictions``   – latest stock predictions (dashboard data source)
3. ``/api/pipeline/*``    – trigger ingestion, preprocessing, retrain
4. ``/api/models/*``      – model registry queries
5. ``/api/upload``        – ad-hoc file upload (CSV / Excel)

Integration points
------------------
- Included in ``api/main.py`` via ``app.include_router(router)``.
- Predictions endpoint is consumed by the modified Streamlit dashboard.
"""

import logging
import os
import shutil
import sys
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATA_DIR
from database.db_connection import get_session, engine
from database.schema import PredictionCache, PipelineRun, WeeklyAggregatedData
from models.model_registry import get_active_version, get_model_history, load_active_model
from pipeline.ingestion import ingest_csv, ingest_excel
from pipeline.scheduler import run_full_pipeline

import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["MedSupply AI"])


# ============================= Health ======================================

@router.get("/health")
def health_check():
    """Liveness check – returns 200 if the API is running."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ============================= Predictions =================================

@router.get("/predictions")
def get_predictions(
    drug_id: Optional[str] = Query(None, description="Filter by drug_id"),
    region: Optional[str] = Query(None, description="Filter by distribution_region"),
    limit: int = Query(100, ge=1, le=5000),
):
    """
    Return the latest predictions from the cache.
    The Streamlit dashboard calls this endpoint to populate its tables/charts.
    """
    with get_session() as session:
        q = session.query(PredictionCache)
        if drug_id:
            q = q.filter(PredictionCache.drug_id == drug_id)
        if region:
            q = q.filter(PredictionCache.distribution_region == region)
        rows = q.order_by(PredictionCache.predicted_at.desc()).limit(limit).all()

        return [
            {
                "drug_id": r.drug_id,
                "drug_name": r.drug_name,
                "distribution_region": r.distribution_region,
                "facility_type": r.facility_type,
                "predicted_demand": r.predicted_demand,
                "predicted_stockout_probability": r.predicted_stockout_probability,
                "recommended_order_qty": r.recommended_order_qty,
                "stockout_risk_level": r.stockout_risk_level,
                "current_stock": r.current_stock,
                "model_version": r.model_version,
                "predicted_at": r.predicted_at.isoformat() if r.predicted_at else None,
            }
            for r in rows
        ]


@router.get("/predictions/summary")
def predictions_summary():
    """
    Aggregate summary for the dashboard header cards:
    total drugs tracked, high-risk count, avg predicted demand, etc.
    """
    with get_session() as session:
        total = session.query(PredictionCache).count()
        high_risk = session.query(PredictionCache).filter(
            PredictionCache.stockout_risk_level == "High"
        ).count()
        medium_risk = session.query(PredictionCache).filter(
            PredictionCache.stockout_risk_level == "Medium"
        ).count()

    return {
        "total_predictions": total,
        "high_risk_count": high_risk,
        "medium_risk_count": medium_risk,
        "low_risk_count": total - high_risk - medium_risk,
    }


# ============================= Pipeline ====================================

@router.post("/pipeline/run")
def trigger_pipeline(force_retrain: bool = Query(False)):
    """
    Trigger the full pipeline (ingestion → preprocessing → features → retrain).
    This is the same flow that runs weekly on the scheduler.
    """
    try:
        result = run_full_pipeline(force_retrain=force_retrain)
        return {"status": "completed", "result": result}
    except Exception as e:
        logger.error("Pipeline run failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/ingest")
def trigger_ingestion():
    """Trigger only the ingestion stage."""
    from pipeline.ingestion import run_full_ingestion
    try:
        rows = run_full_ingestion()
        return {"status": "success", "rows_ingested": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/retrain")
def trigger_retrain():
    """Force model retraining regardless of threshold."""
    from pipeline.retrain import run_retraining
    try:
        result = run_retraining(force=True)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/history")
def pipeline_history(limit: int = Query(20, ge=1, le=200)):
    """Return recent pipeline run audit entries."""
    with get_session() as session:
        runs = (
            session.query(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "run_id": r.run_id,
                "stage": r.stage,
                "status": r.status,
                "records_processed": r.records_processed,
                "error_message": r.error_message,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            }
            for r in runs
        ]


# ============================= File Upload =================================

@router.post("/upload")
async def upload_data_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file for ingestion.
    The file is saved to the data/ directory and ingested immediately.
    """
    allowed_ext = {".csv", ".xlsx", ".xls"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_ext}",
        )

    # Save uploaded file to data/
    dest = DATA_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest
    try:
        if ext == ".csv":
            rows = ingest_csv(str(dest))
        else:
            rows = ingest_excel(str(dest))
        return {"status": "success", "file": file.filename, "rows_ingested": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================= Models ======================================

@router.get("/models/active")
def active_models():
    """List the currently active model version for each model name."""
    model_names = ["random_forest_demand", "xgboost_demand"]
    result = {}
    for name in model_names:
        version = get_active_version(name)
        result[name] = version or "not trained"
    return result


@router.get("/models/{model_name}/history")
def model_history(model_name: str):
    """Return the version history for a specific model."""
    history = get_model_history(model_name)
    if not history:
        raise HTTPException(status_code=404, detail=f"No versions found for {model_name}")
    return history


# ============================= Data Stats ==================================

@router.get("/data/stats")
def data_stats():
    """Quick overview of data volumes in the system."""
    from database.schema import RawStockData, CleanedStockData
    with get_session() as session:
        raw_count = session.query(RawStockData).count()
        clean_count = session.query(CleanedStockData).count()
        weekly_count = session.query(WeeklyAggregatedData).count()
        pred_count = session.query(PredictionCache).count()
    return {
        "raw_stock_rows": raw_count,
        "cleaned_rows": clean_count,
        "weekly_aggregated_rows": weekly_count,
        "cached_predictions": pred_count,
    }
