"""Model retraining service using a leakage-safe unified ML pipeline artifact."""

from __future__ import annotations

import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, RETRAIN_THRESHOLD
from database.db_connection import get_session, engine
from database.schema import ModelVersion, PredictionCache, WeeklyAggregatedData
from ml.feature_spec import TARGET_COLUMN
from ml.train import save_pipeline_artifact, train_demand_pipeline
from models.model_registry import prune_old_versions, promote_model, save_model

logger = logging.getLogger(__name__)


def _count_new_weekly_rows() -> int:
    with get_session() as session:
        latest_model = (
            session.query(ModelVersion)
            .filter(ModelVersion.model_name == "demand_pipeline")
            .order_by(ModelVersion.created_at.desc())
            .first()
        )
        if latest_model and latest_model.created_at:
            return (
                session.query(WeeklyAggregatedData)
                .filter(WeeklyAggregatedData.created_at > latest_model.created_at)
                .count()
            )
        return session.query(WeeklyAggregatedData).count()


def should_retrain() -> bool:
    count = _count_new_weekly_rows()
    logger.info("New weekly rows since last demand_pipeline train: %d", count)
    return count >= RETRAIN_THRESHOLD


def _load_training_data() -> pd.DataFrame:
    query = """
    SELECT
        drug_id,
        drug_name,
        distribution_region,
        NULL as facility_type,
        date(printf('%04d-01-01', year), '+' || ((iso_week - 1) * 7) || ' days') AS stock_received_date,
        total_stock_received AS initial_stock_units,
        avg_reorder_level AS reorder_level,
        avg_monthly_demand AS average_monthly_demand,
        NULL AS delivery_frequency_days,
        avg_lead_time_days AS lead_time_days,
        avg_supplier_reliability AS supplier_reliability_score,
        NULL AS region_disease_outbreaks,
        NULL AS season,
        NULL AS transport_accessibility_score,
        NULL AS power_stability_index,
        NULL AS staff_availability_index,
        NULL AS data_record_quality,
        NULL AS storage_temperature,
        NULL AS storage_humidity,
        NULL AS FEFO_policy_implemented,
        NULL AS warehouse_capacity_utilization,
        NULL AS storage_condition_rating,
        NULL AS delivery_delay_days,
        year,
        iso_week
    FROM weekly_aggregated_data
    ORDER BY year, iso_week
    """
    return pd.read_sql(query, con=engine)


def _refresh_predictions_cache(pipeline_obj, feature_columns, model_version: str) -> int:
    df = _load_training_data()
    if df.empty:
        return 0

    idx = df.groupby(["drug_id", "distribution_region"])["stock_received_date"].idxmax()
    latest = df.loc[idx].copy()
    X = latest[feature_columns].copy()
    predictions = pipeline_obj.predict(X)

    latest["predicted_demand"] = predictions
    lead_time = latest["lead_time_days"].fillna(14).astype(float)
    stock = latest["initial_stock_units"].fillna(0).astype(float)
    latest["recommended_order_qty"] = np.maximum(0, latest["predicted_demand"] * (lead_time / 30.0) + 0.2 * latest["predicted_demand"] - stock)
    latest["predicted_stockout_probability"] = np.where(
        latest["predicted_demand"] > stock,
        np.minimum(1.0, (latest["predicted_demand"] - stock) / (latest["predicted_demand"] + 1e-6)),
        0.05,
    )
    latest["stockout_risk_level"] = pd.cut(
        latest["predicted_stockout_probability"],
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"],
    )

    with get_session() as session:
        session.query(PredictionCache).delete()
        for _, row in latest.iterrows():
            session.add(PredictionCache(
                drug_id=row.get("drug_id"),
                drug_name=row.get("drug_name"),
                distribution_region=row.get("distribution_region"),
                facility_type=row.get("facility_type") or "Unknown",
                predicted_demand=float(row.get("predicted_demand", 0)),
                predicted_stockout_probability=float(row.get("predicted_stockout_probability", 0)),
                recommended_order_qty=float(row.get("recommended_order_qty", 0)),
                stockout_risk_level=str(row.get("stockout_risk_level", "Medium")),
                current_stock=float(row.get("initial_stock_units", 0)),
                model_version=model_version,
                predicted_at=datetime.utcnow(),
            ))
        return len(latest)


def run_retraining(force: bool = False) -> dict:
    result = {"retrained": False, "models": [], "predictions_count": 0}
    if not force and not should_retrain():
        return result

    df = _load_training_data()
    if df.empty or len(df) < 20:
        logger.warning("Not enough rows for retraining: %d", len(df))
        return result

    trained = train_demand_pipeline(df=df, test_size=0.2)
    metrics = trained["metrics"]

    artifact_path = save_pipeline_artifact(trained, ARTIFACTS_DIR / "pipeline.joblib")
    model_version = save_model(
        model=trained["pipeline"],
        model_name="demand_pipeline",
        metrics=metrics,
        trained_on_rows=trained["train_rows"],
        extra_artifacts={"feature_columns": trained["feature_columns"], "artifact_path": str(artifact_path)},
    )
    promote_model("demand_pipeline", model_version)
    prune_old_versions("demand_pipeline")

    result["predictions_count"] = _refresh_predictions_cache(
        pipeline_obj=trained["pipeline"],
        feature_columns=trained["feature_columns"],
        model_version=model_version,
    )
    result["models"].append({"name": "demand_pipeline", "version": model_version, **metrics})
    result["retrained"] = True
    return result
