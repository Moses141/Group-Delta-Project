"""
Model Retraining Service
=========================
Checks whether enough new weekly data has accumulated, and if so, retrains
the XGBoost and Random Forest demand-forecasting models.

Retraining logic
----------------
1. Count new weekly rows since the last retrain.
2. If count >= ``RETRAIN_THRESHOLD`` (from config), proceed.
3. Load all ``weekly_aggregated_data`` from the DB.
4. Train XGBoost and Random Forest regressors for demand prediction.
5. Evaluate on a hold-out split.
6. Save models via the model registry (versioned, on disk + DB).
7. Promote the new versions to "active".
8. Refresh the ``predictions_cache`` table for the dashboard.

Integration points
------------------
- Called by ``pipeline/scheduler.py`` after feature engineering.
- Also callable from the FastAPI ``/api/pipeline/retrain`` endpoint.
- Uses ``models/model_registry.py`` for save / promote / prune.
- Writes ``database.schema.PredictionCache`` so the dashboard auto-updates.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RETRAIN_THRESHOLD
from database.db_connection import get_session, engine
from database.schema import WeeklyAggregatedData, PredictionCache, ModelVersion
from models.model_registry import save_model, promote_model, prune_old_versions

logger = logging.getLogger(__name__)

# Try importing XGBoost (core dependency)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed – XGBoost retraining disabled.")

# Feature columns used for demand prediction (must match feature_engineering output)
FEATURE_COLS = [
    "total_stock_received",
    "avg_reorder_level",
    "avg_lead_time_days",
    "avg_supplier_reliability",
    "stockout_count",
    "avg_expiry_rate",
    "avg_stockout_probability",
    "demand_lag_1w",
    "demand_lag_2w",
    "demand_lag_4w",
    "demand_rolling_4w_mean",
    "demand_rolling_4w_std",
    "stock_demand_ratio",
    "composite_risk_score",
]

TARGET_COL = "avg_monthly_demand"


# ---------------------------------------------------------------------------
# Check whether retraining is needed
# ---------------------------------------------------------------------------
def _count_new_weekly_rows() -> int:
    """
    Count weekly_aggregated_data rows created after the most recent model
    training run.  If no model exists yet, return total rows.
    """
    with get_session() as session:
        latest_model = (
            session.query(ModelVersion)
            .order_by(ModelVersion.created_at.desc())
            .first()
        )
        if latest_model and latest_model.created_at:
            cutoff = latest_model.created_at
            count = (
                session.query(WeeklyAggregatedData)
                .filter(WeeklyAggregatedData.created_at > cutoff)
                .count()
            )
        else:
            count = session.query(WeeklyAggregatedData).count()

    logger.info("New weekly rows since last training: %d (threshold: %d)", count, RETRAIN_THRESHOLD)
    return count


def should_retrain() -> bool:
    """Return True if enough new data warrants retraining."""
    return _count_new_weekly_rows() >= RETRAIN_THRESHOLD


# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------
def _load_training_data() -> pd.DataFrame:
    """Load all weekly aggregated data from the DB."""
    query = "SELECT * FROM weekly_aggregated_data ORDER BY year, iso_week"
    df = pd.read_sql(query, con=engine)
    logger.info("Training data loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Train models
# ---------------------------------------------------------------------------
def _train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Tuple[object, Dict[str, float]]:
    """Train a Random Forest regressor and return (model, metrics)."""
    logger.info("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAPE": float(mean_absolute_percentage_error(y_test, preds)),
    }
    logger.info("Random Forest metrics: %s", metrics)
    return model, metrics


def _train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Tuple[object, Dict[str, float]]:
    """Train an XGBoost regressor and return (model, metrics)."""
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost is not installed")

    logger.info("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, subsample=0.8, colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAPE": float(mean_absolute_percentage_error(y_test, preds)),
    }
    logger.info("XGBoost metrics: %s", metrics)
    return model, metrics


# ---------------------------------------------------------------------------
# Refresh predictions cache
# ---------------------------------------------------------------------------
def _refresh_predictions(model, scaler, feature_cols, model_version: str) -> int:
    """
    Generate predictions for the latest week of each drug × region and write
    them to predictions_cache so the dashboard auto-updates.
    """
    df = _load_training_data()
    if df.empty:
        return 0

    # Get latest week per drug × region
    idx = df.groupby(["drug_id", "distribution_region"])["created_at"].idxmax()
    latest = df.loc[idx].copy()

    # Prepare features
    available_features = [c for c in feature_cols if c in latest.columns]
    X = latest[available_features].fillna(0).values
    if scaler:
        X = scaler.transform(X)

    predictions = model.predict(X)
    latest["predicted_demand"] = predictions

    # Simple risk classification
    latest["stockout_risk_level"] = pd.cut(
        latest["avg_stockout_probability"].fillna(0),
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"],
    )

    # OOQ approximation: predicted_demand * lead_time_factor + safety_stock - current_stock
    latest["recommended_order_qty"] = np.maximum(
        0,
        latest["predicted_demand"] * (latest["avg_lead_time_days"].fillna(14) / 30)
        + latest["predicted_demand"] * 0.2  # safety stock ~20%
        - latest["total_stock_received"].fillna(0),
    )

    # Write to predictions_cache (clear old, insert new)
    written = 0
    with get_session() as session:
        session.query(PredictionCache).delete()
        for _, row in latest.iterrows():
            session.add(PredictionCache(
                drug_id=row.get("drug_id"),
                drug_name=row.get("drug_name"),
                distribution_region=row.get("distribution_region"),
                predicted_demand=float(row.get("predicted_demand", 0)),
                predicted_stockout_probability=float(row.get("avg_stockout_probability", 0)),
                recommended_order_qty=float(row.get("recommended_order_qty", 0)),
                stockout_risk_level=str(row.get("stockout_risk_level", "Medium")),
                current_stock=float(row.get("total_stock_received", 0)),
                model_version=model_version,
                predicted_at=datetime.utcnow(),
            ))
            written += 1

    logger.info("Refreshed predictions_cache: %d entries", written)
    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_retraining(force: bool = False) -> Dict:
    """
    Main entry point.

    Parameters
    ----------
    force : bool
        If True, retrain regardless of threshold.

    Returns
    -------
    dict with keys: retrained (bool), models (list of dicts), predictions_count (int)
    """
    result = {"retrained": False, "models": [], "predictions_count": 0}

    if not force and not should_retrain():
        logger.info("Retraining threshold not met. Skipping.")
        return result

    df = _load_training_data()
    if df.empty or len(df) < 20:
        logger.warning("Not enough training data (%d rows). Skipping.", len(df))
        return result

    # Prepare features
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    if not available_features:
        logger.error("No feature columns found in weekly data. Aborting retrain.")
        return result

    X = df[available_features].fillna(0)
    y = df[TARGET_COL].fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    n_train_rows = len(X_train)
    best_model = None
    best_version = None
    best_mape = float("inf")

    # --- Random Forest ---
    try:
        rf_model, rf_metrics = _train_random_forest(X_train, y_train, X_test, y_test)
        rf_version = save_model(
            model=rf_model,
            model_name="random_forest_demand",
            metrics=rf_metrics,
            trained_on_rows=n_train_rows,
            extra_artifacts={"scaler": scaler, "feature_cols": available_features},
        )
        promote_model("random_forest_demand", rf_version)
        prune_old_versions("random_forest_demand")
        result["models"].append({"name": "random_forest_demand", "version": rf_version, **rf_metrics})
        if rf_metrics["MAPE"] < best_mape:
            best_mape = rf_metrics["MAPE"]
            best_model = rf_model
            best_version = rf_version
    except Exception as e:
        logger.error("Random Forest training failed: %s", e)

    # --- XGBoost ---
    if XGB_AVAILABLE:
        try:
            xgb_model, xgb_metrics = _train_xgboost(X_train, y_train, X_test, y_test)
            xgb_version = save_model(
                model=xgb_model,
                model_name="xgboost_demand",
                metrics=xgb_metrics,
                trained_on_rows=n_train_rows,
                extra_artifacts={"scaler": scaler, "feature_cols": available_features},
            )
            promote_model("xgboost_demand", xgb_version)
            prune_old_versions("xgboost_demand")
            result["models"].append({"name": "xgboost_demand", "version": xgb_version, **xgb_metrics})
            if xgb_metrics["MAPE"] < best_mape:
                best_mape = xgb_metrics["MAPE"]
                best_model = xgb_model
                best_version = xgb_version
        except Exception as e:
            logger.error("XGBoost training failed: %s", e)

    # --- Refresh predictions cache using the best model ---
    if best_model is not None:
        result["predictions_count"] = _refresh_predictions(
            best_model, scaler, available_features, best_version
        )
        result["retrained"] = True

    logger.info("Retraining complete: %s", result)
    return result
