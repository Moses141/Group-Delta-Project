"""
Model Registry
===============
Manages model lifecycle: saving, loading, versioning, and promotion.

Features
--------
- Save a trained model with a timestamped version tag.
- Register metadata (metrics, row count) in the ``model_versions`` DB table.
- Promote a version to "active" (the one served by the API/dashboard).
- Prune old versions beyond ``MODEL_RETENTION_COUNT``.
- Load the currently active model for inference.

Integration points
------------------
- ``pipeline/retrain.py`` calls ``save_model()`` after training.
- ``api/main.py`` calls ``load_active_model()`` for predictions.
- ``pipeline/scheduler.py`` calls ``prune_old_versions()`` on cleanup.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_DIR, MODEL_RETENTION_COUNT
from database.db_connection import get_session
from database.schema import ModelVersion

logger = logging.getLogger(__name__)


def _version_tag() -> str:
    """Generate a version string like ``20260217_020000``."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_model(
    model: Any,
    model_name: str,
    metrics: Dict[str, float],
    trained_on_rows: int,
    extra_artifacts: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Persist a trained model to disk and register it in the DB.

    Parameters
    ----------
    model : Any
        The fitted model object (sklearn, xgboost, keras, etc.).
    model_name : str
        Identifier like ``xgboost_demand`` or ``lstm_demand``.
    metrics : dict
        Evaluation metrics, e.g. ``{"MAE": 1.2, "RMSE": 2.3, "MAPE": 0.06}``.
    trained_on_rows : int
        Number of training rows (for audit).
    extra_artifacts : dict, optional
        Additional objects to save alongside the model (scalers, encoders).

    Returns
    -------
    str
        The version tag assigned to this model.
    """
    version = _version_tag()
    model_subdir = MODEL_DIR / model_name
    model_subdir.mkdir(parents=True, exist_ok=True)

    # Save model file
    model_filename = f"{model_name}_{version}.joblib"
    model_path = model_subdir / model_filename
    joblib.dump(model, model_path)
    logger.info("Model saved: %s", model_path)

    # Save extra artifacts (e.g. scalers) alongside
    if extra_artifacts:
        artifacts_path = model_subdir / f"{model_name}_{version}_artifacts.joblib"
        joblib.dump(extra_artifacts, artifacts_path)
        logger.info("Artifacts saved: %s", artifacts_path)

    # Register in DB
    with get_session() as session:
        record = ModelVersion(
            model_name=model_name,
            version=version,
            file_path=str(model_path),
            metrics_json=json.dumps(metrics),
            is_active=False,
            trained_on_rows=trained_on_rows,
        )
        session.add(record)

    logger.info("Model registered: %s v%s (rows=%d)", model_name, version, trained_on_rows)
    return version


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------
def promote_model(model_name: str, version: str) -> None:
    """
    Mark the given version as the active model.  Deactivates all other
    versions of the same model_name.
    """
    with get_session() as session:
        # Deactivate all
        session.query(ModelVersion).filter_by(model_name=model_name).update(
            {"is_active": False}
        )
        # Activate the target
        target = (
            session.query(ModelVersion)
            .filter_by(model_name=model_name, version=version)
            .first()
        )
        if target:
            target.is_active = True
            logger.info("Promoted %s v%s to active", model_name, version)
        else:
            logger.warning("Version %s not found for %s – nothing promoted", version, model_name)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_active_model(model_name: str) -> Tuple[Any, Optional[Dict]]:
    """
    Load the currently active model from disk.

    Returns
    -------
    (model, artifacts_or_None)
    """
    with get_session() as session:
        record = (
            session.query(ModelVersion)
            .filter_by(model_name=model_name, is_active=True)
            .first()
        )
        if not record:
            logger.warning("No active model found for %s", model_name)
            return None, None

        model_path = record.file_path
        version = record.version

    model = joblib.load(model_path)
    logger.info("Loaded active model: %s v%s", model_name, version)

    # Try loading artifacts
    artifacts_path = Path(model_path).parent / f"{model_name}_{version}_artifacts.joblib"
    artifacts = None
    if artifacts_path.exists():
        artifacts = joblib.load(artifacts_path)

    return model, artifacts


def get_active_version(model_name: str) -> Optional[str]:
    """Return the active version tag for a model, or None."""
    with get_session() as session:
        record = (
            session.query(ModelVersion)
            .filter_by(model_name=model_name, is_active=True)
            .first()
        )
        return record.version if record else None


def get_model_history(model_name: str) -> list:
    """Return a list of dicts for all versions of a model (newest first)."""
    with get_session() as session:
        records = (
            session.query(ModelVersion)
            .filter_by(model_name=model_name)
            .order_by(ModelVersion.created_at.desc())
            .all()
        )
        return [
            {
                "version": r.version,
                "is_active": r.is_active,
                "metrics": json.loads(r.metrics_json) if r.metrics_json else {},
                "trained_on_rows": r.trained_on_rows,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------
def prune_old_versions(model_name: str) -> int:
    """
    Delete model files and DB records beyond MODEL_RETENTION_COUNT.
    Keeps the N newest versions (and always the active one).
    Returns number of versions pruned.
    """
    with get_session() as session:
        all_versions = (
            session.query(ModelVersion)
            .filter_by(model_name=model_name)
            .order_by(ModelVersion.created_at.desc())
            .all()
        )

        if len(all_versions) <= MODEL_RETENTION_COUNT:
            return 0

        to_prune = all_versions[MODEL_RETENTION_COUNT:]
        pruned = 0
        for record in to_prune:
            if record.is_active:
                continue  # never delete the active model
            # Remove file from disk
            try:
                fpath = Path(record.file_path)
                if fpath.exists():
                    fpath.unlink()
                # Remove artifacts file if it exists
                artifacts_path = fpath.parent / f"{model_name}_{record.version}_artifacts.joblib"
                if artifacts_path.exists():
                    artifacts_path.unlink()
            except OSError as e:
                logger.warning("Could not delete file %s: %s", record.file_path, e)

            session.delete(record)
            pruned += 1

    logger.info("Pruned %d old versions of %s", pruned, model_name)
    return pruned
