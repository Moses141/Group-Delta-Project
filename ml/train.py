"""Production training pipeline with leakage-safe chronological split."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from config import ARTIFACTS_DIR
from ml.evaluate import regression_metrics
from ml.feature_engineering import add_lag_features, add_temporal_columns
from ml.feature_spec import TARGET_COLUMN, TIME_COLUMN
from ml.preprocessing import build_preprocessor, select_feature_columns, time_series_split


def train_demand_pipeline(df: pd.DataFrame, model=None, test_size: float = 0.2) -> dict:
    data = add_temporal_columns(df)
    data = add_lag_features(data)
    data = data.dropna(subset=[TARGET_COLUMN, TIME_COLUMN])

    train_df, test_df = time_series_split(data, time_col=TIME_COLUMN, test_size=test_size)

    feature_columns = select_feature_columns(train_df)
    if TARGET_COLUMN in feature_columns:
        raise ValueError("Target leakage detected: target present in feature columns")

    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    estimator = model or RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    preprocessor = build_preprocessor(feature_columns)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, preds)

    return {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": metrics,
        "train_period": (str(train_df[TIME_COLUMN].min()), str(train_df[TIME_COLUMN].max())),
        "test_period": (str(test_df[TIME_COLUMN].min()), str(test_df[TIME_COLUMN].max())),
    }


def save_pipeline_artifact(trained: dict, artifact_path: Path | None = None) -> Path:
    path = artifact_path or (ARTIFACTS_DIR / "pipeline.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": trained["pipeline"],
        "feature_columns": trained["feature_columns"],
        "metrics": trained["metrics"],
        "train_period": trained["train_period"],
        "test_period": trained["test_period"],
    }, path)
    return path
