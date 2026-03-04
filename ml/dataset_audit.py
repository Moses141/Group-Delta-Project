"""Dataset audit utilities for schema, missingness, and leakage diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.feature_spec import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_COLUMNS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TIME_COLUMN,
)


def audit_dataset(path: str | Path) -> dict:
    df = pd.read_csv(path)
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    potential_leakage = [c for c in FORBIDDEN_COLUMNS if c in df.columns]
    unavailable_at_prediction_time = [
        c
        for c in [
            "forecast_error_percent",
            "financial_loss_due_to_expiry_usd",
            "predicted_stockout_probability",
            "expiry_risk_category",
            "stockout_occurred",
            "expiry_rate_percent",
        ]
        if c in df.columns
    ]

    feature_derived_from_target = [
        c for c in ["inventory_turnover", "composite_risk_score", "service_level_estimate"] if c in df.columns
    ]

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missingness_pct": missing_pct.to_dict(),
        "target_columns": [c for c in [TARGET_COLUMN, "stockout_occurred", "expiry_rate_percent"] if c in df.columns],
        "time_column": TIME_COLUMN if TIME_COLUMN in df.columns else None,
        "id_columns": [c for c in ID_COLUMNS if c in df.columns],
        "categorical_columns": [c for c in CATEGORICAL_FEATURES if c in df.columns],
        "numerical_columns": [c for c in NUMERIC_FEATURES if c in df.columns],
        "potential_leakage_columns": potential_leakage,
        "feature_derived_from_target": feature_derived_from_target,
        "not_available_at_prediction_time": unavailable_at_prediction_time,
    }
