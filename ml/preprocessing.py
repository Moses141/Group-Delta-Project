"""Shared preprocessing utilities with chronological splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.feature_spec import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TIME_COLUMN,
)


@dataclass
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame


def time_series_split(df: pd.DataFrame, time_col: str = TIME_COLUMN, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col)
    split_idx = max(1, int(len(out) * (1 - test_size)))
    train = out.iloc[:split_idx].copy()
    test = out.iloc[split_idx:].copy()
    return train, test


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if c in df.columns]
    return [c for c in candidates if c not in FORBIDDEN_COLUMNS and c != TARGET_COLUMN]


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric = [c for c in feature_columns if c in NUMERIC_FEATURES]
    categorical = [c for c in feature_columns if c in CATEGORICAL_FEATURES]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
    )
