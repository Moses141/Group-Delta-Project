"""
Data Preprocessing Service
============================
Reads raw rows from ``raw_stock_data``, applies validation and cleaning,
then writes results to ``cleaned_stock_data``.

Cleaning steps
--------------
1. Remove rows with null drug_id or stock_received_date.
2. Clamp negative stock / demand values to 0 (no negative inventory).
3. Impute missing numeric columns with median.
4. Standardise date formatting and extract year / month / ISO-week.
5. Validate score columns are in [0, 1] range.
6. Deduplicate – only process raw rows not yet in cleaned table.

Integration points
------------------
- Called by ``pipeline/scheduler.py`` right after ingestion.
- Reads ``database.schema.RawStockData``, writes ``database.schema.CleanedStockData``.
"""

import logging
import os, sys
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import func, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.db_connection import get_session, engine
from database.schema import RawStockData, CleanedStockData

logger = logging.getLogger(__name__)

# Columns that must never be negative (clamped to 0)
NON_NEGATIVE_COLS = [
    "initial_stock_units",
    "reorder_level",
    "average_monthly_demand",
    "delivery_frequency_days",
    "lead_time_days",
    "expiry_rate_percent",
    "warehouse_capacity_utilization",
]

# Columns that should be in [0, 1]
SCORE_COLS = [
    "supplier_reliability_score",
    "transport_accessibility_score",
    "power_stability_index",
    "staff_availability_index",
]


def _get_max_cleaned_raw_id() -> int:
    """Return the highest raw_id already present in cleaned_stock_data."""
    with get_session() as session:
        result = session.query(func.max(CleanedStockData.raw_id)).scalar()
        return result or 0


def _load_unprocessed_raw(batch_size: int = 50_000) -> pd.DataFrame:
    """
    Load raw rows that haven't been cleaned yet (raw.id > max cleaned raw_id).
    Returns a pandas DataFrame.
    """
    max_id = _get_max_cleaned_raw_id()
    query = f"SELECT * FROM raw_stock_data WHERE id > {max_id} ORDER BY id LIMIT {batch_size}"
    df = pd.read_sql(query, con=engine)
    logger.info("Loaded %d unprocessed raw rows (id > %d)", len(df), max_id)
    return df


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning / validation rules and return a cleaned DataFrame."""

    initial_count = len(df)

    # ------------------------------------------------------------------
    # 1. Drop rows missing critical identifiers
    # ------------------------------------------------------------------
    df = df.dropna(subset=["drug_id", "stock_received_date"])
    logger.info("After dropping nulls in drug_id/date: %d → %d rows", initial_count, len(df))

    # ------------------------------------------------------------------
    # 2. Parse and normalise dates
    # ------------------------------------------------------------------
    df["stock_received_date"] = pd.to_datetime(df["stock_received_date"], errors="coerce")
    df = df.dropna(subset=["stock_received_date"])
    df["year"] = df["stock_received_date"].dt.year
    df["month"] = df["stock_received_date"].dt.month
    df["iso_week"] = df["stock_received_date"].dt.isocalendar().week.astype(int)

    # ------------------------------------------------------------------
    # 3. Clamp negatives to zero
    # ------------------------------------------------------------------
    for col in NON_NEGATIVE_COLS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # ------------------------------------------------------------------
    # 4. Impute missing numerics with median
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Imputed %s with median %.2f", col, median_val)

    # ------------------------------------------------------------------
    # 5. Validate score columns [0, 1]
    # ------------------------------------------------------------------
    for col in SCORE_COLS:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)

    # ------------------------------------------------------------------
    # 6. Standardise categorical columns
    # ------------------------------------------------------------------
    if "data_record_quality" in df.columns:
        valid_quality = {"High", "Moderate", "Low"}
        df["data_record_quality"] = df["data_record_quality"].apply(
            lambda x: x if x in valid_quality else "Moderate"
        )

    if "storage_condition_rating" in df.columns:
        valid_storage = {"Good", "Fair", "Poor"}
        df["storage_condition_rating"] = df["storage_condition_rating"].apply(
            lambda x: x if x in valid_storage else "Fair"
        )

    if "expiry_risk_category" in df.columns:
        valid_risk = {"Low", "Medium", "High"}
        df["expiry_risk_category"] = df["expiry_risk_category"].apply(
            lambda x: x if x in valid_risk else "Medium"
        )

    logger.info("Cleaning complete: %d rows retained", len(df))
    return df


def _write_cleaned_rows(df: pd.DataFrame) -> int:
    """Bulk-insert cleaned rows into cleaned_stock_data. Returns count inserted."""
    if df.empty:
        return 0

    # Map columns to the CleanedStockData model
    cols_to_keep = [
        "drug_id", "drug_name", "distribution_region", "facility_type",
        "initial_stock_units", "stock_received_date", "reorder_level",
        "average_monthly_demand", "delivery_frequency_days", "lead_time_days",
        "supplier_reliability_score", "region_disease_outbreaks", "season",
        "transport_accessibility_score", "power_stability_index",
        "staff_availability_index", "data_record_quality",
        "FEFO_policy_implemented", "warehouse_capacity_utilization",
        "storage_condition_rating", "stockout_occurred", "expiry_rate_percent",
        "predicted_stockout_probability", "expiry_risk_category",
        "year", "month", "iso_week",
    ]

    # Ensure stock_received_date is date (not datetime)
    df["stock_received_date"] = pd.to_datetime(df["stock_received_date"]).dt.date

    available = [c for c in cols_to_keep if c in df.columns]
    out = df[available].copy()

    # Add raw_id reference if available
    if "id" in df.columns:
        out["raw_id"] = df["id"]

    out["cleaned_at"] = datetime.utcnow()

    inserted = 0
    with get_session() as session:
        records = out.to_dict(orient="records")
        for rec in records:
            sp = session.begin_nested()
            try:
                session.add(CleanedStockData(**rec))
                session.flush()
                sp.commit()
                inserted += 1
            except Exception:
                sp.rollback()
                continue

    logger.info("Wrote %d cleaned rows to DB", inserted)
    return inserted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_preprocessing() -> int:
    """
    Main entry point.  Loads unprocessed raw data, cleans it, and writes to
    the cleaned table.  Returns number of cleaned rows written.
    """
    df = _load_unprocessed_raw()
    if df.empty:
        logger.info("No new raw data to preprocess.")
        return 0

    df_clean = _clean_dataframe(df)
    return _write_cleaned_rows(df_clean)
