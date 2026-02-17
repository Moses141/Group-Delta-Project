"""
Feature Engineering Service
============================
Reads cleaned data, aggregates to ISO-week level per drug × region, and
computes ML-ready features:

- Lag features: demand_lag_1w, demand_lag_2w, demand_lag_4w
- Rolling statistics: demand_rolling_4w_mean, demand_rolling_4w_std
- Derived ratios: stock_demand_ratio, composite_risk_score

Writes results to ``weekly_aggregated_data`` table.

Integration points
------------------
- Called by ``pipeline/scheduler.py`` after preprocessing.
- Reads ``cleaned_stock_data``, writes ``weekly_aggregated_data``.
- The retrain module later reads ``weekly_aggregated_data`` for model training.
"""

import logging
import os, sys
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import func, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.db_connection import get_session, engine
from database.schema import CleanedStockData, WeeklyAggregatedData

logger = logging.getLogger(__name__)


def _load_cleaned_data() -> pd.DataFrame:
    """Load all cleaned rows (we re-aggregate everything for consistency)."""
    query = "SELECT * FROM cleaned_stock_data ORDER BY year, iso_week"
    df = pd.read_sql(query, con=engine)
    logger.info("Loaded %d cleaned rows for feature engineering", len(df))
    return df


def _aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (drug_id, distribution_region, year, iso_week) and compute
    aggregated metrics for each week.
    """
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["drug_id", "drug_name", "distribution_region", "year", "iso_week"]).agg(
        total_stock_received=("initial_stock_units", "sum"),
        avg_monthly_demand=("average_monthly_demand", "mean"),
        avg_reorder_level=("reorder_level", "mean"),
        avg_lead_time_days=("lead_time_days", "mean"),
        avg_supplier_reliability=("supplier_reliability_score", "mean"),
        stockout_count=("stockout_occurred", "sum"),
        avg_expiry_rate=("expiry_rate_percent", "mean"),
        avg_stockout_probability=("predicted_stockout_probability", "mean"),
        record_count=("drug_id", "count"),
    ).reset_index()

    logger.info("Aggregated to %d weekly records", len(agg))
    return agg


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each drug × region series, compute lagged demand and rolling stats.
    Requires the DataFrame to be sorted by (drug_id, region, year, week).
    """
    if df.empty:
        return df

    df = df.sort_values(["drug_id", "distribution_region", "year", "iso_week"])

    # Group key
    group_cols = ["drug_id", "distribution_region"]

    # Lag features
    for lag, col_name in [(1, "demand_lag_1w"), (2, "demand_lag_2w"), (4, "demand_lag_4w")]:
        df[col_name] = df.groupby(group_cols)["avg_monthly_demand"].shift(lag)

    # Rolling 4-week mean and std
    df["demand_rolling_4w_mean"] = (
        df.groupby(group_cols)["avg_monthly_demand"]
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    )
    df["demand_rolling_4w_std"] = (
        df.groupby(group_cols)["avg_monthly_demand"]
        .transform(lambda s: s.rolling(window=4, min_periods=1).std())
    )

    # Stock / demand ratio (guard against division by zero)
    df["stock_demand_ratio"] = np.where(
        df["avg_monthly_demand"] > 0,
        df["total_stock_received"] / df["avg_monthly_demand"],
        0.0,
    )

    # Composite risk score (mirrors run_model_comparison.py logic)
    df["composite_risk_score"] = (
        df["avg_expiry_rate"] * 0.4
        + df["avg_stockout_probability"] * 0.3
        + (1 - df["avg_supplier_reliability"].fillna(0.5)) * 0.2
        + 0.1  # placeholder for data-quality weight
    )

    # Fill NaN lags with 0 (earliest weeks won't have history)
    lag_cols = [
        "demand_lag_1w", "demand_lag_2w", "demand_lag_4w",
        "demand_rolling_4w_mean", "demand_rolling_4w_std",
    ]
    df[lag_cols] = df[lag_cols].fillna(0)

    logger.info("Lag and rolling features added")
    return df


def _upsert_weekly_data(df: pd.DataFrame) -> int:
    """
    Write weekly aggregated rows to DB.  Uses upsert logic:
    if (drug_id, distribution_region, year, iso_week) already exists, update it;
    otherwise insert.  Returns rows written.
    """
    if df.empty:
        return 0

    written = 0
    with get_session() as session:
        for _, row in df.iterrows():
            existing = (
                session.query(WeeklyAggregatedData)
                .filter_by(
                    drug_id=row["drug_id"],
                    distribution_region=row["distribution_region"],
                    year=int(row["year"]),
                    iso_week=int(row["iso_week"]),
                )
                .first()
            )
            data = row.to_dict()
            # Remove non-DB fields
            data.pop("index", None)

            if existing:
                for key, val in data.items():
                    if hasattr(existing, key) and key not in ("id", "created_at"):
                        setattr(existing, key, val)
            else:
                data["created_at"] = datetime.utcnow()
                session.add(WeeklyAggregatedData(**{
                    k: v for k, v in data.items()
                    if hasattr(WeeklyAggregatedData, k)
                }))
            written += 1

    logger.info("Upserted %d weekly aggregated rows", written)
    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_feature_engineering() -> int:
    """
    Main entry point. Aggregates cleaned data to weekly level, adds lag /
    rolling features, and writes to DB. Returns rows written.
    """
    df = _load_cleaned_data()
    if df.empty:
        logger.info("No cleaned data available for feature engineering.")
        return 0

    weekly = _aggregate_to_weekly(df)
    weekly = _add_lag_features(weekly)
    return _upsert_weekly_data(weekly)
