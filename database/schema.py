"""
Database Schema (ORM Models)
=============================
All tables used by MedSupply AI are declared here via SQLAlchemy ORM.

Tables
------
1. **raw_stock_data**         – every ingested row lands here (append-only).
2. **cleaned_stock_data**     – preprocessed + validated rows ready for ML.
3. **weekly_aggregated_data** – aggregated to ISO-week level with features.
4. **model_versions**         – metadata for each saved model file.
5. **predictions_cache**      – latest predictions served to the dashboard.
6. **pipeline_runs**          – audit log of every pipeline execution.

Integration point
-----------------
Imported by database/db_connection.py::init_db() to create tables on startup.
Used by pipeline modules and the API to read/write via SQLAlchemy ORM.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, Float, String, Text, DateTime,
    Boolean, Date, UniqueConstraint, Index,
)
from database.db_connection import Base


# ---------------------------------------------------------------------------
# 1. Raw ingested stock data
# ---------------------------------------------------------------------------
class RawStockData(Base):
    __tablename__ = "raw_stock_data"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    # Source tracking
    source_file = Column(String(512), nullable=False, comment="File path or Google Sheet ID")
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Core fields mirroring the CSV schema
    drug_id = Column(String(64))
    drug_name = Column(String(256))
    manufacturer_country = Column(String(128))
    license_holder = Column(String(256))
    distribution_region = Column(String(128))
    facility_type = Column(String(64))
    initial_stock_units = Column(Float)
    stock_received_date = Column(Date)
    reorder_level = Column(Float)
    average_monthly_demand = Column(Float)
    delivery_frequency_days = Column(Float)
    lead_time_days = Column(Float)
    supplier_reliability_score = Column(Float)
    region_disease_outbreaks = Column(Integer)
    season = Column(String(32))
    transport_accessibility_score = Column(Float)
    power_stability_index = Column(Float)
    staff_availability_index = Column(Float)
    data_record_quality = Column(String(32))
    storage_temperature = Column(Float)
    storage_humidity = Column(Float)
    FEFO_policy_implemented = Column(Integer)
    warehouse_capacity_utilization = Column(Float)
    storage_condition_rating = Column(String(32))
    stockout_occurred = Column(Integer)
    expiry_rate_percent = Column(Float)
    forecast_error_percent = Column(Float)
    financial_loss_due_to_expiry_usd = Column(Float)
    delivery_delay_days = Column(Float)
    predicted_stockout_probability = Column(Float)
    expiry_risk_category = Column(String(32))
    data_source = Column(String(128))

    # Prevent exact duplicate rows from the same source
    __table_args__ = (
        UniqueConstraint(
            "drug_id", "stock_received_date", "facility_type",
            "distribution_region", "source_file",
            name="uq_raw_stock_row",
        ),
        Index("ix_raw_drug_date", "drug_id", "stock_received_date"),
    )


# ---------------------------------------------------------------------------
# 2. Cleaned / validated stock data
# ---------------------------------------------------------------------------
class CleanedStockData(Base):
    __tablename__ = "cleaned_stock_data"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    raw_id = Column(BigInteger, nullable=True, comment="FK reference to raw_stock_data.id")
    cleaned_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    drug_id = Column(String(64))
    drug_name = Column(String(256))
    distribution_region = Column(String(128))
    facility_type = Column(String(64))
    initial_stock_units = Column(Float)
    stock_received_date = Column(Date)
    reorder_level = Column(Float)
    average_monthly_demand = Column(Float)
    delivery_frequency_days = Column(Float)
    lead_time_days = Column(Float)
    supplier_reliability_score = Column(Float)
    region_disease_outbreaks = Column(Integer)
    season = Column(String(32))
    transport_accessibility_score = Column(Float)
    power_stability_index = Column(Float)
    staff_availability_index = Column(Float)
    data_record_quality = Column(String(32))
    FEFO_policy_implemented = Column(Integer)
    warehouse_capacity_utilization = Column(Float)
    storage_condition_rating = Column(String(32))
    stockout_occurred = Column(Integer)
    expiry_rate_percent = Column(Float)
    predicted_stockout_probability = Column(Float)
    expiry_risk_category = Column(String(32))

    # Year / ISO-week extracted during preprocessing for weekly aggregation
    year = Column(Integer)
    month = Column(Integer)
    iso_week = Column(Integer)

    __table_args__ = (
        Index("ix_clean_drug_week", "drug_id", "year", "iso_week"),
    )


# ---------------------------------------------------------------------------
# 3. Weekly aggregated data (feature-engineered, ML-ready)
# ---------------------------------------------------------------------------
class WeeklyAggregatedData(Base):
    __tablename__ = "weekly_aggregated_data"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    drug_id = Column(String(64), nullable=False)
    drug_name = Column(String(256))
    distribution_region = Column(String(128))
    year = Column(Integer, nullable=False)
    iso_week = Column(Integer, nullable=False)

    # Aggregated metrics
    total_stock_received = Column(Float)
    avg_monthly_demand = Column(Float)
    avg_reorder_level = Column(Float)
    avg_lead_time_days = Column(Float)
    avg_supplier_reliability = Column(Float)
    stockout_count = Column(Integer)
    avg_expiry_rate = Column(Float)
    avg_stockout_probability = Column(Float)
    record_count = Column(Integer, comment="How many raw rows aggregated into this week")

    # Feature-engineered columns (populated by feature_engineering.py)
    demand_lag_1w = Column(Float, comment="Demand from 1 week ago")
    demand_lag_2w = Column(Float, comment="Demand from 2 weeks ago")
    demand_lag_4w = Column(Float, comment="Demand from 4 weeks ago")
    demand_rolling_4w_mean = Column(Float, comment="4-week rolling avg demand")
    demand_rolling_4w_std = Column(Float, comment="4-week rolling std demand")
    stock_demand_ratio = Column(Float, comment="stock_received / demand")
    composite_risk_score = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("drug_id", "distribution_region", "year", "iso_week",
                         name="uq_weekly_agg"),
        Index("ix_weekly_drug_week", "drug_id", "year", "iso_week"),
    )


# ---------------------------------------------------------------------------
# 4. Model version registry
# ---------------------------------------------------------------------------
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    model_name = Column(String(128), nullable=False, comment="e.g. xgboost_demand, lstm_demand")
    version = Column(String(64), nullable=False, comment="Timestamp-based: 20260217_020000")
    file_path = Column(String(512), nullable=False)
    metrics_json = Column(Text, comment="JSON blob of evaluation metrics")
    is_active = Column(Boolean, default=False, comment="Currently served model?")
    trained_on_rows = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_version"),
        Index("ix_model_active", "model_name", "is_active"),
    )


# ---------------------------------------------------------------------------
# 5. Predictions cache – latest predictions served to the dashboard
# ---------------------------------------------------------------------------
class PredictionCache(Base):
    __tablename__ = "predictions_cache"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    drug_id = Column(String(64), nullable=False)
    drug_name = Column(String(256))
    distribution_region = Column(String(128))
    facility_type = Column(String(64))

    predicted_demand = Column(Float)
    predicted_stockout_probability = Column(Float)
    recommended_order_qty = Column(Float)
    stockout_risk_level = Column(String(32))  # Low / Medium / High

    current_stock = Column(Float)
    model_version = Column(String(64))
    predicted_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_pred_drug_region", "drug_id", "distribution_region"),
    )


# ---------------------------------------------------------------------------
# 6. Pipeline run audit log
# ---------------------------------------------------------------------------
class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, unique=True, comment="UUID for this run")
    stage = Column(String(64), nullable=False, comment="ingestion | preprocessing | feature_eng | retrain")
    status = Column(String(32), nullable=False, comment="started | success | failed")
    records_processed = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
