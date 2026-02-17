"""
Data Ingestion Service
=======================
Responsible for pulling stock data from three sources:
  1. Local CSV files
  2. Local Excel (.xlsx) files
  3. Google Sheets via Google Sheets API (service-account auth)

Every ingested row is written to the ``raw_stock_data`` table in the DB.
Duplicate prevention uses the composite unique constraint on
(drug_id, stock_received_date, facility_type, distribution_region, source_file).

Integration points
------------------
- Called by ``pipeline/scheduler.py`` on the weekly cron.
- Can also be triggered manually via the FastAPI ``/api/pipeline/ingest`` endpoint.
- Writes to ``database.schema.RawStockData``.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import INGEST_WATCH_DIR, GOOGLE_SHEETS_CREDENTIALS_FILE, GOOGLE_SHEET_IDS, SEED_CSV
from database.db_connection import get_session, engine
from database.schema import RawStockData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mapping: CSV header -> DB column
# ---------------------------------------------------------------------------
# The CSV ships with these headers; we normalise to snake_case DB column names.
COLUMN_MAP = {
    "drug_id": "drug_id",
    "drug_name": "drug_name",
    "manufacturer_country": "manufacturer_country",
    "license_holder": "license_holder",
    "distribution_region": "distribution_region",
    "facility_type": "facility_type",
    "initial_stock_units": "initial_stock_units",
    "stock_received_date": "stock_received_date",
    "reorder_level": "reorder_level",
    "average_monthly_demand": "average_monthly_demand",
    "delivery_frequency_days": "delivery_frequency_days",
    "lead_time_days": "lead_time_days",
    "supplier_reliability_score": "supplier_reliability_score",
    "region_disease_outbreaks": "region_disease_outbreaks",
    "season": "season",
    "transport_accessibility_score": "transport_accessibility_score",
    "power_stability_index": "power_stability_index",
    "staff_availability_index": "staff_availability_index",
    "data_record_quality": "data_record_quality",
    "storage_temperature": "storage_temperature",
    "storage_humidity": "storage_humidity",
    "FEFO_policy_implemented": "FEFO_policy_implemented",
    "warehouse_capacity_utilization": "warehouse_capacity_utilization",
    "storage_condition_rating": "storage_condition_rating",
    "stockout_occurred": "stockout_occurred",
    "expiry_rate_percent": "expiry_rate_percent",
    "forecast_error_percent": "forecast_error_percent",
    "financial_loss_due_to_expiry_usd": "financial_loss_due_to_expiry_usd",
    "delivery_delay_days": "delivery_delay_days",
    "predicted_stockout_probability": "predicted_stockout_probability",
    "expiry_risk_category": "expiry_risk_category",
    "data_source": "data_source",
}

DB_COLUMNS = list(COLUMN_MAP.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_df(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Rename columns, drop unknown columns, and add ``source_file`` tag.
    Returns a DataFrame whose columns match the RawStockData ORM model.
    """
    # Lowercase all column names for consistent matching
    df.columns = [c.strip() for c in df.columns]

    # Keep only recognised columns
    known = [c for c in df.columns if c in COLUMN_MAP]
    df = df[known].rename(columns=COLUMN_MAP)

    # Parse date column
    if "stock_received_date" in df.columns:
        df["stock_received_date"] = pd.to_datetime(
            df["stock_received_date"], errors="coerce"
        ).dt.date

    df["source_file"] = source_label
    return df


def _insert_rows(df: pd.DataFrame) -> int:
    """
    Bulk-insert rows into raw_stock_data, skipping duplicates.
    Uses savepoints so a single bad row doesn't abort the whole batch.
    Returns the number of rows actually inserted.
    """
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    inserted = 0

    with get_session() as session:
        for rec in records:
            sp = session.begin_nested()  # SAVEPOINT
            try:
                obj = RawStockData(**rec)
                session.add(obj)
                session.flush()
                sp.commit()
                inserted += 1
            except Exception:
                sp.rollback()  # rollback only this row
                continue

    logger.info("Inserted %d / %d rows from '%s'", inserted, len(records), df["source_file"].iloc[0])
    return inserted


# ---------------------------------------------------------------------------
# 1. CSV Ingestion
# ---------------------------------------------------------------------------
def ingest_csv(file_path: str) -> int:
    """Read a CSV file into the raw_stock_data table. Returns rows inserted."""
    logger.info("Ingesting CSV: %s", file_path)
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df = _normalize_df(df, source_label=str(file_path))
        return _insert_rows(df)
    except Exception as e:
        logger.error("CSV ingestion failed for %s: %s", file_path, e)
        raise


# ---------------------------------------------------------------------------
# 2. Excel Ingestion
# ---------------------------------------------------------------------------
def ingest_excel(file_path: str, sheet_name: Optional[str] = None) -> int:
    """Read an Excel file into the raw_stock_data table. Returns rows inserted."""
    logger.info("Ingesting Excel: %s (sheet=%s)", file_path, sheet_name or "first")
    try:
        kwargs = {"sheet_name": sheet_name} if sheet_name else {}
        df = pd.read_excel(file_path, **kwargs)
        # If sheet_name returns a dict of DataFrames, concatenate all sheets
        if isinstance(df, dict):
            df = pd.concat(df.values(), ignore_index=True)
        df = _normalize_df(df, source_label=str(file_path))
        return _insert_rows(df)
    except Exception as e:
        logger.error("Excel ingestion failed for %s: %s", file_path, e)
        raise


# ---------------------------------------------------------------------------
# 3. Google Sheets Ingestion
# ---------------------------------------------------------------------------
def ingest_google_sheet(sheet_id: str, worksheet_name: str = "Sheet1") -> int:
    """
    Pull data from a Google Sheet via the Sheets API using a service-account key.
    Requires ``gspread`` and ``google-auth`` packages.
    Returns rows inserted.
    """
    logger.info("Ingesting Google Sheet: %s / %s", sheet_id, worksheet_name)
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_file(
            GOOGLE_SHEETS_CREDENTIALS_FILE, scopes=scopes
        )
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        df = _normalize_df(df, source_label=f"gsheet:{sheet_id}/{worksheet_name}")
        return _insert_rows(df)
    except ImportError:
        logger.error(
            "gspread / google-auth not installed. "
            "Run: pip install gspread google-auth"
        )
        raise
    except Exception as e:
        logger.error("Google Sheets ingestion failed for %s: %s", sheet_id, e)
        raise


# ---------------------------------------------------------------------------
# 4. Scan watch directory for new files
# ---------------------------------------------------------------------------
def scan_and_ingest_directory(directory: Optional[str] = None) -> int:
    """
    Walk the ingestion watch directory, ingest every CSV and Excel file found.
    Returns total rows inserted across all files.
    """
    watch_dir = Path(directory or INGEST_WATCH_DIR)
    if not watch_dir.exists():
        logger.warning("Watch directory does not exist: %s", watch_dir)
        return 0

    total = 0
    for fpath in sorted(watch_dir.iterdir()):
        if fpath.suffix.lower() == ".csv":
            total += ingest_csv(str(fpath))
        elif fpath.suffix.lower() in (".xlsx", ".xls"):
            total += ingest_excel(str(fpath))

    logger.info("Directory scan complete. Total rows inserted: %d", total)
    return total


# ---------------------------------------------------------------------------
# 5. Ingest all configured Google Sheets
# ---------------------------------------------------------------------------
def ingest_all_google_sheets() -> int:
    """Iterate over GOOGLE_SHEET_IDS from config and ingest each."""
    if not GOOGLE_SHEET_IDS:
        logger.info("No Google Sheet IDs configured – skipping.")
        return 0
    total = 0
    for sid in GOOGLE_SHEET_IDS:
        try:
            total += ingest_google_sheet(sid)
        except Exception as e:
            logger.error("Skipping sheet %s due to error: %s", sid, e)
    return total


# ---------------------------------------------------------------------------
# 6. Seed database from the original CSV (first-run helper)
# ---------------------------------------------------------------------------
def seed_from_original_csv() -> int:
    """
    Load the original ``uganda_drug_supply_synthetic.csv`` that ships with the
    project into the DB.  Safe to call multiple times – duplicates are skipped.
    """
    if not os.path.exists(SEED_CSV):
        logger.warning("Seed CSV not found at %s", SEED_CSV)
        return 0
    return ingest_csv(SEED_CSV)


# ---------------------------------------------------------------------------
# Master ingest function (called by the scheduler)
# ---------------------------------------------------------------------------
def run_full_ingestion() -> int:
    """
    Execute all ingestion sources in sequence:
      1. Seed CSV (idempotent)
      2. Watch-directory files
      3. Google Sheets
    Returns total new rows inserted.
    """
    total = 0
    total += seed_from_original_csv()
    total += scan_and_ingest_directory()
    total += ingest_all_google_sheets()
    logger.info("=== Full ingestion complete: %d new rows ===", total)
    return total
