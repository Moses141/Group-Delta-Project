from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import pandas as pd


@dataclass
class ValidationResult:
    ok: bool
    message: str
    df_preview: Optional[pd.DataFrame] = None


def _read_uploaded_csv(file) -> pd.DataFrame:
    """
    Read a Streamlit UploadedFile (or file-like) into a DataFrame.
    """
    if file is None:
        return pd.DataFrame()
    # Streamlit UploadedFile supports getvalue()
    if hasattr(file, "getvalue"):
        data = file.getvalue()
        return pd.read_csv(BytesIO(data))
    return pd.read_csv(file)


def validate_sales_upload(file) -> ValidationResult:
    required = {"transaction_date", "drug_id", "drug_name", "quantity_dispensed"}
    try:
        df = _read_uploaded_csv(file)
    except Exception as e:
        return ValidationResult(False, f"Could not read sales CSV: {e}")

    missing = required - set(df.columns)
    if missing:
        return ValidationResult(False, f"Sales file is missing required columns: {sorted(missing)}")

    # Parseability checks
    try:
        _ = pd.to_datetime(df["transaction_date"], errors="raise")
    except Exception as e:
        return ValidationResult(False, f"Sales `transaction_date` could not be parsed as dates: {e}")

    numeric = pd.to_numeric(df["quantity_dispensed"], errors="coerce")
    if numeric.isna().mean() > 0.05:
        return ValidationResult(False, "Sales `quantity_dispensed` has too many non-numeric values.")

    msg = "Sales file looks valid."
    if "transaction_id" not in df.columns:
        msg += " Optional `transaction_id` missing — deduplication will use drug_id + transaction_date + quantity_dispensed."
    elif df["transaction_id"].isna().any():
        msg += " Some rows lack `transaction_id` — deduplication falls back to drug_id + transaction_date + quantity_dispensed for those rows."

    return ValidationResult(True, msg, df.head(10))


def validate_stock_receipts_upload(file) -> ValidationResult:
    required = {"stock_received_date", "drug_id", "drug_name", "quantity_received"}
    try:
        df = _read_uploaded_csv(file)
    except Exception as e:
        return ValidationResult(False, f"Could not read stock receipts CSV: {e}")

    missing = required - set(df.columns)
    if missing:
        return ValidationResult(False, f"Stock receipts file is missing required columns: {sorted(missing)}")

    try:
        _ = pd.to_datetime(df["stock_received_date"], errors="raise")
    except Exception as e:
        return ValidationResult(False, f"Stock receipts `stock_received_date` could not be parsed as dates: {e}")

    numeric = pd.to_numeric(df["quantity_received"], errors="coerce")
    if numeric.isna().mean() > 0.05:
        return ValidationResult(False, "Stock receipts `quantity_received` has too many non-numeric values.")

    msg = "Stock receipts file looks valid."
    if "stock_id" not in df.columns:
        msg += " Optional `stock_id` missing — deduplication will use drug_id + stock_received_date + quantity_received (+ batch_number if present)."
    elif df["stock_id"].isna().any():
        msg += " Some rows lack `stock_id` — deduplication falls back to drug_id + date + quantity (+ batch) for those rows."

    return ValidationResult(True, msg, df.head(10))


def validate_opening_stock_upload(file) -> ValidationResult:
    required = {"drug_id", "drug_name", "opening_stock_units"}
    try:
        df = _read_uploaded_csv(file)
    except Exception as e:
        return ValidationResult(False, f"Could not read opening stock CSV: {e}")

    missing = required - set(df.columns)
    if missing:
        return ValidationResult(False, f"Opening stock file is missing required columns: {sorted(missing)}")

    numeric = pd.to_numeric(df["opening_stock_units"], errors="coerce")
    if numeric.isna().mean() > 0.05:
        return ValidationResult(False, "Opening stock `opening_stock_units` has too many non-numeric values.")

    return ValidationResult(True, "Opening stock file looks valid.", df.head(10))

