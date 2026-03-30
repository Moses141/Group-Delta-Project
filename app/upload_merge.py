"""
Incremental CSV uploads: append, dedupe, sort.
Kept separate from processing.py so the dashboard can import without loading TensorFlow.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

UPLOAD_MODE_APPEND = "append"
UPLOAD_MODE_REPLACE = "replace"


def merge_sales_transactions(existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Append then dedupe and sort. Prefer transaction_id; else drug_id + transaction_date + quantity_dispensed."""
    combined = pd.concat([existing, new_data], ignore_index=True, sort=False)
    combined["transaction_date"] = pd.to_datetime(combined["transaction_date"], errors="coerce")
    combined["drug_id"] = combined["drug_id"].astype(str)
    if "transaction_id" in combined.columns:
        tid = combined["transaction_id"]
        valid = tid.notna() & (tid.astype(str).str.strip() != "") & (tid.astype(str).str.lower() != "nan")
        if bool(valid.all()):
            combined = combined.drop_duplicates(subset=["transaction_id"], keep="last")
        else:
            combined = combined.drop_duplicates(
                subset=["drug_id", "transaction_date", "quantity_dispensed"],
                keep="last",
            )
    else:
        combined = combined.drop_duplicates(
            subset=["drug_id", "transaction_date", "quantity_dispensed"],
            keep="last",
        )
    combined = combined.sort_values(["drug_id", "transaction_date"]).reset_index(drop=True)
    return combined


def merge_stock_receipts(existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Append then dedupe and sort. Prefer stock_id; else drug_id + stock_received_date + quantity_received (+ batch if present)."""
    combined = pd.concat([existing, new_data], ignore_index=True, sort=False)
    combined["stock_received_date"] = pd.to_datetime(combined["stock_received_date"], errors="coerce")
    combined["drug_id"] = combined["drug_id"].astype(str)
    if "stock_id" in combined.columns:
        sid = combined["stock_id"]
        valid = sid.notna() & (sid.astype(str).str.strip() != "") & (sid.astype(str).str.lower() != "nan")
        if bool(valid.all()):
            combined = combined.drop_duplicates(subset=["stock_id"], keep="last")
        else:
            subset = ["drug_id", "stock_received_date", "quantity_received"]
            if "batch_number" in combined.columns:
                subset.append("batch_number")
            combined = combined.drop_duplicates(subset=subset, keep="last")
    else:
        subset = ["drug_id", "stock_received_date", "quantity_received"]
        if "batch_number" in combined.columns:
            subset.append("batch_number")
        combined = combined.drop_duplicates(subset=subset, keep="last")
    combined = combined.sort_values(["drug_id", "stock_received_date"]).reset_index(drop=True)
    return combined


def persist_sales_upload(data_dir: Path, upload_bytes: bytes, mode: str) -> None:
    """Write sales_transactions.csv: replace bytes, or append + dedupe + sort when mode is append and file exists."""
    path = data_dir / "sales_transactions.csv"
    new_df = pd.read_csv(BytesIO(upload_bytes))
    if mode == UPLOAD_MODE_APPEND and path.exists():
        existing = pd.read_csv(path)
        merged = merge_sales_transactions(existing, new_df)
        merged.to_csv(path, index=False)
    else:
        path.write_bytes(upload_bytes)


def persist_stock_upload(data_dir: Path, upload_bytes: bytes, mode: str) -> None:
    """Write stock_receipts.csv: replace or append + dedupe + sort."""
    path = data_dir / "stock_receipts.csv"
    new_df = pd.read_csv(BytesIO(upload_bytes))
    if mode == UPLOAD_MODE_APPEND and path.exists():
        existing = pd.read_csv(path)
        merged = merge_stock_receipts(existing, new_df)
        merged.to_csv(path, index=False)
    else:
        path.write_bytes(upload_bytes)
