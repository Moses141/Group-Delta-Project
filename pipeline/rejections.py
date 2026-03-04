"""Structured rejected-row logging utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_DIR

REJECTED_ROWS_PATH = Path(LOG_DIR) / "rejected_rows.jsonl"


def log_rejected_row(stage: str, row_data: dict, error_message: str) -> None:
    REJECTED_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "error_message": str(error_message),
        "row_data": row_data,
    }
    with REJECTED_ROWS_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, default=str) + "\n")
