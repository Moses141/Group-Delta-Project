"""
Shared monthly LSTM feature engineering (seasonality + expiry context).
Used by processing (pipeline) and forecasting (training/inference fallbacks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# When no future expiry exists for a drug/month, treat as "far" (~30-day month units)
DEFAULT_MONTHS_TO_NEAREST_EXPIRY = 12.0

LSTM_FEATURE_COLUMNS = [
    "monthly_demand",
    "month_sin",
    "month_cos",
    "months_to_nearest_expiry",
]


def enrich_monthly_with_lstm_features(monthly: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic month encoding and months to nearest future expiry per drug/month.
    If expiry data is missing, months_to_nearest_expiry defaults to a constant.
    """
    df = monthly.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["month_num"] = df["month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)

    if stock.empty or "expiry_date" not in stock.columns:
        df["months_to_nearest_expiry"] = DEFAULT_MONTHS_TO_NEAREST_EXPIRY
        return df.drop(columns=["month_num"], errors="ignore")

    st = stock.copy()
    st["drug_id"] = st["drug_id"].astype(str)
    st["expiry_date"] = pd.to_datetime(st["expiry_date"], errors="coerce")
    st = st.dropna(subset=["expiry_date"])

    expiry_by_drug: dict[str, np.ndarray] = {}
    for did, grp in st.groupby("drug_id"):
        expiry_by_drug[str(did)] = np.sort(grp["expiry_date"].unique())

    mte: list[float] = []
    for _, row in df.iterrows():
        did = str(row["drug_id"])
        month_start = pd.Timestamp(row["month"]).normalize()
        exps = expiry_by_drug.get(did)
        if exps is None or len(exps) == 0:
            mte.append(DEFAULT_MONTHS_TO_NEAREST_EXPIRY)
            continue
        future = exps[exps >= month_start]
        if len(future) == 0:
            mte.append(DEFAULT_MONTHS_TO_NEAREST_EXPIRY)
            continue
        nearest = future[0]
        days = float((nearest - month_start).days)
        mte.append(max(0.0, days / 30.0))

    df["months_to_nearest_expiry"] = mte
    return df.drop(columns=["month_num"], errors="ignore")


def months_to_nearest_expiry_at(stock: pd.DataFrame, drug_id: str, month_start: pd.Timestamp) -> float:
    """Nearest future expiry distance (30-day month units) for one drug at month start; default if none."""
    if stock.empty or "expiry_date" not in stock.columns:
        return DEFAULT_MONTHS_TO_NEAREST_EXPIRY
    st = stock[stock["drug_id"].astype(str) == str(drug_id)]
    if st.empty:
        return DEFAULT_MONTHS_TO_NEAREST_EXPIRY
    exps = pd.to_datetime(st["expiry_date"], errors="coerce").dropna()
    if exps.empty:
        return DEFAULT_MONTHS_TO_NEAREST_EXPIRY
    month_start = pd.Timestamp(month_start).normalize()
    future = np.sort(exps[exps >= month_start].unique())
    if len(future) == 0:
        return DEFAULT_MONTHS_TO_NEAREST_EXPIRY
    days = float((pd.Timestamp(future[0]) - month_start).days)
    return max(0.0, days / 30.0)


def month_sin_cos_for_timestamp(ts: pd.Timestamp) -> tuple[float, float]:
    m = int(pd.Timestamp(ts).month)
    return (
        float(np.sin(2 * np.pi * m / 12.0)),
        float(np.cos(2 * np.pi * m / 12.0)),
    )
