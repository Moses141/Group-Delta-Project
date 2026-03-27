from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from app.forecasting import recursive_forecast_next_3, train_global_lstm
from app.paths import MODELS_DIR, OUTPUTS_DIR, ensure_directories
from app.utils import save_json


@dataclass
class RefreshOutputs:
    monthly_demand: pd.DataFrame
    stock_status: pd.DataFrame
    forecast_next_3: pd.DataFrame
    evaluation_metrics: pd.DataFrame


def load_raw_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(data_dir / "sales_transactions.csv")
    stock = pd.read_csv(data_dir / "stock_receipts.csv")
    opening = pd.read_csv(data_dir / "opening_stock.csv")
    return sales, stock, opening


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["drug_id"] = df["drug_id"].astype(str)
    df["drug_name"] = df["drug_name"].astype(str)
    if "category" in df.columns:
        df["category"] = df["category"].astype(str)
    df["quantity_dispensed"] = pd.to_numeric(df["quantity_dispensed"], errors="coerce").fillna(0).astype(int)
    return df


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stock_received_date"] = pd.to_datetime(df["stock_received_date"])
    df["drug_id"] = df["drug_id"].astype(str)
    df["drug_name"] = df["drug_name"].astype(str)
    df["quantity_received"] = pd.to_numeric(df["quantity_received"], errors="coerce").fillna(0).astype(int)
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    return df


def clean_opening_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["drug_id"] = df["drug_id"].astype(str)
    df["drug_name"] = df["drug_name"].astype(str)
    df["opening_stock_units"] = pd.to_numeric(df["opening_stock_units"], errors="coerce").fillna(0).astype(int)
    return df


def create_monthly_demand(sales: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        sales.groupby(["drug_id", pd.Grouper(key="transaction_date", freq="MS")])["quantity_dispensed"]
        .sum()
        .reset_index()
        .rename(columns={"transaction_date": "month", "quantity_dispensed": "monthly_demand"})
    )
    return monthly


def ensure_continuous_months(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.copy()
    monthly["month"] = pd.to_datetime(monthly["month"])
    all_months = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    rows = []
    for did in monthly["drug_id"].unique():
        sub = monthly[monthly["drug_id"] == did].set_index("month").reindex(all_months, fill_value=0)
        sub["drug_id"] = did
        sub = sub.reset_index().rename(columns={"index": "month"})
        sub["monthly_demand"] = pd.to_numeric(sub["monthly_demand"], errors="coerce").fillna(0).astype(int)
        rows.append(sub[["drug_id", "month", "monthly_demand"]])
    out = pd.concat(rows, ignore_index=True).sort_values(["drug_id", "month"]).reset_index(drop=True)
    return out


def create_stock_status(opening: pd.DataFrame, stock: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    total_received = (
        stock.groupby(["drug_id", "drug_name"], as_index=False)["quantity_received"]
        .sum()
        .rename(columns={"quantity_received": "total_received"})
    )
    total_dispensed = (
        sales.groupby(["drug_id", "drug_name"], as_index=False)["quantity_dispensed"]
        .sum()
        .rename(columns={"quantity_dispensed": "total_dispensed"})
    )
    status = opening.merge(total_received, on=["drug_id", "drug_name"], how="left").merge(
        total_dispensed, on=["drug_id", "drug_name"], how="left"
    )
    status["total_received"] = status["total_received"].fillna(0).astype(int)
    status["total_dispensed"] = status["total_dispensed"].fillna(0).astype(int)
    status["current_stock"] = (
        status["opening_stock_units"].astype(int)
        + status["total_received"].astype(int)
        - status["total_dispensed"].astype(int)
    )
    status["current_stock"] = status["current_stock"].clip(lower=0).astype(int)
    return status[["drug_id", "drug_name", "opening_stock_units", "total_received", "total_dispensed", "current_stock"]]


def save_outputs(monthly: pd.DataFrame, stock_status: pd.DataFrame, forecast_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    ensure_directories()
    monthly.to_csv(OUTPUTS_DIR / "monthly_demand.csv", index=False)
    stock_status.to_csv(OUTPUTS_DIR / "stock_status.csv", index=False)
    forecast_df.to_csv(OUTPUTS_DIR / "next_3_month_forecast.csv", index=False)
    metrics_df.to_csv(OUTPUTS_DIR / "evaluation_metrics.csv", index=False)
    save_json(
        OUTPUTS_DIR / "last_refresh.json",
        {"last_refresh_utc": pd.Timestamp.utcnow().isoformat(), "forecasting_model": "LSTM"},
    )


def refresh_all(data_dir: Path, outputs_dir: Path | None = None) -> RefreshOutputs:
    # outputs_dir kept for compatibility with existing dashboard calls
    _ = outputs_dir
    ensure_directories()

    sales, stock, opening = load_raw_data(data_dir)
    sales = clean_sales_data(sales)
    stock = clean_stock_data(stock)
    opening = clean_opening_stock_data(opening)

    monthly = ensure_continuous_months(create_monthly_demand(sales))
    stock_status = create_stock_status(opening, stock, sales)

    model_path = MODELS_DIR / "lstm_model.h5"
    scaler_path = MODELS_DIR / "lstm_scaler.npy"
    seq_path = OUTPUTS_DIR / "lstm_sequences.npy"

    metrics = train_global_lstm(
        monthly_demand=monthly,
        model_path=model_path,
        scaler_path=scaler_path,
        sequence_path=seq_path,
        seq_len=12,
        epochs=30,
        batch_size=16,
    )
    forecast_df = recursive_forecast_next_3(
        monthly_demand=monthly,
        model_path=model_path,
        scaler_path=scaler_path,
        seq_len=12,
        horizon=3,
    )
    metrics_df = pd.DataFrame([metrics])

    save_outputs(monthly, stock_status, forecast_df, metrics_df)
    return RefreshOutputs(monthly, stock_status, forecast_df, metrics_df)

