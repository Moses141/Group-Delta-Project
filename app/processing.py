from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from pmdarima import auto_arima
except Exception:  # pragma: no cover
    auto_arima = None


@dataclass
class RefreshOutputs:
    monthly_demand: pd.DataFrame
    stock_status: pd.DataFrame
    forecast_next_3: pd.DataFrame
    sarima_orders: dict


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


def load_or_build_sarima_orders(outputs_dir: Path, monthly: pd.DataFrame, seasonal_period: int = 12) -> dict:
    orders_path = outputs_dir / "sarima_orders.json"
    if orders_path.exists():
        with open(orders_path, "r") as f:
            raw = json.load(f)
        # Normalize to tuples
        out = {}
        for did, v in raw.items():
            out[did] = {"order": tuple(v["order"]), "seasonal_order": tuple(v["seasonal_order"])}
        return out

    if auto_arima is None:
        raise RuntimeError(
            "sarima_orders.json not found and pmdarima is not installed. "
            "Install pmdarima or provide outputs/sarima_orders.json."
        )

    # Build orders via auto_arima (one-time setup)
    orders = {}
    for did in monthly["drug_id"].unique():
        y = monthly[monthly["drug_id"] == did].set_index("month")["monthly_demand"].asfreq("MS").fillna(0)
        model = auto_arima(
            y,
            seasonal=True,
            m=seasonal_period,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        orders[did] = {"order": model.order, "seasonal_order": model.seasonal_order}

    # Save
    serializable = {k: {"order": list(v["order"]), "seasonal_order": list(v["seasonal_order"])} for k, v in orders.items()}
    with open(orders_path, "w") as f:
        json.dump(serializable, f, indent=2)
    return orders


def run_forecast_pipeline(monthly: pd.DataFrame, outputs_dir: Path, horizon: int = 3) -> Tuple[pd.DataFrame, dict]:
    monthly = monthly.copy()
    monthly["month"] = pd.to_datetime(monthly["month"])
    orders = load_or_build_sarima_orders(outputs_dir, monthly, seasonal_period=12)

    rows = []
    for did in monthly["drug_id"].unique():
        y = monthly[monthly["drug_id"] == did].set_index("month")["monthly_demand"].asfreq("MS").fillna(0)
        spec = orders.get(did)
        if spec is None:
            # Fallback simple spec (keeps pipeline running)
            spec = {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 12)}
        model = SARIMAX(y, order=spec["order"], seasonal_order=spec["seasonal_order"])
        fitted = model.fit(disp=False)
        fc = fitted.forecast(steps=horizon)
        for t, v in fc.items():
            rows.append({"drug_id": did, "forecast_month": pd.to_datetime(t), "predicted_demand": float(v)})

    forecast_df = pd.DataFrame(rows).sort_values(["drug_id", "forecast_month"]).reset_index(drop=True)
    forecast_df.to_csv(outputs_dir / "next_3_month_forecast.csv", index=False)
    return forecast_df, orders


def save_outputs(outputs_dir: Path, monthly: pd.DataFrame, stock_status: pd.DataFrame):
    outputs_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(outputs_dir / "monthly_demand.csv", index=False)
    stock_status.to_csv(outputs_dir / "stock_status.csv", index=False)

    meta = {"last_refresh_utc": pd.Timestamp.utcnow().isoformat()}
    (outputs_dir / "last_refresh.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def refresh_all(data_dir: Path, outputs_dir: Path) -> RefreshOutputs:
    sales, stock, opening = load_raw_data(data_dir)
    sales = clean_sales_data(sales)
    stock = clean_stock_data(stock)
    opening = clean_opening_stock_data(opening)

    monthly = ensure_continuous_months(create_monthly_demand(sales))
    stock_status = create_stock_status(opening, stock, sales)
    save_outputs(outputs_dir, monthly, stock_status)

    forecast_df, orders = run_forecast_pipeline(monthly, outputs_dir, horizon=3)
    return RefreshOutputs(monthly, stock_status, forecast_df, orders)

