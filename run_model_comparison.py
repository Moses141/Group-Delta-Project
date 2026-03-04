"""Leakage-safe model comparison for demand forecasting."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml.train import train_demand_pipeline


if __name__ == "__main__":
    df = pd.read_csv("uganda_drug_supply_synthetic.csv")

    trained_rf = train_demand_pipeline(
        df=df,
        model=RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        test_size=0.2,
    )

    print("=" * 80)
    print("LEAKAGE-SAFE MODEL COMPARISON (CHRONOLOGICAL HOLDOUT)")
    print("=" * 80)
    print(f"Train period: {trained_rf['train_period']}")
    print(f"Test period:  {trained_rf['test_period']}")
    print(f"Train rows: {trained_rf['train_rows']} | Test rows: {trained_rf['test_rows']}")
    print("Metrics:")
    for key, value in trained_rf["metrics"].items():
        print(f"  {key}: {value:.6f}")
