from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model

from app.utils import mape


@dataclass
class LSTMArtifacts:
    model_path: Path
    scaler_path: Path
    sequence_path: Path


def _build_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            LSTM(50, input_shape=input_shape),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _create_drug_sequences(values: np.ndarray, seq_len: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(seq_len, len(values)):
        x.append(values[i - seq_len : i])
        y.append(values[i])
    if len(x) == 0:
        return np.empty((0, seq_len, 1), dtype=float), np.empty((0,), dtype=float)
    return np.array(x, dtype=float).reshape(-1, seq_len, 1), np.array(y, dtype=float)


def prepare_global_sequences(monthly_demand: pd.DataFrame, seq_len: int = 12) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Build one global LSTM dataset from all drugs.
    """
    monthly_demand = monthly_demand.copy()
    monthly_demand["month"] = pd.to_datetime(monthly_demand["month"])
    monthly_demand = monthly_demand.sort_values(["drug_id", "month"])

    scaler = MinMaxScaler()
    all_values = monthly_demand[["monthly_demand"]].astype(float).values
    scaler.fit(all_values)

    x_all, y_all = [], []
    for did in monthly_demand["drug_id"].unique():
        sub = monthly_demand[monthly_demand["drug_id"] == did]
        vals = sub["monthly_demand"].astype(float).values.reshape(-1, 1)
        vals_scaled = scaler.transform(vals).flatten()
        x_d, y_d = _create_drug_sequences(vals_scaled, seq_len=seq_len)
        if len(x_d) > 0:
            x_all.append(x_d)
            y_all.append(y_d)

    if len(x_all) == 0:
        return np.empty((0, seq_len, 1), dtype=float), np.empty((0,), dtype=float), scaler

    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return x, y, scaler


def train_global_lstm(
    monthly_demand: pd.DataFrame,
    model_path: Path,
    scaler_path: Path,
    sequence_path: Path,
    seq_len: int = 12,
    epochs: int = 30,
    batch_size: int = 16,
) -> dict:
    x, y, scaler = prepare_global_sequences(monthly_demand, seq_len=seq_len)
    if len(x) == 0:
        raise ValueError("Not enough monthly data to create LSTM sequences.")

    # Time-based split
    split = int(len(x) * 0.8)
    if split <= 0 or split >= len(x):
        split = max(1, len(x) - 1)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model = _build_model((seq_len, 1))
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=0,
    )

    # Save model and scaler
    model.save(model_path)
    np.save(scaler_path, {"min_": scaler.min_, "scale_": scaler.scale_, "data_min_": scaler.data_min_, "data_max_": scaler.data_max_, "data_range_": scaler.data_range_}, allow_pickle=True)

    # Save sequences
    np.save(sequence_path, {"X": x, "y": y}, allow_pickle=True)

    # Evaluate (inverse transform)
    y_pred = model.predict(x_test, verbose=0).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    metrics = {
        "MAE": float(mean_absolute_error(y_test_inv, y_pred_inv)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))),
        "MAPE": float(mape(y_test_inv, y_pred_inv)),
    }
    return metrics


def _load_scaler(scaler_path: Path) -> MinMaxScaler:
    payload = np.load(scaler_path, allow_pickle=True).item()
    scaler = MinMaxScaler()
    scaler.min_ = payload["min_"]
    scaler.scale_ = payload["scale_"]
    scaler.data_min_ = payload["data_min_"]
    scaler.data_max_ = payload["data_max_"]
    scaler.data_range_ = payload["data_range_"]
    scaler.n_features_in_ = 1
    return scaler


def recursive_forecast_next_3(
    monthly_demand: pd.DataFrame,
    model_path: Path,
    scaler_path: Path,
    seq_len: int = 12,
    horizon: int = 3,
) -> pd.DataFrame:
    monthly_demand = monthly_demand.copy()
    monthly_demand["month"] = pd.to_datetime(monthly_demand["month"])
    monthly_demand = monthly_demand.sort_values(["drug_id", "month"])

    model = load_model(model_path)
    scaler = _load_scaler(scaler_path)

    rows = []
    for did in monthly_demand["drug_id"].unique():
        sub = monthly_demand[monthly_demand["drug_id"] == did].sort_values("month")
        values = sub["monthly_demand"].astype(float).values
        if len(values) < seq_len:
            continue

        window = scaler.transform(values.reshape(-1, 1)).flatten()[-seq_len:].tolist()
        last_month = sub["month"].max()
        for step in range(1, horizon + 1):
            x_input = np.array(window[-seq_len:], dtype=float).reshape(1, seq_len, 1)
            pred_scaled = float(model.predict(x_input, verbose=0)[0][0])
            window.append(pred_scaled)
            pred = float(scaler.inverse_transform(np.array([[pred_scaled]])).flatten()[0])
            rows.append(
                {
                    "drug_id": did,
                    "forecast_month": (last_month + pd.DateOffset(months=step)).strftime("%Y-%m-%d"),
                    "predicted_demand": round(max(0.0, pred), 2),
                }
            )

    return pd.DataFrame(rows).sort_values(["drug_id", "forecast_month"]).reset_index(drop=True)

