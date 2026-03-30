from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model

from app.monthly_lstm_features import (
    LSTM_FEATURE_COLUMNS,
    enrich_monthly_with_lstm_features,
    month_sin_cos_for_timestamp,
    months_to_nearest_expiry_at,
)
from app.utils import load_json, mape, save_json


@dataclass
class LSTMArtifacts:
    model_path: Path
    scaler_path: Path
    sequence_path: Path


NUM_LSTM_FEATURES = len(LSTM_FEATURE_COLUMNS)


def _build_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            LSTM(50, input_shape=input_shape),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _create_drug_sequences(feature_matrix: np.ndarray, seq_len: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """feature_matrix: (n_timesteps, n_features), target = next monthly_demand (column 0), scaled."""
    if feature_matrix.shape[0] <= seq_len:
        n_feat = feature_matrix.shape[1] if feature_matrix.ndim == 2 else 1
        return np.empty((0, seq_len, n_feat), dtype=float), np.empty((0,), dtype=float)
    n_feat = feature_matrix.shape[1]
    x, y = [], []
    for i in range(seq_len, len(feature_matrix)):
        x.append(feature_matrix[i - seq_len : i])
        y.append(feature_matrix[i, 0])
    return np.array(x, dtype=float), np.array(y, dtype=float)


def _inverse_first_feature(scaler: MinMaxScaler, y_scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform column 0 only (demand); other columns padded with 0 for multivariate scaler."""
    y_scaled = np.asarray(y_scaled).reshape(-1)
    n = len(y_scaled)
    n_features = int(getattr(scaler, "n_features_in_", NUM_LSTM_FEATURES))
    if n_features <= 1:
        return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    rest = np.zeros((n, n_features - 1))
    stacked = np.column_stack([y_scaled, rest])
    return scaler.inverse_transform(stacked)[:, 0]


def prepare_global_sequences(
    monthly_demand: pd.DataFrame,
    seq_len: int = 12,
    stock_receipts: pd.DataFrame | None = None,
    existing_scaler_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Build one global LSTM dataset from all drugs (multivariate: demand + seasonality + expiry).
    If existing_scaler_path points to a compatible saved scaler, update it with partial_fit on
    current feature rows (incremental bounds); otherwise fit a new scaler on full data.
    """
    monthly_demand = monthly_demand.copy()
    monthly_demand["month"] = pd.to_datetime(monthly_demand["month"])
    monthly_demand = monthly_demand.sort_values(["drug_id", "month"])

    stock = stock_receipts if stock_receipts is not None else pd.DataFrame()
    if not all(c in monthly_demand.columns for c in LSTM_FEATURE_COLUMNS):
        monthly_demand = enrich_monthly_with_lstm_features(monthly_demand, stock)

    all_values = monthly_demand[LSTM_FEATURE_COLUMNS].astype(float).values
    scaler: MinMaxScaler
    if existing_scaler_path is not None and existing_scaler_path.exists():
        try:
            scaler = _load_scaler(existing_scaler_path)
            if int(getattr(scaler, "n_features_in_", 0)) != NUM_LSTM_FEATURES:
                scaler = MinMaxScaler()
                scaler.fit(all_values)
            else:
                try:
                    scaler.partial_fit(all_values)
                except Exception:
                    scaler = MinMaxScaler()
                    scaler.fit(all_values)
        except Exception:
            scaler = MinMaxScaler()
            scaler.fit(all_values)
    else:
        scaler = MinMaxScaler()
        scaler.fit(all_values)

    x_all, y_all = [], []
    for did in monthly_demand["drug_id"].unique():
        sub = monthly_demand[monthly_demand["drug_id"] == did]
        vals = sub[LSTM_FEATURE_COLUMNS].astype(float).values
        vals_scaled = scaler.transform(vals)
        x_d, y_d = _create_drug_sequences(vals_scaled, seq_len=seq_len)
        if len(x_d) > 0:
            x_all.append(x_d)
            y_all.append(y_d)

    if len(x_all) == 0:
        return (
            np.empty((0, seq_len, NUM_LSTM_FEATURES), dtype=float),
            np.empty((0,), dtype=float),
            scaler,
        )

    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return x, y, scaler


def _scaler_payload(scaler: MinMaxScaler) -> dict:
    d = {
        "min_": scaler.min_,
        "scale_": scaler.scale_,
        "data_min_": scaler.data_min_,
        "data_max_": scaler.data_max_,
        "data_range_": scaler.data_range_,
        "n_features_in_": int(getattr(scaler, "n_features_in_", NUM_LSTM_FEATURES)),
    }
    if hasattr(scaler, "n_samples_seen_") and scaler.n_samples_seen_ is not None:
        d["n_samples_seen_"] = scaler.n_samples_seen_
    return d


def train_global_lstm(
    monthly_demand: pd.DataFrame,
    model_path: Path,
    scaler_path: Path,
    sequence_path: Path,
    seq_len: int = 12,
    epochs: int = 30,
    batch_size: int = 16,
    stock_receipts: pd.DataFrame | None = None,
    epochs_finetune: int = 8,
    sequence_growth_full_threshold: float = 2.0,
) -> dict:
    meta_path = _lstm_meta_path(model_path)
    prev_seq_count = 0
    if meta_path.exists():
        try:
            prev_seq_count = int(load_json(meta_path).get("last_sequence_count", 0))
        except Exception:
            prev_seq_count = 0

    existing_scaler = scaler_path if scaler_path.exists() else None
    x, y, scaler = prepare_global_sequences(
        monthly_demand,
        seq_len=seq_len,
        stock_receipts=stock_receipts,
        existing_scaler_path=existing_scaler,
    )
    if len(x) == 0:
        raise ValueError("Not enough monthly data to create LSTM sequences.")

    n_features = x.shape[2]
    last_n = len(x)

    force_full_retrain = (
        prev_seq_count > 0
        and last_n >= prev_seq_count * sequence_growth_full_threshold
    )

    # Time-based split
    split = int(len(x) * 0.8)
    if split <= 0 or split >= len(x):
        split = max(1, len(x) - 1)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    loaded = None if force_full_retrain else _load_model_if_compatible(model_path, n_features)
    if loaded is not None:
        model = loaded
        model.compile(optimizer="adam", loss="mse")
        epochs_use = epochs_finetune
        train_mode = "finetune"
    else:
        model = _build_model((seq_len, n_features))
        epochs_use = epochs
        train_mode = "full"

    model.fit(
        x_train,
        y_train,
        epochs=epochs_use,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=0,
    )

    model.save(model_path)
    np.save(scaler_path, _scaler_payload(scaler), allow_pickle=True)

    np.save(sequence_path, {"X": x, "y": y}, allow_pickle=True)

    save_json(
        meta_path,
        {
            "last_sequence_count": last_n,
            "train_mode": train_mode,
            "epochs_used": epochs_use,
        },
    )

    y_pred = model.predict(x_test, verbose=0).flatten()
    y_test_inv = _inverse_first_feature(scaler, y_test)
    y_pred_inv = _inverse_first_feature(scaler, y_pred)

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
    scaler.n_features_in_ = int(payload.get("n_features_in_", 1))
    if "n_samples_seen_" in payload:
        scaler.n_samples_seen_ = payload["n_samples_seen_"]
    return scaler


def _lstm_meta_path(model_path: Path) -> Path:
    return model_path.parent / "lstm_training_meta.json"


def _load_model_if_compatible(model_path: Path, n_features: int):
    if not model_path.exists():
        return None
    try:
        m = load_model(model_path, compile=False)
    except Exception:
        try:
            m = load_model(model_path, compile=False, safe_mode=False)
        except Exception:
            try:
                m = load_model(
                    model_path,
                    compile=False,
                    safe_mode=False,
                    custom_objects={"mse": "mean_squared_error"},
                )
            except Exception:
                return None
    try:
        if int(m.input_shape[-1]) != n_features:
            del m
            return None
    except Exception:
        del m
        return None
    return m


def recursive_forecast_next_3(
    monthly_demand: pd.DataFrame,
    model_path: Path,
    scaler_path: Path,
    seq_len: int = 12,
    horizon: int = 3,
    stock_receipts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    monthly_demand = monthly_demand.copy()
    monthly_demand["month"] = pd.to_datetime(monthly_demand["month"])
    monthly_demand = monthly_demand.sort_values(["drug_id", "month"])

    stock = stock_receipts if stock_receipts is not None else pd.DataFrame()

    try:
        model = load_model(model_path, compile=False)
    except Exception:
        try:
            model = load_model(model_path, compile=False, safe_mode=False)
        except Exception:
            model = load_model(
                model_path,
                compile=False,
                safe_mode=False,
                custom_objects={"mse": "mean_squared_error"},
            )
    scaler = _load_scaler(scaler_path)
    n_features_model = int(model.input_shape[-1])

    rows = []
    for did in monthly_demand["drug_id"].unique():
        sub = monthly_demand[monthly_demand["drug_id"] == did].sort_values("month")
        values = sub["monthly_demand"].astype(float).values
        if len(values) < seq_len:
            continue

        last_month = sub["month"].max()

        if n_features_model == 1:
            # Legacy univariate checkpoint
            window = scaler.transform(values.reshape(-1, 1)).flatten()[-seq_len:].tolist()
            for step in range(1, horizon + 1):
                x_input = np.array(window[-seq_len:], dtype=float).reshape(1, seq_len, 1)
                pred_scaled = float(model.predict(x_input, verbose=0)[0][0])
                window.append(pred_scaled)
                pred = float(
                    scaler.inverse_transform(np.array([[pred_scaled]])).flatten()[0]
                )
                rows.append(
                    {
                        "drug_id": did,
                        "forecast_month": (last_month + pd.DateOffset(months=step)).strftime("%Y-%m-%d"),
                        "predicted_demand": round(max(0.0, pred), 2),
                    }
                )
            continue

        # Multivariate: ensure feature columns
        if not all(c in sub.columns for c in LSTM_FEATURE_COLUMNS):
            sub = enrich_monthly_with_lstm_features(sub, stock)
        feat = sub[LSTM_FEATURE_COLUMNS].astype(float).values
        vals_scaled = scaler.transform(feat)
        window = vals_scaled[-seq_len:].copy()

        for step in range(1, horizon + 1):
            x_input = window.reshape(1, seq_len, n_features_model)
            pred_scaled = float(model.predict(x_input, verbose=0)[0][0])
            fc_month = last_month + pd.DateOffset(months=step)
            msin, mcos = month_sin_cos_for_timestamp(fc_month)
            mte = months_to_nearest_expiry_at(stock, str(did), fc_month)
            demand_inv = float(_inverse_first_feature(scaler, np.array([pred_scaled]))[0])
            raw_row = np.array([[demand_inv, msin, mcos, mte]])
            row_scaled = scaler.transform(raw_row)[0]
            window = np.vstack([window[1:], row_scaled.reshape(1, -1)])
            pred = float(demand_inv)
            rows.append(
                {
                    "drug_id": did,
                    "forecast_month": fc_month.strftime("%Y-%m-%d"),
                    "predicted_demand": round(max(0.0, pred), 2),
                }
            )

    return pd.DataFrame(rows).sort_values(["drug_id", "forecast_month"]).reset_index(drop=True)
