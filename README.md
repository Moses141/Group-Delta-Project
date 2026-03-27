# Pharmacy Forecasting (LSTM)

Deep learning-based pharmacy demand forecasting and decision support system.

## Overview

This project is designed around an LSTM time-series model from the beginning.
It supports a practical pharmacy workflow:

1. Upload latest pharmacy records (sales, stock receipts, opening stock)
2. Validate and process data
3. Regenerate monthly demand, stock status, and 3-month forecasts
4. Review recommendations in the dashboard

## Why LSTM

LSTM is used because it models sequential dependencies in monthly demand data,
allowing the system to learn temporal patterns and forecast future demand.

## Project structure

- `app/dashboard.py` – Streamlit interface (Overview, Procurement Planner, Drug Detail, Upload & Refresh Data)
- `app/processing.py` – data processing and refresh pipeline
- `app/forecasting.py` – LSTM sequence preparation, training, recursive forecasting
- `app/validation.py` – upload schema/date/numeric validation
- `app/paths.py` – shared directory paths
- `app/utils.py` – utility helpers
- `data/` – raw CSVs and uploads folder
- `outputs/` – processed outputs for dashboard
- `models/` – trained LSTM model artifacts
- `notebooks/` – development notebooks aligned with LSTM workflow

## Outputs generated

- `outputs/monthly_demand.csv`
- `outputs/stock_status.csv`
- `outputs/lstm_sequences.npy`
- `outputs/next_3_month_forecast.csv`
- `outputs/evaluation_metrics.csv`
- `models/lstm_model.h5`

## Run dashboard

### Option A: Batch script (Windows)

```bat
run_dashboard.bat
```

### Option B: Manual

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## Training and forecasting flow

The Upload & Refresh page triggers the full pipeline:

- validate uploaded files
- replace raw authoritative files in `data/`
- process monthly demand and stock status
- train the global LSTM model
- generate recursive 3-month forecasts per drug
- update dashboard outputs

No notebook interaction is required for normal pharmacy use.
