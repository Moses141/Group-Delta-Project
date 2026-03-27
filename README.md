# Pharmacy Demand Forecasting System (LSTM)

A machine learning-based pharmacy demand forecasting and decision support system using an LSTM model.

---

## 1) Project Overview

This system helps a pharmacy answer practical questions such as:

- How much medicine will be needed in the next 3 months?
- Which drugs are at risk of stockout?
- Which drugs should be reordered soon?

### Real-world problem it solves

Pharmacies often face:

- **Stockouts** (patients cannot get needed medicines)
- **Overstock and expiry** (financial loss from expired products)
- **Poor planning** (late procurement and emergency purchases)

This project helps reduce those problems by combining:

- historical dispensing and stock data
- demand forecasting
- stock status and reorder support
- a user-friendly dashboard

---

## 2) How the System Works (High-Level Flow)

The workflow is simple:

1. User uploads pharmacy data (sales, stock receipts, opening stock)
2. System validates file structure and values
3. Data is cleaned and converted into monthly demand
4. LSTM model learns demand patterns from history
5. System predicts next 3 months demand per drug
6. Dashboard shows forecasts, stock status, and reorder recommendations

### Simple flow diagram

```text
Upload Data
    ->
Validation
    ->
Data Processing
    ->
LSTM Training + Forecasting
    ->
Dashboard Insights (Demand, Stock, Reorder)
```

---

## 3) System Architecture

The system is organized into five layers:

### Data Layer

- Raw pharmacy files:
  - sales transactions
  - stock receipts
  - opening stock

### Processing Layer

- Cleans and standardizes data
- Aggregates daily transactions into monthly demand
- Computes stock balance:
  - opening stock + received - dispensed

### Machine Learning Layer

- Builds LSTM sequences from monthly demand
- Trains one global LSTM model
- Generates 3-month forecasts per drug

### Application Layer

- Streamlit dashboard for users
- Upload & Refresh page for regular updates
- Decision support pages (overview, procurement, drug detail)

### Output Layer

- Saves processed files used by the dashboard
- Saves model artifacts for reuse

---

## 4) Project Structure

```text
pharmacy_forecasting/
|
|- app/
|  |- dashboard.py
|  |- processing.py
|  |- forecasting.py
|  |- validation.py
|  |- paths.py
|  |- utils.py
|
|- data/
|  |- uploads/
|  |- sales_transactions.csv
|  |- stock_receipts.csv
|  |- opening_stock.csv
|
|- outputs/
|  |- monthly_demand.csv
|  |- stock_status.csv
|  |- lstm_sequences.npy
|  |- next_3_month_forecast.csv
|  |- evaluation_metrics.csv
|
|- models/
|  |- lstm_model.h5
|
|- notebooks/
|  |- 01_data_preparation.ipynb
|  |- 02_exploratory_analysis.ipynb
|  |- 03_sequence_preparation.ipynb
|  |- 04_lstm_training.ipynb
|  |- 05_forecasting_and_evaluation.ipynb
|
|- requirements.txt
|- run_dashboard.bat
|- README.md
```

### What each key file does

#### `app/`

- `dashboard.py`  
  Main user interface (Overview, Procurement Planner, Drug Detail, Upload & Refresh Data)

- `processing.py`  
  Data cleaning, monthly aggregation, stock status generation, refresh pipeline

- `forecasting.py`  
  LSTM sequence preparation, model training, and recursive forecasting

- `validation.py`  
  Checks uploaded files (required columns, date parsing, numeric checks)

- `paths.py`  
  Central path and directory management

- `utils.py`  
  Shared helper functions

#### `data/`

- Stores current authoritative pharmacy input files
- `uploads/` can store uploaded files or temporary artifacts

#### `outputs/`

- Stores processed datasets and forecast outputs used by the dashboard

#### `models/`

- Stores trained LSTM model artifacts

#### `notebooks/`

- `01_data_preparation.ipynb`: data cleaning + monthly demand + stock status  
- `02_exploratory_analysis.ipynb`: trend/seasonality visualization  
- `03_sequence_preparation.ipynb`: LSTM sequence generation  
- `04_lstm_training.ipynb`: model training  
- `05_forecasting_and_evaluation.ipynb`: forecasting and evaluation

---

## 5) Machine Learning Model (LSTM)

### What is LSTM?

LSTM (Long Short-Term Memory) is a neural network designed for sequential data.

### Why use LSTM here?

Pharmacy demand is time-based. What happened in previous months affects future demand.
LSTM is useful because it can learn these time relationships.

### In simple terms

- Input: past monthly demand values
- Learning: pattern of how demand changes over time
- Output: expected future demand

---

## 6) Forecasting Process

The model uses sequence-based forecasting:

- Sequence length: **12 months**
- Prediction target: **next month**

### Sequence example

```text
Months 1-12  -> predict Month 13
Months 2-13  -> predict Month 14
...
```

### 3-month forecasting (recursive)

1. Use latest 12 months to predict month +1
2. Append prediction to sequence
3. Predict month +2
4. Repeat for month +3

This produces a 3-month forecast per drug.

---

## 7) Dashboard Features

The dashboard provides:

- Demand trends and forecast visualization
- Stock status and stockout risk indicators
- Reorder suggestions
- Days of supply analysis
- Drug-level detail view
- Upload & Refresh workflow for ongoing updates

Pages:

- **Overview**
- **Procurement Planner**
- **Drug Detail**
- **Upload & Refresh Data**

---

## 8) How to Use the System

### Step-by-step

1. Open the dashboard
2. Go to **Upload & Refresh Data**
3. Upload latest full:
   - sales transactions
   - stock receipts
   - opening stock (optional if unchanged after setup)
4. Click process/refresh
5. Review updated results in:
   - Overview
   - Procurement Planner
   - Drug Detail

---

## 9) Installation & Setup

### Option A (Windows launcher)

```bat
run_dashboard.bat
```

### Option B (manual)

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## 10) Outputs

### `outputs/monthly_demand.csv`

- Monthly total demand per drug
- Used for model training and trend charts

### `outputs/stock_status.csv`

- Stock balance per drug:
  - opening stock
  - total received
  - total dispensed
  - current stock

### `outputs/next_3_month_forecast.csv`

- Predicted demand for next 3 months per drug
- Used by forecast charts and reorder logic

### Additional outputs

- `outputs/lstm_sequences.npy`: prepared model sequences
- `outputs/evaluation_metrics.csv`: model performance summary
- `models/lstm_model.h5`: trained LSTM model

---

## 11) Limitations

- Forecast quality depends heavily on input data quality
- Assumes future behavior is similar to historical patterns
- Not a real-time streaming system (manual refresh workflow)

---

## 12) Future Improvements

- Real-time data integration (e.g., from pharmacy information systems)
- Automated alerts for stockout and expiry risk
- Multi-branch pharmacy support
- Role-based access for pharmacists, managers, and procurement teams

---

## Final Note

This project is designed as a practical, explainable, and modular LSTM-based pharmacy forecasting solution.  
It combines machine learning with operational pharmacy decision support in a format suitable for academic and portfolio presentation.

---

## 13) LSTM Model Details

This section explains the model design in more depth, using clear language and practical reasoning.

### 13.1 Model Architecture

The model is intentionally simple and focused:

- **Input:** 12 months of demand history
- **Features per time step:** 1 (monthly demand value)
- **Core layer:** one LSTM layer
- **Output layer:** one Dense neuron for the next-month prediction

Typical architecture used in this project:

```text
Input (12, 1) -> LSTM(50 to 64 units) -> Dense(1)
```

Why this design:

- One LSTM layer is enough to learn short- and medium-range demand patterns
- Dense(1) gives a direct numeric forecast for the next month
- This keeps training stable and explainable for a practical pharmacy system

### 13.2 Input Representation

The data is converted into supervised sequences.

- Sequence length = **12**
- Target = **next month demand**

Example:

- Months Jan-Dec -> predict Jan (next year)
- Months Feb-Jan -> predict Feb (next year)

Tensor shape:

- **X:** `(samples, timesteps, features)` = `(samples, 12, 1)`
- **y:** `(samples,)`

This format allows the model to learn temporal relationships from fixed-size history windows.

### 13.3 Training Configuration

Typical training setup:

- **Epochs:** 30-50
- **Batch size:** 16-32
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam

Why these choices:

- **MSE** is standard for continuous-value prediction and penalizes large errors more strongly
- **Adam** converges quickly and works well with noisy real-world demand data
- **30-50 epochs** is usually enough to learn patterns without excessive training time
- **Batch size 16-32** balances gradient stability and computation speed

### 13.4 Scaling / Normalization

The model uses **MinMaxScaler** before training.

Why scaling is important:

- Neural networks train better when inputs are on a similar numeric range
- Unscaled demand values can slow learning or make optimization unstable

After prediction:

- Outputs are inverse-transformed back to original demand units
- This ensures dashboard forecasts are practical and interpretable

### 13.5 Forecasting Strategy

The system uses **recursive multi-step forecasting**:

1. Predict next month using the latest 12 months
2. Add that prediction to the input window
3. Predict the following month
4. Repeat until 3 future months are produced

This mirrors real deployment conditions, where only past data and generated predictions are available for future steps.

### 13.6 Evaluation

The model is evaluated using:

- **MAE** (Mean Absolute Error): average absolute prediction error
- **RMSE** (Root Mean Squared Error): emphasizes larger errors
- **MAPE** (Mean Absolute Percentage Error): error as a percentage

Why MAPE is important in business settings:

- It is easy for non-technical stakeholders to interpret
- Example: a 10% error is easier to discuss operationally than raw unit error alone

### 13.7 Model Behavior and Interpretation

LSTM is suitable here because it learns temporal dependencies in demand history.

In practice, it can capture:

- **Trend:** gradual growth or decline in demand
- **Seasonality:** repeated monthly/annual patterns
- **Recent changes:** short-term shifts caused by recent consumption behavior

So the model is not just fitting averages; it is learning sequence dynamics over time.

### 13.8 Hyperparameters Summary

| Parameter | Value Used |
|---|---|
| Sequence length | 12 |
| Input shape | (12, 1) |
| LSTM units | 50-64 |
| Epochs | 30-50 |
| Batch size | 16-32 |
| Optimizer | Adam |
| Loss | MSE |

### 13.9 Design Decisions

Key decisions and rationale:

- **Why LSTM:** demand is sequential; month-to-month dependencies matter
- **Why one global model for all drugs:** simpler maintenance, shared pattern learning, and practical deployment
- **Why sequence length = 12:** captures one full year cycle, which is meaningful for seasonality in pharmacy demand

These choices prioritize a balance of performance, simplicity, and operational usability.

### 13.10 Limitations of the LSTM Approach

Important practical limitations:

- Sensitive to data scaling and preprocessing quality
- Requires enough historical data to learn stable patterns
- Can underfit if configuration is too small or training is insufficient
- Forecast quality may degrade when data behavior changes suddenly

These limitations are manageable with regular data refresh, monitoring, and periodic retraining.
