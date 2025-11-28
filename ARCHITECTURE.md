# 🏗️ MedSupply AI - System Architecture Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [ML Pipeline Architecture](#ml-pipeline-architecture)
5. [Model Architecture Details](#model-architecture-details)
6. [Dashboard Architecture](#dashboard-architecture)
7. [Deployment Architecture](#deployment-architecture)
8. [API Design](#api-design)

---

## Overview

MedSupply AI follows a **modular, pipeline-based architecture** that processes pharmaceutical supply chain data through multiple stages: data ingestion, preprocessing, feature engineering, model training, prediction, and visualization.

The system is designed with **separation of concerns**, allowing each component to be developed, tested, and deployed independently.

---

## System Components

### 1. Data Ingestion Layer

**Purpose**: Collect and standardize input data from multiple sources

**Input Sources**:
- CSV/Excel files (facility stock records)
- Historical consumption data
- HMIS-style aggregated reports
- Synthetic data generation (for testing)

**Key Functions**:
```python
# Data loading
df = pd.read_csv('uganda_drug_supply_synthetic.csv')

# Data validation
- Column existence check
- Data type validation
- Missing value detection
```

**Output**: Raw pandas DataFrame (15,000+ records, 32 features)

---

### 2. Data Preprocessing Module

**Location**: `inventory_management.ipynb` - Cell 7

**Functions**:
- `clean_data(df)`: Handles missing values, date parsing, temporal extraction

**Process**:
1. **Date Conversion**: Convert `stock_received_date` to datetime
2. **Temporal Features**: Extract year, month, week
3. **Missing Value Imputation**: 
   - Numeric: Median imputation
   - Categorical: Mode imputation
4. **Data Type Standardization**: Ensure consistent types

**Output**: Cleaned DataFrame ready for feature engineering

---

### 3. Feature Engineering Module

**Location**: `inventory_management.ipynb` - Cell 8

**Functions**:
- `engineer_features(df)`: Creates 9 new features from raw data

**Engineered Features**:

| Feature | Description | Formula |
|---------|-------------|---------|
| `days_since_receipt` | Age of stock in days | `(now - stock_received_date).days` |
| `composite_risk_score` | Weighted risk indicator | `0.4*expiry_rate + 0.3*stockout_prob + 0.2*(1-staff_avail) + 0.1*(1-data_quality)` |
| `inventory_turnover` | Demand efficiency metric | `average_monthly_demand / (initial_stock + 1)` |
| `service_level_estimate` | Service quality proxy | `1 - predicted_stockout_probability` |
| `has_high_quality_data` | Data quality flag | Binary (1 if High, else 0) |
| `has_good_storage` | Storage condition flag | Binary (1 if Good, else 0) |
| `year`, `month`, `week` | Temporal features | Extracted from date |

**Output**: Enhanced DataFrame with 41 features (32 original + 9 engineered)

---

### 4. Model Preparation Module

**Location**: `inventory_management.ipynb` - Cell 9

**Functions**:
- `prepare_modeling_data(df)`: Selects features and creates target variables

**Feature Selection** (19 features):
- Inventory & Demand: `initial_stock_units`, `reorder_level`, `average_monthly_demand`, `delivery_frequency_days`, `lead_time_days`
- Operational: `supplier_reliability_score`, `staff_availability_index`, `FEFO_policy_implemented`, `warehouse_capacity_utilization`
- External: `region_disease_outbreaks`, `transport_accessibility_score`, `power_stability_index`
- Temporal: `month`, `week`
- Engineered: `composite_risk_score`, `inventory_turnover`, `service_level_estimate`, `has_high_quality_data`, `has_good_storage`

**Target Variables**:
- `y_demand`: `average_monthly_demand` (regression)
- `y_stockout`: `stockout_occurred` (binary classification)
- `y_expiry`: `expiry_rate_percent` (regression)

**Data Splitting**:
- Train: 80% (12,000 samples)
- Test: 20% (3,000 samples)
- Random state: 42 (reproducibility)

---

## ML Pipeline Architecture

### Model Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL INITIALIZATION                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   SARIMA     │  │   Random     │  │   XGBoost    │     │
│  │  (Time       │  │   Forest     │  │  (Gradient   │     │
│  │  Series)     │  │  (Ensemble)  │  │   Boosting)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐                                          │
│  │     LSTM     │                                          │
│  │   (Deep      │                                          │
│  │   Learning)  │                                          │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                           │
│  • Hyperparameter tuning (GridSearchCV)                     │
│  • Cross-validation (TimeSeriesSplit)                       │
│  • Early stopping (LSTM)                                     │
│  • Model persistence                                         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION                         │
│  Metrics:                                                    │
│  • MAE (Mean Absolute Error)                                 │
│  • RMSE (Root Mean Squared Error)                           │
│  • MAPE (Mean Absolute Percentage Error)                     │
│  • MPE (Mean Percentage Error) for OOQ                      │
│  • Accuracy, Precision, Recall, F1-Score (Classification)   │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL SELECTION                          │
│  Best Model: Random Forest (Demand)                         │
│  Best Model: XGBoost Optimized (Stockout)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Architecture Details

### 1. SARIMA Model

**Type**: Seasonal AutoRegressive Integrated Moving Average  
**Location**: `inventory_management.ipynb` - Cell 21

**Architecture**:
- **Order**: (1, 1, 1) - AR(1), I(1), MA(1)
- **Seasonal Order**: (1, 1, 1, 12) - Monthly seasonality
- **Method**: Maximum Likelihood Estimation

**Use Case**: Time-series forecasting with seasonal patterns  
**Strengths**: Works well with smaller datasets, easier to implement  
**Limitations**: Requires stationary data, linear relationships

**Code Structure**:
```python
model = SARIMAX(train_ts, 
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
fitted_model = model.fit(disp=False, maxiter=50)
forecast = fitted_model.forecast(steps=len(test_ts))
```

---

### 2. Random Forest Model

**Type**: Ensemble Learning (Bagging)  
**Location**: `inventory_management.ipynb` - Cell 18

**Architecture**:
- **n_estimators**: 200 trees
- **max_depth**: 10
- **random_state**: 42
- **n_jobs**: -1 (parallel processing)

**Performance**: 
- MAPE: 0.06% (Best performing)
- MAE: 1.27
- RMSE: 4.45

**Use Case**: Demand forecasting (regression)  
**Strengths**: High accuracy, feature importance, handles non-linearity  
**Limitations**: Less interpretable than linear models

---

### 3. XGBoost Model

**Type**: Gradient Boosting  
**Location**: `inventory_management.ipynb` - Cell 18

**Architecture**:
- **n_estimators**: 200
- **max_depth**: 6
- **learning_rate**: 0.1
- **subsample**: 0.8
- **colsample_bytree**: 0.8

**Performance**:
- MAPE: 1.02%
- MAE: 22.03
- RMSE: 55.77

**Use Case**: Demand forecasting and stockout prediction  
**Strengths**: Handles non-linear patterns, feature importance  
**Limitations**: Requires more technical expertise

---

### 4. LSTM Model

**Type**: Deep Learning (Recurrent Neural Network)  
**Location**: `inventory_management.ipynb` - Cell 22

**Architecture**:
```
Input Layer: (batch_size, 1, 19 features)
    ↓
LSTM Layer 1: 50 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 2: 50 units, return_sequences=False
    ↓
Dropout: 0.2
    ↓
Dense Layer: 25 units, ReLU
    ↓
Output Layer: 1 unit (demand forecast)
```

**Hyperparameters**:
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss**: MSE (Mean Squared Error)
- **Epochs**: 30 (with early stopping)
- **Batch Size**: 32
- **Validation Split**: 0.2

**Use Case**: Complex temporal pattern recognition  
**Strengths**: Can capture long-term dependencies  
**Limitations**: Requires large datasets, more computing power

---

### 5. Optimal Order Quantity (OOQ) Calculator

**Location**: `inventory_management.ipynb` - Cell 16

**Formula**:
```
OOQ = max(0, forecast_weekly_demand * lead_time_weeks + safety_stock - current_balance)

Where:
safety_stock = z * sigma_demand * sqrt(lead_time_weeks)
z = 1.65 (for 95% service level)
lead_time_weeks = lead_time_days / 7
```

**Implementation**:
```python
def calculate_ooq(demand_forecast, lead_time_days, current_stock, 
                  reorder_level, service_level=0.95):
    lead_time_weeks = lead_time_days / 7
    z_score = 1.65  # 95% service level
    demand_std = demand_forecast * 0.2  # 20% volatility estimate
    safety_stock = z_score * demand_std * np.sqrt(lead_time_weeks)
    ooq = max(0, (demand_forecast * lead_time_weeks) + safety_stock - current_stock)
    return ooq, safety_stock
```

**Performance**: MPE = 0.07% (Target: ≤7.3%) ✅

---

## Dashboard Architecture

### Streamlit Application Structure

**File**: `medsupply_dashboard.py`

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│              STREAMLIT APPLICATION                       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              PAGE ROUTER                         │   │
│  │  (Sidebar Navigation)                            │   │
│  └──────────────────────────────────────────────────┘   │
│                        ↓                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Dashboard   │  │   Facility   │  │    Model     │ │
│  │   Overview   │  │   Analysis   │  │ Performance  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐                                      │
│  │   OOQ        │                                      │
│  │  Calculator  │                                     │
│  └──────────────┘                                      │
└─────────────────────────────────────────────────────────┘
```

### Page Components

#### 1. Dashboard Page
- **Key Metrics**: 4-column metric display
- **Success Box**: Target achievement indicator
- **Facility Predictions Table**: Real-time predictions
- **Charts**: Bar chart (Stock vs Demand), Pie chart (Risk Distribution)

#### 2. Facility Analysis Page
- **Facility Selector**: Dropdown menu
- **Inventory Status**: Current stock, consumption, stockout probability
- **Recommendations**: OOQ suggestions, warnings, FEFO tips

#### 3. Model Performance Page
- **Model Comparison**: MAPE, MAE, RMSE charts
- **Radar Chart**: Multi-metric comparison
- **Characteristics Chart**: Ease of implementation, data requirements, computing power
- **Summary Table**: Performance metrics with rankings

#### 4. OOQ Calculator Page
- **Input Fields**: Current stock, weekly demand, lead time
- **Sliders**: Service level, demand volatility
- **Output**: Recommended OOQ, safety stock
- **Visualization**: Indicator chart

---

## Deployment Architecture

### Current Deployment (Development)

```
┌─────────────────────────────────────────────────────────┐
│              LOCAL DEVELOPMENT ENVIRONMENT               │
│                                                          │
│  ┌──────────────┐         ┌──────────────┐             │
│  │   Jupyter    │         │  Streamlit   │             │
│  │   Notebook   │         │  Dashboard   │             │
│  │  (Training)  │         │  (Port 8501) │             │
│  └──────────────┘         └──────────────┘             │
│         ↓                        ↓                       │
│  ┌──────────────────────────────────────┐               │
│  │      Local File System               │               │
│  │  (CSV files, models, visualizations) │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

### Production Deployment (Recommended)

```
┌─────────────────────────────────────────────────────────┐
│              PRODUCTION ARCHITECTURE                     │
│                                                          │
│  ┌──────────────┐         ┌──────────────┐             │
│  │   Data       │         │   FastAPI    │             │
│  │   Pipeline   │────────▶│   Backend    │             │
│  │  (ETL)       │         │  (Port 8000) │             │
│  └──────────────┘         └──────────────┘             │
│                                  ↓                       │
│  ┌──────────────┐         ┌──────────────┐             │
│  │   ML Model   │         │  Streamlit   │             │
│  │   Service    │         │  Dashboard   │             │
│  │  (MLflow)    │         │  (Port 8501) │             │
│  └──────────────┘         └──────────────┘             │
│                                  ↓                       │
│  ┌──────────────────────────────────────┐               │
│  │      Cloud Storage / Database        │               │
│  │  (PostgreSQL / S3 / Azure Blob)      │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

### Docker Deployment (Future)

```dockerfile
# Dockerfile structure
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Expose ports
EXPOSE 8501 8000

# Run Streamlit
CMD ["streamlit", "run", "medsupply_dashboard.py", "--server.port=8501"]
```

---

## API Design

### MedicineSupplyForecaster Class

**Location**: `inventory_management.ipynb` - Cell 17

**Class Structure**:
```python
class MedicineSupplyForecaster:
    def __init__(self, demand_model, stockout_model, feature_names):
        self.demand_model = demand_model
        self.stockout_model = stockout_model
        self.feature_names = feature_names
        self.scaler = StandardScaler()
    
    def predict(self, facility_data):
        """
        Returns:
        {
            'demand_forecast': float,
            'stockout_probability': float,
            'optimal_order_quantity': float,
            'safety_stock': float,
            'reorder_recommended': bool
        }
        """
```

### Future FastAPI Endpoints (Recommended)

```python
# /predict_ooq
POST /api/v1/predict/ooq
Request: {
    "facility_id": "string",
    "drug_id": "string",
    "current_stock": float,
    "lead_time_days": int
}
Response: {
    "optimal_order_quantity": float,
    "safety_stock": float,
    "confidence": float
}

# /stock_alerts
GET /api/v1/alerts/stock
Response: [
    {
        "facility_id": "string",
        "drug_id": "string",
        "risk_level": "high|medium|low",
        "days_until_stockout": int
    }
]

# /feature_importance
GET /api/v1/models/feature-importance
Response: {
    "model": "random_forest",
    "features": [
        {"name": "average_monthly_demand", "importance": 0.35},
        ...
    ]
}
```

---

## Data Flow Diagram

```
┌─────────────┐
│  Raw Data   │ (CSV/Excel)
│  (15K rows) │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Data Cleaning   │ → Missing values, date parsing
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Feature         │ → 9 new features created
│ Engineering     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Train/Test      │ → 80/20 split
│ Split           │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Model Training  │ → SARIMA, RF, XGBoost, LSTM
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Model Selection │ → Best model chosen
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Predictions     │ → OOQ, Stockout Risk, Demand
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Dashboard       │ → Streamlit visualization
└─────────────────┘
```

---

## Performance Metrics

### Model Evaluation Metrics

**Regression Metrics** (Demand Forecasting):
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error  
- **MAPE**: Mean Absolute Percentage Error
- **MPE**: Mean Percentage Error (for OOQ)

**Classification Metrics** (Stockout Prediction):
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Target Performance

- **OOQ MPE**: ≤ 7.3% ✅ (Achieved: 0.07%)
- **Demand MAPE**: Minimize (Achieved: 0.06%)
- **Stockout F1**: Maximize (Achieved: 0.349)

---

## Security Considerations

1. **Data Privacy**: Synthetic data used for development
2. **Model Validation**: Cross-validation prevents overfitting
3. **Input Validation**: Data type and range checks
4. **Error Handling**: Try-except blocks for robustness

---

## Scalability Considerations

1. **Data Volume**: Handles 15,000+ records efficiently
2. **Model Training**: Parallel processing (n_jobs=-1)
3. **Dashboard**: Streamlit handles real-time updates
4. **Future**: Can scale to cloud infrastructure

---

**Last Updated**: February 2026  
**Version**: 1.0.0

