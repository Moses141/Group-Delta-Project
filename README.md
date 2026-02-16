<<<<<<< HEAD
# Group Delta
- Puoch Mabor Makuei, S23B23/055
- Kiisa Angela, S23B23/027
- Nkangi Moses, M23B23/027

# Pharmacy Demand & Stock Monitoring Dashboard

A comprehensive Streamlit dashboard for predicting medicine demand using SARIMA models and monitoring stock levels in real-time.

## Features

- 📈 **Demand Forecasting**: Uses SARIMA time series models to predict future medication demand
- 📦 **Stock Monitoring**: Real-time stock level tracking with alerts for low stock
- 📊 **Historical Analysis**: View trends and patterns in medication demand over time
- 🔍 **Detailed Insights**: Comprehensive medication information and operational metrics
- ⚠️ **Smart Alerts**: Automatic notifications for critical stock levels

## Installation

1. Install required packages:
```bash
pip install -r requirements_dashboard.txt
```

2. Ensure you have the data file `uganda_drug_supply_synthetic.csv` in the same directory as the dashboard.

## Usage

Run the dashboard using Streamlit:

```bash
streamlit run pharmacy_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Dashboard Sections

### 1. Demand Forecast Tab
- Visual forecast chart with confidence intervals
- Forecasted demand for the next N weeks
- Historical vs predicted demand comparison

### 2. Stock Analysis Tab
- Stock level projections over time
- Reorder point and safety stock calculations
- Stock status indicators (Adequate, Low, Critical)
- Order recommendations

### 3. Historical Trends Tab
- Time series visualization of historical demand
- Monthly average demand patterns
- Statistical summary of demand data

### 4. Detailed View Tab
- Complete medication information
- Operational metrics (lead time, reorder levels, etc.)
- Raw data preview

## Configuration

### Sidebar Controls

- **Select Medication**: Choose which medication to analyze
- **Forecast Weeks**: Number of weeks to forecast (1-12)
- **Current Stock**: Enter current inventory level
- **Lead Time**: Days required for new stock delivery
- **Safety Stock %**: Percentage buffer for safety stock

## Stock Status Indicators

- 🟢 **Adequate**: Stock levels are healthy
- 🟡 **Reorder Soon**: Consider reordering in near future
- 🟠 **Low**: Action required - reorder within 2-3 days
- 🔴 **Critical**: Urgent - reorder immediately

## Model Details

The dashboard uses SARIMA (Seasonal AutoRegressive Integrated Moving Average) models:
- **Order**: (1,1,1) - ARIMA parameters
- **Seasonal Order**: (1,0,1,52) for weekly data with yearly seasonality
- **Training**: Uses 80% of historical data for training
- **Forecasting**: Generates predictions with 95% confidence intervals

## Data Requirements

The dashboard expects a CSV file with the following columns:
- `drug_name`: Medication name
- `stock_received_date`: Date of stock receipt
- `average_monthly_demand`: Target variable for forecasting
- Additional columns for medication details (optional)

## Notes

- The model automatically applies log transformation if all demand values are positive
- Forecasts are generated weekly
- Stock calculations assume constant demand rate
- Model retrains automatically when medication is changed

## Troubleshooting

**Issue**: "Data file not found"
- Solution: Ensure `uganda_drug_supply_synthetic.csv` is in the same directory

**Issue**: "Insufficient data"
- Solution: Need at least 20 data points for reliable forecasting

**Issue**: Model training fails
- Solution: Check that statsmodels is properly installed and data is valid

## Future Enhancements

- [ ] Multi-medication comparison
- [ ] Export forecasts to CSV/Excel
- [ ] Email/SMS alerts for critical stock
- [ ] Integration with inventory management systems
- [ ] Advanced model selection (auto ARIMA)
- [ ] Batch forecasting for all medications
=======
# 🏥 MedSupply AI - Uganda

### Machine Learning–Based Forecasting Model for Reducing Medicine Expiry and Stock-Outs in Ugandan Public Health Facilities

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

**Group Delta Project — Uganda Christian University**

**Team Members:**
- **Puoch Mabor Makuei** — S23B23/055
- **Kiisa Angela** — S23B23/027  
- **Nkangi Moses** — M23B23/024 (Project Lead)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

MedSupply AI is an intelligent pharmaceutical supply chain management system designed to address critical challenges in Uganda's public health facilities: **medicine stockouts**, **overstocking**, and **drug expiries**. 

The system leverages **machine learning forecasting**, **predictive analytics**, and **interactive dashboards** to support data-driven decision-making in medicine procurement, distribution, and inventory management.

### Key Achievements

✅ **MPE ≤ 7.3% Target Achieved**: Optimal Order Quantity (OOQ) predictions achieve **0.07% Mean Percentage Error** (104x better than target)  
✅ **99.94% Demand Forecast Accuracy**: Random Forest model achieves near-perfect demand prediction  
✅ **Multi-Model Comparison**: Comprehensive evaluation of SARIMA, Random Forest, XGBoost, and LSTM models  
✅ **Production-Ready Dashboard**: Interactive Streamlit interface for real-time decision support

---

## 🔍 Problem Statement

Uganda's public health facilities face a critical paradox: **simultaneous stockouts and high medicine expiries**, leading to:

- **Treatment interruptions** due to stockouts
- **Financial losses** (e.g., ~US$550,000 in expired ARVs documented)
- **Poor patient outcomes** from medicine unavailability
- **Inefficient resource allocation** across facilities

**Root Causes:**
- Reliance on simple consumption averages and manual requisitions
- Lack of non-linear pattern capture (seasonality, outbreaks, policy changes)
- No operational forecasting system tailored to Uganda's facility constraints
- Fragmented data visibility across the supply chain

**Our Solution:** A validated ML forecasting system that delivers weekly Optimal Order Quantity (OOQ) predictions with **MPE ≤ 7.3%** and integrates FEFO-aware alerts and redistribution recommendations.

---

## ✨ Key Features

### 1. **Multi-Model Forecasting Engine**
- **SARIMA**: Time-series forecasting with seasonal patterns
- **Random Forest**: High-accuracy demand prediction (MAPE: 0.06%)
- **XGBoost**: Gradient boosting for non-linear patterns
- **LSTM**: Deep learning for complex temporal sequences

### 2. **Optimal Order Quantity (OOQ) Calculator**
- Automatic calculation based on demand forecast, lead time, and safety stock
- Service level optimization (default: 95%)
- Real-time reorder recommendations

### 3. **Stockout Prediction**
- Binary classification model (XGBoost Classifier)
- Probability-based risk assessment
- Early warning alerts

### 4. **Interactive Dashboard**
- **Real-time facility predictions**
- **Model performance metrics**
- **OOQ calculator tool**
- **Facility-level analysis**
- **Visual comparison charts**

### 5. **Feature Engineering**
- Temporal features (year, month, week, days since receipt)
- Composite risk scores
- Inventory turnover metrics
- Service level estimates

### 6. **Comprehensive Model Comparison**
- 6-panel visualization dashboard
- Multi-metric performance analysis
- Model characteristics comparison
- Automated insights generation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   CSV/Excel  │  │  Stock Cards │  │  HMIS Data   │           │
│  │   Files      │  │  (Bin Cards) │  │  Reports     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              DATA PREPROCESSING & FEATURE ENGINEERING           │
│  • Missing value imputation (median/mode)                       │
│  • Date parsing and temporal feature extraction                 │
│  • Composite risk score calculation                             │
│  • Inventory turnover metrics                                   │
│  • Categorical encoding                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              MACHINE LEARNING FORECASTING MODULE                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ SARIMA   │  │ Random   │  │ XGBoost  │  │   LSTM   │         │
│  │ (Time    │  │ Forest   │  │ (Gradient│  │  (Deep   │         │
│  │ Series)  │  │ (Ensemble│  │ Boosting)│  │ Learning)│         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│                    Model Selection & Evaluation                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PREDICTIVE ANALYTICS ENGINE                        │
│  • Demand Forecast → OOQ Calculation                            │
│  • Stockout Probability → Risk Alerts                           │
│  • Expiry Risk Assessment → FEFO Recommendations                │
│  • Facility-Level Predictions                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              INTERACTIVE DASHBOARD (Streamlit)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Dashboard   │  │   Facility   │  │    Model     │           │
│  │   Overview   │  │   Analysis   │  │ Performance  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐                                               │
│  │   OOQ        │                                               │
│  │  Calculator  │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Moses141/Group-Delta-Project.git
cd Group-Delta-Project
```

### Step 2: Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate medsupply-ai
```

### Step 3: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit; print('All packages installed successfully!')"
```

---

## 💻 Usage

### Option 1: Run the Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook inventory_management.ipynb

# Execute all cells in order
# The notebook includes:
# - Data loading and preprocessing
# - Model training (SARIMA, RF, XGBoost, LSTM)
# - Model comparison visualizations
# - OOQ calculations
# - Performance evaluation
```

### Option 2: Run Standalone Model Comparison Script

```bash
python run_model_comparison.py
```

This script will:
- Load and preprocess the data
- Train all models (SARIMA, Random Forest, XGBoost, LSTM)
- Generate comprehensive comparison visualizations
- Save results as `model_comparison_comprehensive.png`

### Option 3: Launch the Interactive Dashboard

```bash
streamlit run medsupply_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Dashboard Features:**
- **Dashboard**: Overview with key metrics and facility predictions
- **Facility Analysis**: Detailed facility-level insights
- **Model Performance**: Comprehensive model comparison visualizations
- **OOQ Calculator**: Interactive Optimal Order Quantity calculator

### Option 4: Use the Forecasting API (Programmatic)

```python
from inventory_management import MedicineSupplyForecaster
import pandas as pd

# Load your trained models (from notebook)
forecaster = MedicineSupplyForecaster(
    demand_model=best_demand_model,
    stockout_model=best_stockout_model,
    feature_names=feature_names
)

# Make predictions for a facility
facility_data = pd.Series({
    'initial_stock_units': 1000,
    'average_monthly_demand': 500,
    'lead_time_days': 21,
    # ... other features
})

predictions = forecaster.predict(facility_data)
print(f"Demand Forecast: {predictions['demand_forecast']}")
print(f"Optimal Order Quantity: {predictions['optimal_order_quantity']}")
print(f"Stockout Probability: {predictions['stockout_probability']:.1%}")
```

---

## 📊 Model Performance

### Demand Forecasting Results

| Model | MAE | RMSE | MAPE | Status |
|-------|-----|------|------|--------|
| **Random Forest** | 1.27 | 4.45 | **0.06%** | ✅ Best |
| **XGBoost** | 22.03 | 55.77 | 1.02% | ⚙️ Good |
| **SARIMA** | ~385 | ~580 | ~15% | 📊 Practical |
| **LSTM** | ~1.4 | ~5.0 | ~0.07% | 💻 Advanced |

### Stockout Prediction Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost (Optimized)** | 58.4% | 0.492 | 0.221 | **0.349** |
| **Random Forest** | 61.7% | 0.551 | 0.200 | 0.293 |

### OOQ Prediction Performance

- **Target MPE**: ≤ 7.3%
- **Achieved MPE**: **0.07%** ✅
- **Improvement**: 104x better than target
- **Sample Size**: 2,430 facilities
- **95% Confidence Interval**: ±0.01%

### Model Comparison Insights

- **✅ SARIMA**: Most practical choice — accurate, easier to implement, works with smaller datasets
- **⚙️ XGBoost & Random Forest**: Similar performance to SARIMA but need more technical expertise
- **💻 LSTM**: Potentially more accurate but requires larger datasets and more computing power

---

## 📁 Project Structure

```
Group-Delta-Project/
│
├── README.md                          # This file
├── ARCHITECTURE.md                    # Detailed architecture documentation
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (optional)
│
├── inventory_management.ipynb         # Main Jupyter notebook (data pipeline + ML)
├── medsupply_dashboard.py            # Streamlit interactive dashboard
├── run_model_comparison.py            # Standalone model training script
│
├── uganda_drug_supply_synthetic.csv   # Synthetic dataset (15,000 records)
├── model_comparison_comprehensive.png # Generated visualization
│
└── docs/                              # Additional documentation (if any)
    └── ...
```

### Key Files Description

- **`inventory_management.ipynb`**: Complete ML pipeline including data preprocessing, model training, evaluation, and visualization
- **`medsupply_dashboard.py`**: Production-ready Streamlit dashboard with 4 main pages
- **`run_model_comparison.py`**: Standalone script for training all models and generating comparisons
- **`uganda_drug_supply_synthetic.csv`**: Synthetic dataset representing Ugandan health facility data (32 features, 15,000 records)

---

## 🔬 Data Format

### Input Data Requirements

The system accepts CSV or Excel files with the following columns:

**Required Columns (32 total):**

| Category | Columns |
|---------|---------|
| **Drug Info** | `drug_id`, `drug_name`, `manufacturer_country`, `license_holder` |
| **Facility Info** | `distribution_region`, `facility_type` |
| **Inventory** | `initial_stock_units`, `stock_received_date`, `reorder_level` |
| **Demand** | `average_monthly_demand`, `delivery_frequency_days`, `lead_time_days` |
| **Operational** | `supplier_reliability_score`, `staff_availability_index`, `FEFO_policy_implemented` |
| **External** | `region_disease_outbreaks`, `season`, `transport_accessibility_score`, `power_stability_index` |
| **Storage** | `storage_temperature`, `storage_humidity`, `warehouse_capacity_utilization`, `storage_condition_rating` |
| **Outcomes** | `stockout_occurred` (0/1), `expiry_rate_percent`, `forecast_error_percent`, `financial_loss_due_to_expiry_usd` |
| **Metadata** | `data_record_quality`, `data_source` |

**Data Format:**
- CSV format (UTF-8 encoding)
- Date format: `YYYY-MM-DD` (e.g., `2025-03-06`)
- Missing values: Automatically handled (median imputation for numeric, mode for categorical)

### Adding New Data

To continuously train the model with new data:

```python
import pandas as pd

# Load existing data
existing_data = pd.read_csv('uganda_drug_supply_synthetic.csv')

# Load new data (must have same columns)
new_data = pd.read_csv('new_facility_data.csv')

# Append new data
combined_data = pd.concat([existing_data, new_data], ignore_index=True)

# Save updated dataset
combined_data.to_csv('uganda_drug_supply_synthetic.csv', index=False)

# Retrain models (run notebook cells 3-23)
```

---

## 🎓 Research Context

This project addresses a critical gap identified in the literature:

- **Problem**: Uganda faces high medicine wastage (UGX 316.65 billion expired in FY 2023/2024)
- **Root Cause**: Over 80% of outlets experience expiries due to poor forecasting
- **Gap**: No operational ML forecasting tools designed for Uganda's health supply chain
- **Solution**: This project provides a practical, ML-driven forecasting and decision-support platform

**Alignment with:**
- **SDG 3**: Good Health and Well-being
- **Uganda Vision 2040**: Technology-enabled healthcare
- **CRISP-DM Methodology**: Cross-Industry Standard Process for Data Mining

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Nkangi Moses** (M23B23/024) - Project Lead, ML Development
- **Puoch Mabor Makuei** (S23B23/055) - Data Engineering, Visualization
- **Kiisa Angela** (S23B23/027) - Dashboard Development, Testing

---

## 🙏 Acknowledgments

- Uganda Christian University for academic support
- Ministry of Health, Uganda for domain expertise
- National Medical Stores (NMS) for supply chain insights
- Open-source ML community for tools and libraries

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Repository**: [https://github.com/Moses141/Group-Delta-Project](https://github.com/Moses141/Group-Delta-Project)
- **Issues**: [GitHub Issues](https://github.com/Moses141/Group-Delta-Project/issues)

---

## 📈 Future Work

- [ ] Integration with real HMIS data streams
- [ ] FastAPI backend for production deployment
- [ ] SMS/WhatsApp alert system
- [ ] Mobile app for field workers
- [ ] Multi-facility redistribution optimization
- [ ] Real-time data pipeline with Apache Kafka
- [ ] Docker containerization for easy deployment

---

**Last Updated**: February 2026  
**Version**: 1.0.0  
**Status**: Active Development
>>>>>>> main

