<<<<<<< HEAD
# Quick Start Guide - Pharmacy Dashboard

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements_dashboard.txt
```

### Step 2: (Optional) Pre-train Models for Faster Loading
```bash
python train_models.py
```
This will train SARIMA models for all medications and save them. The dashboard will load faster if models are pre-trained.

### Step 3: Launch Dashboard
```bash
streamlit run pharmacy_dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## 📋 What You'll See

1. **Main Dashboard**: Overview metrics and stock status
2. **Demand Forecast Tab**: Visual predictions with confidence intervals
3. **Stock Analysis Tab**: Stock projections and reorder recommendations
4. **Historical Trends Tab**: Past demand patterns and statistics
5. **Detailed View Tab**: Complete medication information

## 🎛️ Using the Dashboard

### Selecting a Medication
- Use the dropdown in the sidebar to select any medication from your dataset

### Adjusting Forecast Settings
- **Forecast Weeks**: Choose how many weeks ahead to predict (1-12 weeks)
- The model will automatically generate forecasts with 95% confidence intervals

### Monitoring Stock Levels
1. Enter your **Current Stock** level in the sidebar
2. Set **Lead Time** (days until new stock arrives)
3. Adjust **Safety Stock %** (recommended: 20%)
4. The dashboard will automatically calculate:
   - Days of stock remaining
   - Reorder point
   - Stock status (Adequate/Low/Critical)

### Understanding Stock Status
- 🟢 **Adequate**: Stock is healthy, no action needed
- 🟡 **Reorder Soon**: Consider placing an order
- 🟠 **Low**: Action required - reorder within 2-3 days
- 🔴 **Critical**: Urgent - reorder immediately!

## 💡 Tips

1. **Pre-train Models**: Run `train_models.py` regularly (e.g., weekly) to keep models updated with new data
2. **Monitor Multiple Medications**: Switch between medications using the sidebar dropdown
3. **Export Data**: Use the forecast tables to copy data for reports
4. **Adjust Safety Stock**: Increase safety stock % for critical medications or unreliable suppliers

## 🔧 Troubleshooting

**Dashboard won't start?**
- Check that Streamlit is installed: `pip install streamlit`
- Verify the data file exists: `uganda_drug_supply_synthetic.csv`

**Models train slowly?**
- Pre-train models using `train_models.py` for faster loading
- Models are cached, so switching medications is fast after first load

**Insufficient data error?**
- Need at least 20 data points per medication
- Check that your data file has enough historical records

## 📊 Example Workflow

1. **Morning Check**: Open dashboard, select medication, check stock status
2. **Review Forecast**: Look at demand forecast for next 4 weeks
3. **Check Stock Analysis**: See if reorder is needed
4. **Make Decision**: Place order if status is Low or Critical
5. **Monitor**: Check back daily for critical medications

## 🎯 Best Practices

- Review forecasts weekly for all medications
- Update current stock levels daily
- Adjust lead times based on supplier reliability
- Use higher safety stock for critical medications
- Export forecasts for planning meetings

Happy forecasting! 📈💊
=======
# 🚀 Quick Start Guide

Get MedSupply AI up and running in 5 minutes!

## Prerequisites Check

```bash
python --version  # Should be 3.8 or higher
pip --version     # Should be installed
```

## Installation (3 Steps)

### Step 1: Clone Repository
```bash
git clone https://github.com/Moses141/Group-Delta-Project.git
cd Group-Delta-Project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit; print(' All packages installed!')"
```

## Running the System

### Option A: Interactive Dashboard (Recommended for First-Time Users)

```bash
streamlit run medsupply_dashboard.py
```

**What you'll see:**
- Dashboard opens at `http://localhost:8501`
- 4 pages: Dashboard, Facility Analysis, Model Performance, OOQ Calculator
- Interactive charts and real-time predictions

### Option B: Jupyter Notebook (For Data Scientists)

```bash
jupyter notebook inventory_management.ipynb
```

**What to do:**
1. Run all cells in order (Cell → Run All)
2. Wait for model training (5-10 minutes)
3. View visualizations and results

### Option C: Standalone Script (For Quick Model Comparison)

```bash
python run_model_comparison.py
```

**What happens:**
- Trains all 4 models (SARIMA, RF, XGBoost, LSTM)
- Generates comparison visualizations
- Saves `model_comparison_comprehensive.png`

## First Prediction

### Using the Dashboard

1. Open dashboard: `streamlit run medsupply_dashboard.py`
2. Go to **"OOQ Calculator"** page
3. Enter values:
   - Current Stock: 1000
   - Weekly Demand: 500
   - Lead Time: 21 days
4. Click **"Calculate OOQ"**
5. See recommended order quantity!

### Using Python Code

```python
import pandas as pd
from inventory_management import MedicineSupplyForecaster

# Load your trained model (after running notebook)
# forecaster = MedicineSupplyForecaster(...)

# Make a prediction
facility_data = pd.Series({
    'initial_stock_units': 1000,
    'average_monthly_demand': 2000,
    'lead_time_days': 21,
    'reorder_level': 500,
    # ... other required features
})

# predictions = forecaster.predict(facility_data)
# print(f"Order {predictions['optimal_order_quantity']:.0f} units")
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: 
```bash
pip install streamlit
```

### Issue: "TensorFlow not available"
**Solution**: 
- LSTM will use fallback (this is OK)
- Or install: `pip install tensorflow`

### Issue: "FileNotFoundError: uganda_drug_supply_synthetic.csv"
**Solution**: 
- Ensure you're in the project root directory
- Check file exists: `ls uganda_drug_supply_synthetic.csv`

### Issue: Dashboard won't start
**Solution**: 
```bash
# Check if port 8501 is in use
netstat -ano | findstr :8501

# Use different port
streamlit run medsupply_dashboard.py --server.port 8502
```

## Next Steps

1. ✅ Run the dashboard and explore features
2. 📊 Review model performance metrics
3. 🔧 Customize for your data (see README.md)
4. 🚀 Deploy to production (see ARCHITECTURE.md)

## Getting Help

- 📖 Read [README.md](README.md) for full documentation
- 🏗️ See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- 🐛 Report issues on [GitHub Issues](https://github.com/Moses141/Group-Delta-Project/issues)

---

**Happy Forecasting! 🎯**
>>>>>>> main

