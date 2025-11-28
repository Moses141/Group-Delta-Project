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

