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

