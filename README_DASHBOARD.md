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

