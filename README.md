# Group Delta
- Puoch Mabor Makuei, S23B23/055
- Kiisa Angela, S23B23/027
- Nkangi Moses, M23B23/027

# Pharmacy Demand & Stock Monitoring Dashboard

A comprehensive Streamlit dashboard for predicting medicine demand using SARIMA models and monitoring stock levels in real-time.

## Features

- 📥 **Record Stock**: Record drug purchases (sales) and restocks through the dashboard; data is persisted in a local SQLite database
- 📈 **Demand Forecasting**: Uses SARIMA time series models to predict future medication demand
- 📦 **Stock Monitoring**: Real-time stock level tracking with alerts for low stock (automatically updated from recorded transactions)
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

### 1. Record Stock Tab
- Record **purchases** (sales to customers) or **restocks** (new inventory received)
- Select drug and location; transactions update stock levels across the dashboard
- View recent transactions for the selected medication
- Data stored in `pharmacy_stock.db` (SQLite, created automatically)

### 2. Demand Forecast Tab
- Visual forecast chart with confidence intervals
- Forecasted demand for the next N weeks
- Historical vs predicted demand comparison

### 3. Stock Analysis Tab
- Stock level projections over time
- Reorder point and safety stock calculations
- Stock status indicators (Adequate, Low, Critical)
- Order recommendations

### 4. Multi-Branch Monitoring Tab (when branch data available)
- Stock levels by facility/region
- Branch-level alerts for critical or low stock

### 5. Historical Trends Tab
- Time series visualization of historical demand
- Monthly average demand patterns
- Statistical summary of demand data

### 6. Detailed View Tab
- Complete medication information
- Operational metrics (lead time, reorder levels, etc.)
- Raw data preview

## Configuration

### Sidebar Controls

- **Select Medication**: Choose which medication to analyze
- **Forecast Weeks**: Number of weeks to forecast (1-12)
- **Current Stock**: Defaults to database total (initial + restocks - purchases); can override manually
- **Lead Time**: Days required for new stock delivery
- **Safety Stock %**: Percentage buffer for safety stock

## Stock Status Indicators

- 🟢 **Adequate**: Stock levels are healthy
- 🟡 **Reorder Soon**: Consider reordering in near future
- 🟠 **Low**: Action required - reorder within 2-3 days
- 🔴 **Critical**: Urgent - reorder immediately

## Automatic Model Retraining

Models retrain automatically using:
- **Data**: Last 365 days only (CSV + database purchase transactions)
- **Schedule**: Daily at 2:00 AM (when configured)
- **Database**: Purchase transactions from the Record Stock tab improve demand forecasts

### Set up daily retraining at 2am (Windows)

```bash
python setup_daily_retrain.py
```

To remove the scheduled task:
```bash
python setup_daily_retrain.py remove
```

Or run retraining manually anytime:
```bash
python train_models.py
```

## Model Details

The dashboard uses SARIMA (Seasonal AutoRegressive Integrated Moving Average) models:
- **Order**: (1,1,1) - ARIMA parameters
- **Seasonal Order**: (1,0,1,52) for weekly data with yearly seasonality
- **Training**: Uses 80% of historical data; data limited to last 365 days (CSV + DB purchases)
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

## Database

Stock transactions are stored in `pharmacy_stock.db` (SQLite). The dashboard computes current stock as:
- **Initial stock** (from CSV) + **Restocks** − **Purchases** per drug/location

No database setup required—the file is created automatically on first run.

## Future Enhancements

- [ ] Multi-medication comparison
- [ ] Export forecasts to CSV/Excel
- [ ] Email/SMS alerts for critical stock
- [ ] Integration with inventory management systems
- [ ] Advanced model selection (auto ARIMA)
- [ ] Batch forecasting for all medications

