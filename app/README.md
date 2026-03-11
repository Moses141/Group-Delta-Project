# Pharmacy Dashboard

Simple Streamlit dashboard for pharmacists: reorder suggestions, demand trends, and stock context.

## Run instructions

1. **Generate outputs first**  
   Run the Jupyter notebooks (01 → 05) so that these files exist:
   - `outputs/monthly_demand.csv`
   - `outputs/next_3_month_forecast.csv`  
   Optionally, `data/stock_receipts.csv` and `data/sales_transactions.csv` are used for drug names and stock context.

2. **Install dependencies**  
   From the `pharmacy_forecasting` folder:
   ```bash
   pip install -r app/requirements.txt
   ```

3. **Start the dashboard**  
   From the `pharmacy_forecasting` folder:
   ```bash
   streamlit run app/dashboard.py
   ```
   Or from inside `app/`:
   ```bash
   streamlit run dashboard.py
   ```

The app will open in your browser. Use the sidebar to select a drug and filter by category.
