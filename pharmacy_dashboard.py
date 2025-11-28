import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import os
warnings.filterwarnings("ignore")

# Model imports
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Pharmacy Demand & Stock Dashboard",
    page_icon="💊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the drug supply data"""
    try:
        df = pd.read_csv("uganda_drug_supply_synthetic.csv")
        df['stock_received_date'] = pd.to_datetime(df['stock_received_date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Data file 'uganda_drug_supply_synthetic.csv' not found!")
        return None

@st.cache_data
def get_medications(df):
    """Get list of available medications"""
    if 'drug_name' in df.columns:
        return sorted(df['drug_name'].unique().tolist())
    return []

def prepare_time_series(df, medication, target_col='average_monthly_demand', time_col='stock_received_date'):
    """Prepare time series data for a specific medication"""
    df_med = df[df['drug_name'] == medication].copy() if 'drug_name' in df.columns else df.copy()
    df_med = df_med.sort_values(time_col).reset_index(drop=True)
    
    # Resample to weekly (mean for average_monthly_demand)
    series = df_med.set_index(time_col)[target_col].resample("W").mean().to_frame("y")
    series['y'] = series['y'].interpolate(limit_direction='both').ffill().bfill()
    
    # Apply log transform if all values positive
    apply_log = (series['y'] > 0).all()
    if apply_log:
        series['y'] = np.log1p(series['y'])
    
    return series, apply_log

@st.cache_resource
def load_pretrained_model(medication):
    """Load a pre-trained model if available"""
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        return None
    
    # Try to load models index
    index_file = os.path.join(models_dir, "models_index.pkl")
    if os.path.exists(index_file):
        try:
            with open(index_file, 'rb') as f:
                models_index = pickle.load(f)
            
            if medication in models_index:
                model_file = models_index[medication]['file']
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    return model_data
        except Exception as e:
            st.warning(f"Could not load pre-trained model: {e}")
    
    return None

def train_sarima_model(series, train_size=0.8):
    """Train SARIMA model on the time series"""
    n = len(series)
    train_size_idx = int(n * train_size)
    train_series = series.iloc[:train_size_idx]['y']
    
    # Determine SARIMA order based on data length
    sarima_order = (1, 1, 1)
    if len(train_series) >= 60:
        seasonal_order = (1, 0, 1, 52)  # Weekly data, 52 weeks = 1 year
    else:
        seasonal_order = (0, 0, 0, 0)
    
    try:
        model = sm.tsa.SARIMAX(
            train_series,
            order=sarima_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        return res, train_series, train_size_idx
    except Exception as e:
        st.error(f"SARIMA model training failed: {e}")
        return None, train_series, train_size_idx

def forecast_demand(model, steps, apply_log=False):
    """Generate demand forecast"""
    try:
        forecast = model.get_forecast(steps=steps)
        pred_mean = forecast.predicted_mean.values
        conf_int = forecast.conf_int().values
        
        if apply_log:
            pred_mean = np.expm1(pred_mean)
            conf_int = np.expm1(conf_int)
        
        return pred_mean, conf_int
    except Exception as e:
        st.error(f"Forecast generation failed: {e}")
        return None, None

def calculate_stock_status(current_stock, predicted_demand, lead_time_days=7, safety_stock_pct=0.2):
    """Calculate stock status and reorder recommendations"""
    # Convert weekly demand to daily
    daily_demand = predicted_demand / 7
    
    # Calculate days of stock remaining
    if daily_demand > 0:
        days_remaining = current_stock / daily_demand
    else:
        days_remaining = np.inf
    
    # Calculate reorder point (demand during lead time + safety stock)
    demand_during_lead = daily_demand * lead_time_days
    safety_stock = demand_during_lead * safety_stock_pct
    reorder_point = demand_during_lead + safety_stock
    
    # Determine status
    if days_remaining < lead_time_days:
        status = "Critical"
        color = "red"
    elif days_remaining < lead_time_days * 1.5:
        status = "Low"
        color = "orange"
    elif current_stock < reorder_point:
        status = "Reorder Soon"
        color = "yellow"
    else:
        status = "Adequate"
        color = "green"
    
    return {
        'status': status,
        'color': color,
        'days_remaining': days_remaining,
        'reorder_point': reorder_point,
        'safety_stock': safety_stock,
        'daily_demand': daily_demand
    }

def main():
    st.markdown('<p class="main-header">💊 Pharmacy Demand & Stock Monitoring Dashboard</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Medication selection
    medications = get_medications(df)
    if not medications:
        st.error("No medications found in data. Please check the data file.")
        return
    
    selected_med = st.sidebar.selectbox(
        "Select Medication",
        medications,
        index=0
    )
    
    # Forecast parameters
    st.sidebar.subheader("Forecast Settings")
    forecast_weeks = st.sidebar.slider("Forecast Weeks", 1, 12, 4)
    
    # Stock monitoring parameters
    st.sidebar.subheader("Stock Monitoring")
    current_stock = st.sidebar.number_input(
        "Current Stock (units)",
        min_value=0,
        value=10000,
        step=100
    )
    lead_time_days = st.sidebar.number_input(
        "Lead Time (days)",
        min_value=1,
        value=7,
        step=1
    )
    safety_stock_pct = st.sidebar.slider(
        "Safety Stock (%)",
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.05
    )
    
    # Get medication data
    med_data = df[df['drug_name'] == selected_med] if 'drug_name' in df.columns else df
    
    # Prepare time series
    with st.spinner("Preparing data and loading model..."):
        # Try to load pre-trained model first
        pretrained_data = load_pretrained_model(selected_med)
        
        if pretrained_data is not None:
            # Use pre-trained model
            model = pretrained_data['model']
            series = pretrained_data['series']
            apply_log = pretrained_data['apply_log']
            train_size_idx = pretrained_data['train_size_idx']
            st.success(f"✅ Using pre-trained model (trained on {pretrained_data.get('trained_date', 'unknown date')})")
        else:
            # Train new model
            series, apply_log = prepare_time_series(df, selected_med)
            
            if len(series) < 20:
                st.error(f"Insufficient data for {selected_med}. Need at least 20 data points.")
                return
            
            # Train SARIMA model
            model, train_series, train_size_idx = train_sarima_model(series)
            
            if model is None:
                return
            
            st.info("💡 Tip: Run 'python train_models.py' to pre-train models for faster loading")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate current metrics
    latest_demand = series['y'].iloc[-1]
    if apply_log:
        latest_demand = np.expm1(latest_demand)
    
    avg_demand = series['y'].mean()
    if apply_log:
        avg_demand = np.expm1(avg_demand)
    
    # Forecast next period
    pred_mean, conf_int = forecast_demand(model, forecast_weeks, apply_log)
    
    if pred_mean is None:
        return
    
    next_week_demand = pred_mean[0] if len(pred_mean) > 0 else 0
    
    # Stock status
    stock_status = calculate_stock_status(
        current_stock,
        next_week_demand,
        lead_time_days,
        safety_stock_pct
    )
    
    with col1:
        st.metric(
            "Current Stock",
            f"{current_stock:,.0f}",
            f"{stock_status['days_remaining']:.1f} days remaining"
        )
    
    with col2:
        st.metric(
            "Next Week Demand",
            f"{next_week_demand:,.0f}",
            f"±{abs(conf_int[0][1] - conf_int[0][0]):,.0f}" if conf_int is not None else ""
        )
    
    with col3:
        status_color = {
            "Critical": "🔴",
            "Low": "🟠",
            "Reorder Soon": "🟡",
            "Adequate": "🟢"
        }
        st.metric(
            "Stock Status",
            f"{status_color.get(stock_status['status'], '⚪')} {stock_status['status']}",
            f"Reorder at {stock_status['reorder_point']:,.0f}"
        )
    
    with col4:
        st.metric(
            "Average Demand",
            f"{avg_demand:,.0f}",
            f"Latest: {latest_demand:,.0f}"
        )
    
    # Stock status alert
    if stock_status['status'] in ['Critical', 'Low']:
        st.markdown(f"""
        <div class="{'danger-card' if stock_status['status'] == 'Critical' else 'warning-card'}">
            <strong>⚠️ Alert:</strong> Stock level is {stock_status['status'].lower()}! 
            Current stock will last approximately {stock_status['days_remaining']:.1f} days. 
            Consider reordering immediately.
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Demand Forecast", "📦 Stock Analysis", "📊 Historical Trends", "🔍 Detailed View"])
    
    with tab1:
        st.subheader("Demand Forecast")
        
        # Prepare forecast dates
        last_date = series.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(weeks=1),
            periods=forecast_weeks,
            freq='W'
        )
        
        # Historical data - filter to show only last year from current date
        one_year_ago = last_date - timedelta(days=365)
        historical_mask = series.index >= one_year_ago
        historical_dates = series.index[historical_mask]
        historical_values = series['y'].values[historical_mask]
        if apply_log:
            historical_values = np.expm1(historical_values)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Historical data
        ax.plot(historical_dates, historical_values, 'b-', label='Historical Demand', linewidth=2)
        
        # Forecast
        ax.plot(forecast_dates, pred_mean, 'r--', label='Forecasted Demand', linewidth=2, marker='o')
        
        # Confidence intervals
        if conf_int is not None:
            ax.fill_between(
                forecast_dates,
                conf_int[:, 0],
                conf_int[:, 1],
                alpha=0.3,
                color='red',
                label='95% Confidence Interval'
            )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Demand (units)', fontsize=12)
        ax.set_title(f'Demand Forecast for {selected_med}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Forecast table
        st.subheader("Forecast Details")
        forecast_df = pd.DataFrame({
            'Week': forecast_dates.strftime('%Y-%m-%d'),
            'Forecasted Demand': pred_mean,
            'Lower Bound (95%)': conf_int[:, 0] if conf_int is not None else pred_mean,
            'Upper Bound (95%)': conf_int[:, 1] if conf_int is not None else pred_mean
        })
        forecast_df['Forecasted Demand'] = forecast_df['Forecasted Demand'].round(0)
        forecast_df['Lower Bound (95%)'] = forecast_df['Lower Bound (95%)'].round(0)
        forecast_df['Upper Bound (95%)'] = forecast_df['Upper Bound (95%)'].round(0)
        st.dataframe(forecast_df, use_container_width=True)
    
    with tab2:
        st.subheader("Stock Level Analysis")
        
        # Calculate stock projections
        daily_demand = stock_status['daily_demand']
        days_projected = min(90, int(stock_status['days_remaining'] * 1.5))
        projection_dates = pd.date_range(
            start=datetime.now(),
            periods=days_projected,
            freq='D'
        )
        
        projected_stock = []
        stock = current_stock
        for i, date in enumerate(projection_dates):
            if stock > 0:
                stock = max(0, stock - daily_demand)
            projected_stock.append(stock)
        
        # Stock projection plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(projection_dates, projected_stock, 'b-', linewidth=2, label='Projected Stock')
        ax.axhline(y=stock_status['reorder_point'], color='orange', linestyle='--', label='Reorder Point')
        ax.axhline(y=stock_status['safety_stock'], color='red', linestyle='--', label='Safety Stock')
        ax.fill_between(projection_dates, 0, stock_status['safety_stock'], alpha=0.2, color='red', label='Critical Zone')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Stock Level (units)', fontsize=12)
        ax.set_title(f'Stock Projection for {selected_med}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Stock metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Stock Metrics")
            st.write(f"**Current Stock:** {current_stock:,.0f} units")
            st.write(f"**Reorder Point:** {stock_status['reorder_point']:,.0f} units")
            st.write(f"**Safety Stock:** {stock_status['safety_stock']:,.0f} units")
        
        with col2:
            st.markdown("### ⏱️ Time Metrics")
            st.write(f"**Days Remaining:** {stock_status['days_remaining']:.1f} days")
            st.write(f"**Lead Time:** {lead_time_days} days")
            st.write(f"**Daily Demand:** {daily_demand:,.1f} units/day")
        
        with col3:
            st.markdown("### 📋 Recommendations")
            if stock_status['status'] == 'Critical':
                st.error("🚨 **URGENT:** Reorder immediately!")
            elif stock_status['status'] == 'Low':
                st.warning("⚠️ **Action Required:** Reorder within 2-3 days")
            elif stock_status['status'] == 'Reorder Soon':
                st.info("ℹ️ **Monitor:** Consider reordering soon")
            else:
                st.success("✅ **Adequate:** Stock levels are healthy")
            
            recommended_order = max(0, stock_status['reorder_point'] - current_stock + stock_status['safety_stock'])
            if recommended_order > 0:
                st.write(f"**Recommended Order:** {recommended_order:,.0f} units")
    
    with tab3:
        st.subheader("Historical Demand Trends")
        
        # Prepare historical data
        hist_values = series['y'].values
        if apply_log:
            hist_values = np.expm1(hist_values)
        
        # Monthly aggregation
        monthly_data = series.copy()
        monthly_data['y'] = hist_values
        monthly_data['month'] = monthly_data.index.month
        monthly_data['year'] = monthly_data.index.year
        monthly_agg = monthly_data.groupby(['year', 'month'])['y'].mean().reset_index()
        monthly_agg['date'] = pd.to_datetime(monthly_agg[['year', 'month']].assign(day=1))
        
        # Plot historical trends
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Time series plot
        axes[0].plot(series.index, hist_values, 'b-', linewidth=2)
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Demand (units)', fontsize=12)
        axes[0].set_title('Historical Demand Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # Monthly average
        axes[1].bar(monthly_agg['date'], monthly_agg['y'], width=20, alpha=0.7, color='steelblue')
        axes[1].set_xlabel('Month', fontsize=12)
        axes[1].set_ylabel('Average Demand (units)', fontsize=12)
        axes[1].set_title('Monthly Average Demand', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Demand Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{hist_values.mean():,.0f}",
                    f"{np.median(hist_values):,.0f}",
                    f"{hist_values.std():,.0f}",
                    f"{hist_values.min():,.0f}",
                    f"{hist_values.max():,.0f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📅 Recent Demand")
            recent_data = pd.DataFrame({
                'Date': series.index[-10:].strftime('%Y-%m-%d'),
                'Demand': hist_values[-10:]
            })
            recent_data['Demand'] = recent_data['Demand'].round(0)
            st.dataframe(recent_data, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("Detailed Medication Information")
        
        # Medication details
        med_details = med_data.iloc[0] if len(med_data) > 0 else None
        
        if med_details is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏥 Medication Details")
                detail_fields = {
                    'Drug Name': 'drug_name',
                    'Manufacturer Country': 'manufacturer_country',
                    'License Holder': 'license_holder',
                    'Distribution Region': 'distribution_region',
                    'Facility Type': 'facility_type'
                }
                
                for label, field in detail_fields.items():
                    if field in med_details:
                        st.write(f"**{label}:** {med_details[field]}")
            
            with col2:
                st.markdown("### 📊 Operational Metrics")
                metric_fields = {
                    'Reorder Level': 'reorder_level',
                    'Lead Time (days)': 'lead_time_days',
                    'Delivery Frequency (days)': 'delivery_frequency_days',
                    'Supplier Reliability': 'supplier_reliability_score'
                }
                
                for label, field in metric_fields.items():
                    if field in med_details:
                        value = med_details[field]
                        if isinstance(value, (int, float)):
                            st.write(f"**{label}:** {value:,.0f}" if isinstance(value, float) else f"**{label}:** {value:,}")
                        else:
                            st.write(f"**{label}:** {value}")
        
        # Raw data preview
        st.markdown("### 📋 Raw Data Preview")
        st.dataframe(med_data.head(20), use_container_width=True)

if __name__ == "__main__":
    main()

