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

# Database for stock transactions
from db_pharmacy import (
    init_db,
    record_transaction,
    get_transactions_for_drug,
    get_net_transactions_by_drug,
    get_inventory_locations,
)

# Page configuration
st.set_page_config(
    page_title="Delta",
    page_icon="logo.png",
    layout="wide"
)

# Initialize database for stock transactions
init_db()

# Custom CSS - Dark palette, professional design system
st.markdown("""
<style>
    /* Base - dark theme */
    .stApp { background: #0d1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
    
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; color: #e5e7eb; }
    
    /* Header - dark branding */
    .dashboard-header {
        text-align: center;
        padding: 2rem 0 2.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .dashboard-header .brand { font-size: 1.75rem; font-weight: 700; color: #e5e7eb; letter-spacing: -0.02em; }
    .dashboard-header .tagline { font-size: 0.95rem; color: #9ca3af; font-weight: 500; margin-top: 0.35rem; }
    .dashboard-header img { vertical-align: middle; margin-right: 0.75rem; }
    
    /* Sidebar - dark */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] { font-weight: 600; color: #e5e7eb !important; }
    [data-testid="stSidebar"] .stSelectbox > div { border-radius: 8px; }
    [data-testid="stSidebar"] hr { margin: 1rem 0; border-color: rgba(255, 255, 255, 0.08); }
    
    /* Metric cards - dark elevated */
    div[data-testid="stMetric"] {
        background: #161b22;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.06);
        transition: box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
    div[data-testid="stMetric"] label { font-size: 0.8rem !important; color: #9ca3af !important; font-weight: 500 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; color: #e5e7eb !important; }
    
    /* Tabs - dark pill style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(22, 27, 34, 0.8);
        padding: 0.35rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] { background: #21262d !important; color: #e5e7eb !important; box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important; }
    
    /* DataFrames - dark */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Section headers - light on dark */
    h2, h3 { color: #e5e7eb !important; font-weight: 600 !important; letter-spacing: -0.01em !important; }
    p, span, label { color: #e5e7eb !important; }
    
    /* Alert cards - dark with colored accents */
    .warning-card, .danger-card, .info-card {
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        font-size: 0.95rem;
        border: none;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        font-weight: 500;
    }
    .warning-card { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .danger-card { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }
    .info-card { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3); }
    
    /* Alerts */
    [data-testid="stAlert"] { border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); }
    
    /* Form elements - dark */
    .stForm { border-radius: 10px; padding: 1.5rem; background: #161b22; border: 1px solid rgba(255, 255, 255, 0.06); box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
    
    /* Dividers */
    hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent); margin: 1.5rem 0; }
    
    /* Main content - transparent so dark app bg shows */
    .main .block-container { background: transparent; }
    [data-testid="stCaptionContainer"] { color: #9ca3af !important; }
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

@st.cache_data
def get_branches(df):
    """Get list of available branches/facilities"""
    # Try different possible column names for branches
    branch_cols = ['facility_type', 'distribution_region', 'branch', 'facility', 'location']
    for col in branch_cols:
        if col in df.columns:
            branches = sorted(df[col].unique().tolist())
            return branches, col
    return [], None

def get_branch_stock_for_medication(df, medication, branch_col, net_by_drug_id=None):
    """Get per-branch stock data for a medication (DB-adjusted)."""
    return get_branch_stock_data(df, medication, branch_col, net_by_drug_id)


def get_branch_stock_data(df, medication, branch_col, net_by_drug_id=None):
    """Get stock data for a medication across all branches, adjusted by DB transactions."""
    if branch_col is None or 'drug_name' not in df.columns:
        return None
    
    if net_by_drug_id is None:
        net_by_drug_id = get_net_transactions_by_drug(df)
    
    # Get latest stock data per branch
    branch_data = []
    branches = df[branch_col].unique()
    
    for branch in branches:
        branch_df = df[(df['drug_name'] == medication) & (df[branch_col] == branch)].copy()
        if len(branch_df) > 0:
            # Get most recent record for this branch
            branch_df = branch_df.sort_values('stock_received_date', ascending=False)
            latest = branch_df.iloc[0]
            drug_id = latest.get('drug_id')
            base_stock = latest.get('initial_stock_units', 0) or 0
            net = net_by_drug_id.get(drug_id, 0) if net_by_drug_id else 0
            current_stock = base_stock + net
            
            branch_data.append({
                'branch': branch,
                'current_stock': max(0, current_stock),
                'reorder_level': latest.get('reorder_level', 0),
                'average_demand': latest.get('average_monthly_demand', 0),
                'last_update': latest.get('stock_received_date', None),
                'lead_time': latest.get('lead_time_days', 7),
                'supplier_reliability': latest.get('supplier_reliability_score', 0)
            })
    
    return pd.DataFrame(branch_data) if branch_data else None

# Use shared model data preparation (365-day lookback, merges DB purchase data)
from model_data import prepare_time_series

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
    import base64
    logo_html = ""
    try:
        with open("logo.png", "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="48" height="48" style="vertical-align: middle; margin-right: 0.75rem;">'
    except FileNotFoundError:
        logo_html = ""

    st.markdown(f"""
    <div class="dashboard-header">
        <div class="brand">{logo_html}Delta Dashboard</div>
        <div class="tagline">Pharmacy Demand & Stock Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.markdown("### Controls")
    
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
    
    # Get per-branch stock for selected medication
    branches, branch_col = get_branches(df)
    net_by_drug_id = get_net_transactions_by_drug(df)
    branch_stock_df = get_branch_stock_for_medication(df, selected_med, branch_col, net_by_drug_id)
    
    # Branch selection for stock (per-branch view)
    branch_options = []
    if branch_stock_df is not None and len(branch_stock_df) > 0:
        branch_options = branch_stock_df['branch'].tolist()
    
    selected_branch = None
    db_stock = 10000  # default when no branch data
    if branch_options:
        selected_branch = st.sidebar.selectbox(
            "Branch",
            branch_options,
            index=0,
            key=f"branch_select_{selected_med}"
        )
        row = branch_stock_df[branch_stock_df['branch'] == selected_branch].iloc[0]
        db_stock = int(row['current_stock'])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Forecast**")
    forecast_weeks = st.sidebar.slider("Forecast Weeks", 1, 12, 4)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Stock**")
    current_stock = st.sidebar.number_input(
        "Current Stock (units)",
        min_value=0,
        value=db_stock,
        step=100,
        key=f"current_stock_{selected_med}_{selected_branch or 'all'}_{db_stock}",
        help=f"Stock for selected branch. Record purchases/restocks in the Record Stock tab."
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
    with st.spinner("Loading..."):
        # Try to load pre-trained model first
        pretrained_data = load_pretrained_model(selected_med)
        
        if pretrained_data is not None:
            # Use pre-trained model
            model = pretrained_data['model']
            series = pretrained_data['series']
            apply_log = pretrained_data['apply_log']
            train_size_idx = pretrained_data['train_size_idx']
            st.caption("Using cached forecast model")
        else:
            # Train new model
            series, apply_log = prepare_time_series(df, selected_med, lookback_days=365)
            
            if len(series) < 20:
                st.error(f"Insufficient data for {selected_med}. Need at least 20 data points.")
                return
            
            # Train SARIMA model
            model, train_series, train_size_idx = train_sarima_model(series)
            
            if model is None:
                return
            
            
    
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
            f"{next_week_demand:,.0f}"
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
            f"{status_color.get(stock_status['status'], '⚪')} {stock_status['status']}"
        )
    
    with col4:
        st.metric(
            "Average Demand",
            f"{avg_demand:,.0f}"
        )
    
    # Stock status alert
    if stock_status['status'] in ['Critical', 'Low']:
        st.markdown(f"""
        <div class="{'danger-card' if stock_status['status'] == 'Critical' else 'warning-card'}">
            Stock level is {stock_status['status'].lower()} — approximately {stock_status['days_remaining']:.1f} days remaining. Consider reordering.
        </div>
        """, unsafe_allow_html=True)
    
    # has_branches uses branches, branch_col from earlier
    has_branches = len(branches) > 0 and branch_col is not None
    
    # Tabs for different views
    if has_branches:
        tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📥 Record Stock", "📈 Demand Forecast", "📦 Stock Analysis", 
            "🏢 Multi-Branch Monitoring", "📊 Historical Trends", "🔍 Detailed View"
        ])
    else:
        tab0, tab1, tab2, tab3, tab4 = st.tabs([
            "📥 Record Stock", "📈 Demand Forecast", "📦 Stock Analysis", 
            "📊 Historical Trends", "🔍 Detailed View"
        ])
    
    # Record Stock tab - input purchases and restocks
    with tab0:
        st.markdown("#### Record transaction")
        st.caption("Record purchases (sales) or restocks to update stock levels.")
        
        locations = get_inventory_locations(df, drug_name=selected_med)
        if not locations:
            locations = get_inventory_locations(df)
        if not locations:
            st.warning("No inventory locations found. Check that the data file has drug_id and facility_type.")
        else:
            with st.form("record_stock_form", clear_on_submit=True):
                loc_options = {loc['label']: loc for loc in locations}
                selected_label = st.selectbox(
                    "Inventory Location",
                    options=list(loc_options.keys()),
                    help="Select the drug and facility to update"
                )
                loc = loc_options[selected_label]
                
                trans_type = st.radio("Transaction Type", ["restock", "purchase"], 
                    format_func=lambda x: "Restock (add inventory)" if x == "restock" else "Purchase (sale to customer)")
                quantity = st.number_input("Quantity (units)", min_value=1, value=100, step=1)
                trans_date = st.date_input("Transaction Date", value=datetime.now().date())
                notes = st.text_input("Notes (optional)", placeholder="e.g., Batch #, supplier name")
                
                submitted = st.form_submit_button("Record Transaction")
            
            if submitted:
                ok = record_transaction(
                    drug_id=loc['drug_id'],
                    drug_name=loc['drug_name'],
                    location_key=loc['location_key'],
                    transaction_type=trans_type,
                    quantity=quantity,
                    transaction_date=trans_date,
                    notes=notes
                )
                if ok:
                    st.success(f"✅ Recorded {quantity} units {trans_type} for {loc['drug_name']} at {loc['location_key']}. Stock levels updated.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Failed to record transaction. Please check your inputs.")
            
            # Show recent transactions
            st.markdown("### Recent Transactions")
            trans_df = get_transactions_for_drug(drug_name=selected_med, limit=20)
            if trans_df.empty:
                st.info("No transactions recorded yet.")
            else:
                display_trans = trans_df[['transaction_date', 'drug_name', 'transaction_type', 'quantity', 'location_key', 'notes']].copy()
                display_trans.columns = ['Date', 'Drug', 'Type', 'Quantity', 'Location', 'Notes']
                st.dataframe(display_trans, use_container_width=True, hide_index=True)
    
    with tab1:
        st.markdown("#### Demand forecast")

        # Prepare forecast dates
        last_date = series.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(weeks=1),
            periods=forecast_weeks,
            freq='W'
        )

        # Historical data - filter to show only last year
        eight_weeks_ago = last_date - timedelta(weeks=8)
        historical_mask = series.index >= eight_weeks_ago
        historical_dates = series.index[historical_mask]
        historical_values = series['y'].values[historical_mask]

        if apply_log:
            historical_values = np.expm1(historical_values)

        # ------------------- Demand forecast chart (dark) -------------------
        import matplotlib.dates as mdates

        all_dates = pd.to_datetime(list(historical_dates) + list(forecast_dates))
        all_values = np.concatenate([historical_values, pred_mean])

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')

        fill_color = '#22c55e'
        line_color = '#22c55e'

        ax.fill_between(all_dates, 0, all_values, color=fill_color, alpha=0.4)
        ax.plot(all_dates, all_values, color=line_color, linewidth=2)

        if conf_int is not None:
            ax.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1],
                            color=line_color, alpha=0.08)

        ax.axvline(
        last_date,
        linestyle='--',
        linewidth=2.5,
        color='#fbbf24',   # bright amber
        alpha=0.9,
        zorder=5
        )

        y_max = ax.get_ylim()[1]

        ax.text(
            last_date,
            y_max * 0.95,
            "  FORECAST  ",
            color='#0d1117',         # dark text
            fontsize=9,
            weight='bold',
            ha='center',
            va='top',
            bbox=dict(
                facecolor='#fbbf24',  # same as separator line
                edgecolor='none',
                boxstyle='round,pad=0.3'
            ),
            zorder=6
        )


        ax.set_title(f"Expected demand for {selected_med}", fontsize=15, weight="500", color='#e5e7eb')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.yaxis.set_tick_params(labelsize=10, colors='#9ca3af')
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.xticks(rotation=0, color='#9ca3af', fontsize=10)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', length=0)
        ax.grid(axis='y', alpha=0.15, color='#6b7280', linestyle='-')
        ax.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig)

        
        st.markdown("---")
        st.markdown("**Forecast details**")
        forecast_df = pd.DataFrame({
            'Week': forecast_dates.strftime('%Y-%m-%d'),
            'Forecasted Demand': pred_mean,
            'Lower Bound (95%)': conf_int[:, 0] if conf_int is not None else pred_mean,
            'Upper Bound (95%)': conf_int[:, 1] if conf_int is not None else pred_mean
        })
        forecast_df['Forecasted Demand'] = forecast_df['Forecasted Demand'].round(0)
        forecast_df['Lower Bound (95%)'] = forecast_df['Lower Bound (95%)'].round(0)
        forecast_df['Upper Bound (95%)'] = forecast_df['Upper Bound (95%)'].round(0)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("#### Stock projection")
        
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
        
        # Stock projection plot - dark
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        ax.fill_between(projection_dates, 0, projected_stock, color='#22c55e', alpha=0.4)
        ax.plot(projection_dates, projected_stock, color='#22c55e', linewidth=2)
        ax.axhline(y=stock_status['reorder_point'], color='#f59e0b', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.axhline(y=stock_status['safety_stock'], color='#ef4444', linestyle='--', alpha=0.6, linewidth=1)
        
        ax.set_title(f"Stock projection — {selected_med}", fontsize=13, weight="500", color='#e5e7eb')
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors='#9ca3af', labelsize=10)
        ax.grid(axis='y', alpha=0.15, color='#6b7280')
        ax.set_axisbelow(True)
        plt.xticks(rotation=25)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Stock metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Stock metrics**")
            st.write(f"**Current Stock:** {current_stock:,.0f} units")
            st.write(f"**Reorder Point:** {stock_status['reorder_point']:,.0f} units")
            st.write(f"**Safety Stock:** {stock_status['safety_stock']:,.0f} units")
        
        with col2:
            st.markdown("**Time metrics**")
            st.write(f"**Days Remaining:** {stock_status['days_remaining']:.1f} days")
            st.write(f"**Lead Time:** {lead_time_days} days")
            st.write(f"**Daily Demand:** {daily_demand:,.1f} units/day")
        
        with col3:
            st.markdown("**Recommendation**")
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
    
    # Multi-Branch Monitoring Tab
    if has_branches:
        with tab3:
            st.markdown("#### Multi-branch overview")
            
            # Get branch stock data
            branch_stock_df = get_branch_stock_data(df, selected_med, branch_col, net_by_drug_id)
            
            if branch_stock_df is not None and len(branch_stock_df) > 0:
                # Calculate stock status for each branch
                branch_stock_df['daily_demand'] = branch_stock_df['average_demand'] / 30
                branch_stock_df['days_remaining'] = branch_stock_df['current_stock'] / branch_stock_df['daily_demand']
                branch_stock_df['days_remaining'] = branch_stock_df['days_remaining'].replace([np.inf, -np.inf], np.nan)
                
                # Determine status
                def get_stock_status(row):
                    days = row['days_remaining']
                    lead_time = row['lead_time']
                    if pd.isna(days):
                        return "Unknown"
                    elif days < lead_time:
                        return "Critical"
                    elif days < lead_time * 1.5:
                        return "Low"
                    elif row['current_stock'] < row['reorder_level']:
                        return "Reorder Soon"
                    else:
                        return "Adequate"
                
                branch_stock_df['status'] = branch_stock_df.apply(get_stock_status, axis=1)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                total_stock = branch_stock_df['current_stock'].sum()
                total_demand = branch_stock_df['average_demand'].sum()
                critical_branches = len(branch_stock_df[branch_stock_df['status'] == 'Critical'])
                low_branches = len(branch_stock_df[branch_stock_df['status'] == 'Low'])
                
                with col1:
                    st.metric("Total Stock", f"{total_stock:,.0f}", "Across all branches")
                with col2:
                    st.metric("Total Demand", f"{total_demand:,.0f}", "Monthly average")
                with col3:
                    st.metric("Critical Branches", critical_branches, f"{low_branches} low stock" if low_branches > 0 else "")
                with col4:
                    avg_days = branch_stock_df['days_remaining'].mean()
                    st.metric("Avg Days Remaining", f"{avg_days:.1f}", "Across branches")
                
                # Stock comparison chart - dark
                st.markdown("**Stock by branch**")
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                fig.patch.set_facecolor('#0d1117')
                for ax in axes:
                    ax.set_facecolor('#0d1117')
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.tick_params(colors='#9ca3af', labelsize=10)
                    ax.grid(axis='x', alpha=0.15, color='#6b7280')
                    ax.set_axisbelow(True)
                
                branch_sorted = branch_stock_df.sort_values('current_stock', ascending=True)
                colors = {'Critical': '#ef4444', 'Low': '#f59e0b', 'Reorder Soon': '#eab308', 'Adequate': '#22c55e', 'Unknown': '#6b7280'}
                bar_colors = [colors.get(s, '#6b7280') for s in branch_sorted['status']]
                
                axes[0].barh(branch_sorted['branch'], branch_sorted['current_stock'], color=bar_colors, alpha=0.85, height=0.6)
                axes[0].axvline(x=branch_sorted['reorder_level'].mean(), color='#6b7280', linestyle='--', alpha=0.5, linewidth=1)
                axes[0].set_title('Stock levels', fontsize=13, weight='500', color='#e5e7eb')
                axes[0].set_xlabel('')
                
                axes[1].barh(branch_sorted['branch'], branch_sorted['days_remaining'], color=bar_colors, alpha=0.85, height=0.6)
                axes[1].axvline(x=branch_sorted['lead_time'].mean(), color='#6b7280', linestyle='--', alpha=0.5, linewidth=1)
                axes[1].set_title('Days remaining', fontsize=13, weight='500', color='#e5e7eb')
                axes[1].set_xlabel('')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Detailed branch table
                st.markdown("**Branch details**")
                display_df = branch_stock_df[[
                    'branch', 'current_stock', 'reorder_level', 
                    'average_demand', 'days_remaining', 'status', 'lead_time'
                ]].copy()
                display_df.columns = ['Branch', 'Current Stock', 'Reorder Level', 
                                     'Avg Monthly Demand', 'Days Remaining', 'Status', 'Lead Time (days)']
                display_df['Current Stock'] = display_df['Current Stock'].round(0).astype(int)
                display_df['Reorder Level'] = display_df['Reorder Level'].round(0).astype(int)
                display_df['Avg Monthly Demand'] = display_df['Avg Monthly Demand'].round(0).astype(int)
                display_df['Days Remaining'] = display_df['Days Remaining'].round(1)
                display_df['Lead Time (days)'] = display_df['Lead Time (days)'].round(0).astype(int)
                
                # Status as plain text for cleaner table
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Alerts section
                critical_branches_list = branch_stock_df[branch_stock_df['status'] == 'Critical']
                if len(critical_branches_list) > 0:
                    st.markdown("**Critical alerts**")
                    for _, branch in critical_branches_list.iterrows():
                        st.markdown(f"""
                        <div class="danger-card">
                            <strong>{branch['branch']}</strong> — Critical: {branch['current_stock']:,.0f} units ({branch['days_remaining']:.1f} days left)
                        </div>
                        """, unsafe_allow_html=True)
                
                low_branches_list = branch_stock_df[branch_stock_df['status'] == 'Low']
                if len(low_branches_list) > 0:
                    st.markdown("**Low stock**")
                    for _, branch in low_branches_list.iterrows():
                        st.markdown(f"""
                        <div class="warning-card">
                            <strong>{branch['branch']}</strong> — Low: {branch['current_stock']:,.0f} units ({branch['days_remaining']:.1f} days left)
                        </div>
                        """, unsafe_allow_html=True)
                
                # Stock distribution pie chart
                st.markdown("**Stock distribution**")
                col1, col2 = st.columns(2)
                
                with col1:
                    status_counts = branch_stock_df['status'].value_counts()
                    fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
                    fig_pie.patch.set_facecolor('#0d1117')
                    ax_pie.set_facecolor('#0d1117')
                    colors_pie = {'Adequate': '#22c55e', 'Reorder Soon': '#eab308', 'Low': '#f59e0b', 'Critical': '#ef4444'}
                    pie_colors = [colors_pie.get(s, '#6b7280') for s in status_counts.index]
                    ax_pie.pie(status_counts.values, labels=status_counts.index, autopct='%1.0f%%', 
                              colors=pie_colors, startangle=90, explode=[0.02]*len(status_counts), 
                              textprops={'fontsize': 10, 'color': '#e5e7eb'})
                    ax_pie.set_title('By status', fontsize=12, weight='500', color='#e5e7eb')
                    st.pyplot(fig_pie)
                
                with col2:
                    branch_stock_df_pct = branch_stock_df.copy()
                    branch_stock_df_pct['stock_pct'] = (branch_stock_df_pct['current_stock'] / total_stock * 100).round(1)
                    branch_stock_df_pct = branch_stock_df_pct.sort_values('stock_pct', ascending=False)
                    
                    fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
                    fig_bar.patch.set_facecolor('#0d1117')
                    ax_bar.set_facecolor('#0d1117')
                    ax_bar.barh(branch_stock_df_pct['branch'], branch_stock_df_pct['stock_pct'], 
                               color='#22c55e', alpha=0.75, height=0.6)
                    for spine in ax_bar.spines.values():
                        spine.set_visible(False)
                    ax_bar.tick_params(colors='#9ca3af', labelsize=10)
                    ax_bar.grid(axis='x', alpha=0.15, color='#6b7280')
                    ax_bar.set_title('Share of total stock', fontsize=12, weight='500', color='#e5e7eb')
                    ax_bar.set_xlabel('')
                    st.pyplot(fig_bar)
                
            else:
                st.warning(f"No branch data available for {selected_med}")
    
    # Adjust tab numbers for Historical Trends and Detailed View
    if has_branches:
        hist_tab = tab4
        detail_tab = tab5
    else:
        hist_tab = tab3
        detail_tab = tab4
    
    with hist_tab:
        st.markdown("#### Historical demand")
        
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
        
        # Plot historical trends - dark
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#0d1117')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors='#9ca3af', labelsize=10)
            ax.grid(axis='y', alpha=0.15, color='#6b7280')
            ax.set_axisbelow(True)
        
        axes[0].fill_between(series.index, 0, hist_values, color='#22c55e', alpha=0.4)
        axes[0].plot(series.index, hist_values, color='#22c55e', linewidth=2)
        axes[0].set_title('Demand over time', fontsize=13, weight='500', color='#e5e7eb')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=25)
        
        axes[1].bar(monthly_agg['date'], monthly_agg['y'], width=18, alpha=0.8, color='#22c55e', edgecolor='none')
        axes[1].set_title('Monthly average', fontsize=13, weight='500', color='#e5e7eb')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=25)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Statistics**")
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
            st.markdown("**Recent**")
            recent_data = pd.DataFrame({
                'Date': series.index[-10:].strftime('%Y-%m-%d'),
                'Demand': hist_values[-10:]
            })
            recent_data['Demand'] = recent_data['Demand'].round(0)
            st.dataframe(recent_data, use_container_width=True, hide_index=True)
    
    with detail_tab:
        st.markdown("#### Medication details")
        
        # Medication details
        med_details = med_data.iloc[0] if len(med_data) > 0 else None
        
        if med_details is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Details**")
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
                st.markdown("**Operational**")
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
        st.markdown("**Data preview**")
        st.dataframe(med_data.head(20), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()

