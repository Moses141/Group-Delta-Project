import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import os
warnings.filterwarnings("ignore")

# Model imports
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Design System
# ---------------------------------------------------------------------------
COLORS = {
    "bg":          "#F8F9FB",
    "card":        "#FFFFFF",
    "text":        "#111827",
    "text_sec":    "#6B7280",
    "border":      "#E5E7EB",
    "primary":     "#2563EB",
    "primary_light": "#DBEAFE",
    "success":     "#059669",
    "success_light": "#ECFDF5",
    "warning":     "#D97706",
    "warning_light": "#FFFBEB",
    "danger":      "#DC2626",
    "danger_light": "#FEF2F2",
    "chart_grid":  "#F3F4F6",
    "chart_line":  "#374151",
}

# Lucide-style SVG icons (monochrome stroke)
_ICON_ATTRS = 'xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'

ICONS = {
    "pill": f'<svg {_ICON_ATTRS} width="22" height="22" viewBox="0 0 24 24"><path d="m10.5 1.5 8 8a4.94 4.94 0 1 1-7 7l-8-8a4.94 4.94 0 1 1 7-7Z"/><path d="m8.5 8.5 7 7"/></svg>',
    "bar_chart": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
    "trending_up": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "package": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>',
    "building": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"/><path d="M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2"/><path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2"/><path d="M10 6h4"/><path d="M10 10h4"/><path d="M10 14h4"/><path d="M10 18h4"/></svg>',
    "clock": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "search": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "alert_tri": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "check_circle": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>',
    "info": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "clipboard": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/></svg>',
    "sliders": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/></svg>',
    "activity": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "calendar": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>',
    "hospital": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><path d="M12 6v4"/><path d="M14 14h-4"/><path d="M14 18h-4"/><path d="M14 8h-4"/><path d="M18 12h2a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2h2"/><path d="M18 22V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v18"/></svg>',
    "layers": f'<svg {_ICON_ATTRS} width="18" height="18" viewBox="0 0 24 24"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22.54 12.43-1.96-.89"/><path d="m2.58 12.35 8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.91"/><path d="m2.58 16.35 8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.91"/></svg>',
}

def icon_html(name, size=18, color=None):
    """Return an inline SVG icon string."""
    c = color or COLORS["text_sec"]
    svg = ICONS.get(name, "")
    if not svg:
        return ""
    return f'<span style="display:inline-flex;align-items:center;color:{c};vertical-align:middle;margin-right:6px;">{svg}</span>'

def section_header(icon_name, text, level=3):
    """Render a section header with an icon."""
    tag = f"h{level}"
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:8px;font-family:\'Inter\',sans-serif;'
        f'font-weight:600;color:{COLORS["text"]};margin-top:1.5rem;margin-bottom:1rem;">'
        f'{icon_html(icon_name, color=COLORS["text_sec"])}{text}</{tag}>',
        unsafe_allow_html=True,
    )

def status_dot(color_hex, size=8):
    """Return a small colored dot span."""
    return (
        f'<span style="display:inline-block;width:{size}px;height:{size}px;'
        f'border-radius:50%;background:{color_hex};margin-right:6px;flex-shrink:0;"></span>'
    )

STATUS_PALETTE = {
    "Critical":     COLORS["danger"],
    "Low":          COLORS["warning"],
    "Reorder Soon": "#F59E0B",
    "Adequate":     COLORS["success"],
    "Unknown":      COLORS["text_sec"],
}

# ---------------------------------------------------------------------------
# Matplotlib global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "axes.facecolor": COLORS["card"],
    "figure.facecolor": COLORS["card"],
    "axes.edgecolor": COLORS["border"],
    "axes.labelcolor": COLORS["text"],
    "axes.titleweight": "bold",
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text_sec"],
    "ytick.color": COLORS["text_sec"],
    "grid.color": COLORS["chart_grid"],
    "grid.alpha": 0.6,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.dpi": 110,
})

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pharmacy Demand & Stock Dashboard",
    page_icon="\u25C7",  # clean diamond glyph
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
/* ---- Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stMarkdownContainer"],
[data-testid="stText"], .stTextInput input, .stNumberInput input,
.stSelectbox div, button {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

/* ---- Backgrounds ---- */
[data-testid="stAppViewContainer"] {{
    background-color: {COLORS["bg"]};
}}
[data-testid="stSidebar"] {{
    background-color: {COLORS["card"]};
    border-right: 1px solid {COLORS["border"]};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{
    font-weight: 600;
    font-size: 1rem;
    color: {COLORS["text"]};
    letter-spacing: 0.01em;
}}

/* ---- Main container spacing ---- */
.main .block-container {{
    padding: 2.4rem 3rem 3rem 3rem;
    max-width: 1280px;
}}

/* ---- Headers ---- */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Inter', sans-serif !important;
    color: {COLORS["text"]} !important;
}}
h1 {{ font-weight: 700 !important; }}
h2 {{ font-weight: 600 !important; }}
h3 {{ font-weight: 600 !important; font-size: 1.1rem !important; }}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {{
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}}
[data-testid="stMetricLabel"] {{
    font-weight: 500 !important;
    color: {COLORS["text_sec"]} !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
[data-testid="stMetricValue"] {{
    font-weight: 700 !important;
    color: {COLORS["text"]} !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.78rem !important;
}}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: {COLORS["text_sec"]} !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.7rem 1.2rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {COLORS["primary"]} !important;
    border-bottom: 2px solid {COLORS["primary"]} !important;
    font-weight: 600 !important;
}}

/* ---- Data frames ---- */
[data-testid="stDataFrame"] {{
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    overflow: hidden;
}}

/* ---- Buttons ---- */
.stButton > button {{
    border-radius: 6px;
    font-weight: 500;
    border: 1px solid {COLORS["border"]};
    transition: all 0.15s ease;
}}
.stButton > button:hover {{
    border-color: {COLORS["primary"]};
    color: {COLORS["primary"]};
}}

/* ---- Alert cards ---- */
.alert-card {{
    background: {COLORS["card"]};
    padding: 1.2rem 1.5rem;
    border-radius: 8px;
    border-left: 3px solid {COLORS["border"]};
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 0.85rem;
    font-size: 0.92rem;
    line-height: 1.6;
    color: {COLORS["text"]};
    display: flex;
    align-items: flex-start;
    gap: 10px;
}}
.alert-card.critical {{
    border-left-color: {COLORS["danger"]};
    background: {COLORS["danger_light"]};
}}
.alert-card.warning {{
    border-left-color: {COLORS["warning"]};
    background: {COLORS["warning_light"]};
}}
.alert-card.info {{
    border-left-color: {COLORS["primary"]};
    background: {COLORS["primary_light"]};
}}
.alert-card.success {{
    border-left-color: {COLORS["success"]};
    background: {COLORS["success_light"]};
}}

/* ---- Main header ---- */
.main-header {{
    font-size: 1.65rem;
    font-weight: 700;
    color: {COLORS["text"]};
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.8rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid {COLORS["border"]};
}}

/* ---- Sidebar section divider ---- */
.sidebar-section {{
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLORS["text_sec"]};
    margin-top: 1.8rem;
    margin-bottom: 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid {COLORS["border"]};
}}

/* ---- Info/Success/Warning/Error overrides ---- */
[data-testid="stAlert"] {{
    border-radius: 8px !important;
    border: 1px solid {COLORS["border"]} !important;
    font-size: 0.9rem !important;
}}

/* ---- Selectbox/inputs ---- */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input {{
    border-radius: 6px;
}}

/* ---- Spinner ---- */
.stSpinner > div {{
    border-top-color: {COLORS["primary"]} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the drug supply data"""
    try:
        df = pd.read_csv("uganda_drug_supply_synthetic.csv")
        df['stock_received_date'] = pd.to_datetime(df['stock_received_date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Data file 'uganda_drug_supply_synthetic.csv' not found.")
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
    branch_cols = ['facility_type', 'distribution_region', 'branch', 'facility', 'location']
    for col in branch_cols:
        if col in df.columns:
            return sorted(df[col].unique().tolist()), col
    return [], None

def get_branch_stock_data(df, medication, branch_col):
    """Get stock data for a medication across all branches"""
    if branch_col is None or 'drug_name' not in df.columns:
        return None

    branch_data = []
    for branch in df[branch_col].unique():
        branch_df = df[(df['drug_name'] == medication) & (df[branch_col] == branch)].copy()
        if len(branch_df) > 0:
            branch_df = branch_df.sort_values('stock_received_date', ascending=False)
            latest = branch_df.iloc[0]
            branch_data.append({
                'branch': branch,
                'current_stock': latest.get('initial_stock_units', 0),
                'reorder_level': latest.get('reorder_level', 0),
                'average_demand': latest.get('average_monthly_demand', 0),
                'last_update': latest.get('stock_received_date', None),
                'lead_time': latest.get('lead_time_days', 7),
                'supplier_reliability': latest.get('supplier_reliability_score', 0),
            })

    return pd.DataFrame(branch_data) if branch_data else None

def prepare_time_series(df, medication, target_col='average_monthly_demand', time_col='stock_received_date'):
    """Prepare time series data for a specific medication"""
    df_med = df[df['drug_name'] == medication].copy() if 'drug_name' in df.columns else df.copy()
    df_med = df_med.sort_values(time_col).reset_index(drop=True)

    series = df_med.set_index(time_col)[target_col].resample("W").mean().to_frame("y")
    series['y'] = series['y'].interpolate(limit_direction='both').ffill().bfill()

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

    sarima_order = (1, 1, 1)
    seasonal_order = (1, 0, 1, 52) if len(train_series) >= 60 else (0, 0, 0, 0)

    try:
        model = sm.tsa.SARIMAX(
            train_series,
            order=sarima_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
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
    daily_demand = predicted_demand / 7

    if daily_demand > 0:
        days_remaining = current_stock / daily_demand
    else:
        days_remaining = np.inf

    demand_during_lead = daily_demand * lead_time_days
    safety_stock = demand_during_lead * safety_stock_pct
    reorder_point = demand_during_lead + safety_stock

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
        'daily_demand': daily_demand,
    }


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def _status_bar_color(status):
    mapping = {
        'Critical': COLORS["danger"],
        'Low': COLORS["warning"],
        'Reorder Soon': "#F59E0B",
        'Adequate': COLORS["success"],
    }
    return mapping.get(status, COLORS["text_sec"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Header
    st.markdown(
        f'<div class="main-header">{icon_html("pill", 22, COLORS["primary"])}'
        f'Pharmacy Demand &amp; Stock Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Load data
    df = load_data()
    if df is None:
        return

    # ---- Sidebar ----
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:0.6rem 0 0.4rem 0;">'
        f'{icon_html("sliders", 18, COLORS["text_sec"])}'
        f'<span style="font-weight:600;font-size:1rem;color:{COLORS["text"]};">Dashboard Controls</span></div>',
        unsafe_allow_html=True,
    )

    medications = get_medications(df)
    if not medications:
        st.error("No medications found in data. Please check the data file.")
        return

    selected_med = st.sidebar.selectbox("Select Medication", medications, index=0)

    st.sidebar.markdown(f'<div class="sidebar-section">Forecast Settings</div>', unsafe_allow_html=True)
    forecast_weeks = st.sidebar.slider("Forecast Weeks", 1, 12, 4)

    st.sidebar.markdown(f'<div class="sidebar-section">Stock Monitoring</div>', unsafe_allow_html=True)
    current_stock = st.sidebar.number_input("Current Stock (units)", min_value=0, value=10000, step=100)
    lead_time_days = st.sidebar.number_input("Lead Time (days)", min_value=1, value=7, step=1)
    safety_stock_pct = st.sidebar.slider("Safety Stock (%)", min_value=0.0, max_value=0.5, value=0.2, step=0.05)

    # Get medication data
    med_data = df[df['drug_name'] == selected_med] if 'drug_name' in df.columns else df

    # ---- Model loading ----
    with st.spinner("Preparing data and loading model..."):
        pretrained_data = load_pretrained_model(selected_med)

        if pretrained_data is not None:
            model = pretrained_data['model']
            series = pretrained_data['series']
            apply_log = pretrained_data['apply_log']
            train_size_idx = pretrained_data['train_size_idx']
            st.markdown(
                f'<div class="alert-card success">{icon_html("check_circle", 18, COLORS["success"])}'
                f'Using pre-trained model (trained on {pretrained_data.get("trained_date", "unknown date")})</div>',
                unsafe_allow_html=True,
            )
        else:
            series, apply_log = prepare_time_series(df, selected_med)

            if len(series) < 20:
                st.error(f"Insufficient data for {selected_med}. Need at least 20 data points.")
                return

            model, train_series, train_size_idx = train_sarima_model(series)

            if model is None:
                return

            st.markdown(
                f'<div class="alert-card info">{icon_html("info", 18, COLORS["primary"])}'
                f'Tip: Run <code>python train_models.py</code> to pre-train models for faster loading.</div>',
                unsafe_allow_html=True,
            )

    # ---- KPI Metrics ----
    latest_demand = series['y'].iloc[-1]
    if apply_log:
        latest_demand = np.expm1(latest_demand)

    avg_demand = series['y'].mean()
    if apply_log:
        avg_demand = np.expm1(avg_demand)

    pred_mean, conf_int = forecast_demand(model, forecast_weeks, apply_log)
    if pred_mean is None:
        return

    next_week_demand = pred_mean[0] if len(pred_mean) > 0 else 0

    stock_status = calculate_stock_status(current_stock, next_week_demand, lead_time_days, safety_stock_pct)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Stock", f"{current_stock:,.0f}", f"{stock_status['days_remaining']:.1f} days remaining")

    with col2:
        delta_str = f"\u00B1{abs(conf_int[0][1] - conf_int[0][0]):,.0f}" if conf_int is not None else ""
        st.metric("Next Week Demand", f"{next_week_demand:,.0f}", delta_str)

    with col3:
        st.metric("Stock Status", stock_status['status'], f"Reorder at {stock_status['reorder_point']:,.0f}")

    with col4:
        st.metric("Average Demand", f"{avg_demand:,.0f}", f"Latest: {latest_demand:,.0f}")

    # Stock alert
    if stock_status['status'] in ['Critical', 'Low']:
        level = "critical" if stock_status['status'] == 'Critical' else "warning"
        ic = "alert_tri"
        ic_color = COLORS["danger"] if level == "critical" else COLORS["warning"]
        st.markdown(
            f'<div class="alert-card {level}">'
            f'{icon_html(ic, 18, ic_color)}'
            f'<div><strong>Alert:</strong> Stock level is {stock_status["status"].lower()}. '
            f'Current stock will last approximately {stock_status["days_remaining"]:.1f} days. '
            f'Consider reordering immediately.</div></div>',
            unsafe_allow_html=True,
        )

    # ---- Tabs ----
    branches, branch_col = get_branches(df)
    has_branches = len(branches) > 0 and branch_col is not None

    if has_branches:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Demand Forecast", "Stock Analysis", "Multi-Branch Monitoring",
            "Historical Trends", "Detailed View",
        ])
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Demand Forecast", "Stock Analysis", "Historical Trends", "Detailed View",
        ])

    # ===== TAB 1: Demand Forecast =====
    with tab1:
        section_header("trending_up", "Demand Forecast")

        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=forecast_weeks, freq='W')

        one_year_ago = last_date - timedelta(days=365)
        historical_mask = series.index >= one_year_ago
        historical_dates = series.index[historical_mask]
        historical_values = series['y'].values[historical_mask]
        if apply_log:
            historical_values = np.expm1(historical_values)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(historical_dates, historical_values, color=COLORS["chart_line"], linewidth=1.8, label='Historical Demand')
        ax.plot(forecast_dates, pred_mean, color=COLORS["primary"], linewidth=2, linestyle='--', marker='o', markersize=5, label='Forecasted Demand')

        if conf_int is not None:
            ax.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1],
                            alpha=0.15, color=COLORS["primary"], label='95% Confidence Interval')

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Demand (units)', fontsize=11)
        ax.set_title(f'Demand Forecast \u2014 {selected_med}', fontsize=13, fontweight='bold', pad=12)
        ax.legend(loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        section_header("clipboard", "Forecast Details")
        forecast_df = pd.DataFrame({
            'Week': forecast_dates.strftime('%Y-%m-%d'),
            'Forecasted Demand': pred_mean,
            'Lower Bound (95%)': conf_int[:, 0] if conf_int is not None else pred_mean,
            'Upper Bound (95%)': conf_int[:, 1] if conf_int is not None else pred_mean,
        })
        for c in ['Forecasted Demand', 'Lower Bound (95%)', 'Upper Bound (95%)']:
            forecast_df[c] = forecast_df[c].round(0)
        st.dataframe(forecast_df, use_container_width=True)

    # ===== TAB 2: Stock Analysis =====
    with tab2:
        section_header("package", "Stock Level Analysis")

        daily_demand = stock_status['daily_demand']
        days_projected = min(90, int(stock_status['days_remaining'] * 1.5))
        projection_dates = pd.date_range(start=datetime.now(), periods=days_projected, freq='D')

        projected_stock = []
        stock = current_stock
        for _ in projection_dates:
            if stock > 0:
                stock = max(0, stock - daily_demand)
            projected_stock.append(stock)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(projection_dates, projected_stock, color=COLORS["chart_line"], linewidth=1.8, label='Projected Stock')
        ax.axhline(y=stock_status['reorder_point'], color=COLORS["warning"], linewidth=1.2, linestyle='--', label='Reorder Point')
        ax.axhline(y=stock_status['safety_stock'], color=COLORS["danger"], linewidth=1.2, linestyle='--', label='Safety Stock')
        ax.fill_between(projection_dates, 0, stock_status['safety_stock'], alpha=0.08, color=COLORS["danger"], label='Critical Zone')

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Stock Level (units)', fontsize=11)
        ax.set_title(f'Stock Projection \u2014 {selected_med}', fontsize=13, fontweight='bold', pad=12)
        ax.legend(loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)

        with col1:
            section_header("bar_chart", "Stock Metrics")
            st.write(f"**Current Stock:** {current_stock:,.0f} units")
            st.write(f"**Reorder Point:** {stock_status['reorder_point']:,.0f} units")
            st.write(f"**Safety Stock:** {stock_status['safety_stock']:,.0f} units")

        with col2:
            section_header("clock", "Time Metrics")
            st.write(f"**Days Remaining:** {stock_status['days_remaining']:.1f} days")
            st.write(f"**Lead Time:** {lead_time_days} days")
            st.write(f"**Daily Demand:** {daily_demand:,.1f} units/day")

        with col3:
            section_header("clipboard", "Recommendations")
            if stock_status['status'] == 'Critical':
                st.markdown(
                    f'<div class="alert-card critical">{icon_html("alert_tri", 18, COLORS["danger"])}'
                    f'<strong>Urgent:</strong> Reorder immediately.</div>',
                    unsafe_allow_html=True,
                )
            elif stock_status['status'] == 'Low':
                st.markdown(
                    f'<div class="alert-card warning">{icon_html("alert_tri", 18, COLORS["warning"])}'
                    f'<strong>Action Required:</strong> Reorder within 2\u20133 days.</div>',
                    unsafe_allow_html=True,
                )
            elif stock_status['status'] == 'Reorder Soon':
                st.markdown(
                    f'<div class="alert-card info">{icon_html("info", 18, COLORS["primary"])}'
                    f'<strong>Monitor:</strong> Consider reordering soon.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="alert-card success">{icon_html("check_circle", 18, COLORS["success"])}'
                    f'<strong>Adequate:</strong> Stock levels are healthy.</div>',
                    unsafe_allow_html=True,
                )

            recommended_order = max(0, stock_status['reorder_point'] - current_stock + stock_status['safety_stock'])
            if recommended_order > 0:
                st.write(f"**Recommended Order:** {recommended_order:,.0f} units")

    # ===== TAB 3 (conditional): Multi-Branch Monitoring =====
    if has_branches:
        with tab3:
            section_header("building", "Multi-Branch Stock Monitoring")

            branch_stock_df = get_branch_stock_data(df, selected_med, branch_col)

            if branch_stock_df is not None and len(branch_stock_df) > 0:
                branch_stock_df['daily_demand'] = branch_stock_df['average_demand'] / 30
                branch_stock_df['days_remaining'] = branch_stock_df['current_stock'] / branch_stock_df['daily_demand']
                branch_stock_df['days_remaining'] = branch_stock_df['days_remaining'].replace([np.inf, -np.inf], np.nan)

                def get_stock_status_row(row):
                    days = row['days_remaining']
                    lt = row['lead_time']
                    if pd.isna(days):
                        return "Unknown"
                    elif days < lt:
                        return "Critical"
                    elif days < lt * 1.5:
                        return "Low"
                    elif row['current_stock'] < row['reorder_level']:
                        return "Reorder Soon"
                    return "Adequate"

                branch_stock_df['status'] = branch_stock_df.apply(get_stock_status_row, axis=1)

                total_stock = branch_stock_df['current_stock'].sum()
                total_demand = branch_stock_df['average_demand'].sum()
                critical_branches = len(branch_stock_df[branch_stock_df['status'] == 'Critical'])
                low_branches = len(branch_stock_df[branch_stock_df['status'] == 'Low'])

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Stock", f"{total_stock:,.0f}", "Across all branches")
                with c2:
                    st.metric("Total Demand", f"{total_demand:,.0f}", "Monthly average")
                with c3:
                    st.metric("Critical Branches", critical_branches,
                              f"{low_branches} low stock" if low_branches > 0 else "")
                with c4:
                    avg_days = branch_stock_df['days_remaining'].mean()
                    st.metric("Avg Days Remaining", f"{avg_days:.1f}", "Across branches")

                # Bar charts
                section_header("bar_chart", "Stock Levels by Branch")
                fig, axes = plt.subplots(2, 1, figsize=(14, 9))
                branch_sorted = branch_stock_df.sort_values('current_stock', ascending=True)
                bar_colors = [_status_bar_color(s) for s in branch_sorted['status']]

                axes[0].barh(branch_sorted['branch'], branch_sorted['current_stock'], color=bar_colors, alpha=0.8)
                axes[0].axvline(x=branch_sorted['reorder_level'].mean(), color=COLORS["danger"],
                                linewidth=1, linestyle='--', label='Avg Reorder Level')
                axes[0].set_xlabel('Current Stock (units)', fontsize=11)
                axes[0].set_ylabel('Branch', fontsize=11)
                axes[0].set_title(f'Stock Levels by Branch \u2014 {selected_med}', fontsize=13, fontweight='bold', pad=12)
                axes[0].legend()

                axes[1].barh(branch_sorted['branch'], branch_sorted['days_remaining'], color=bar_colors, alpha=0.8)
                axes[1].axvline(x=branch_sorted['lead_time'].mean(), color=COLORS["warning"],
                                linewidth=1, linestyle='--', label='Avg Lead Time')
                axes[1].set_xlabel('Days of Stock Remaining', fontsize=11)
                axes[1].set_ylabel('Branch', fontsize=11)
                axes[1].set_title('Days of Stock Remaining by Branch', fontsize=13, fontweight='bold', pad=12)
                axes[1].legend()

                plt.tight_layout()
                st.pyplot(fig)

                # Branch details table
                section_header("clipboard", "Branch Details")
                display_df = branch_stock_df[[
                    'branch', 'current_stock', 'reorder_level',
                    'average_demand', 'days_remaining', 'status', 'lead_time',
                ]].copy()
                display_df.columns = [
                    'Branch', 'Current Stock', 'Reorder Level',
                    'Avg Monthly Demand', 'Days Remaining', 'Status', 'Lead Time (days)',
                ]
                display_df['Current Stock'] = display_df['Current Stock'].round(0).astype(int)
                display_df['Reorder Level'] = display_df['Reorder Level'].round(0).astype(int)
                display_df['Avg Monthly Demand'] = display_df['Avg Monthly Demand'].round(0).astype(int)
                display_df['Days Remaining'] = display_df['Days Remaining'].round(1)
                display_df['Lead Time (days)'] = display_df['Lead Time (days)'].round(0).astype(int)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Alerts
                critical_list = branch_stock_df[branch_stock_df['status'] == 'Critical']
                if len(critical_list) > 0:
                    section_header("alert_tri", "Critical Stock Alerts")
                    for _, branch in critical_list.iterrows():
                        st.markdown(
                            f'<div class="alert-card critical">'
                            f'{icon_html("alert_tri", 18, COLORS["danger"])}'
                            f'<div><strong>{branch["branch"]}:</strong> Critical stock level. '
                            f'Current stock: {branch["current_stock"]:,.0f} units '
                            f'({branch["days_remaining"]:.1f} days remaining). '
                            f'Lead time: {branch["lead_time"]} days. '
                            f'<strong>Immediate action required.</strong></div></div>',
                            unsafe_allow_html=True,
                        )

                low_list = branch_stock_df[branch_stock_df['status'] == 'Low']
                if len(low_list) > 0:
                    section_header("alert_tri", "Low Stock Warnings")
                    for _, branch in low_list.iterrows():
                        st.markdown(
                            f'<div class="alert-card warning">'
                            f'{icon_html("alert_tri", 18, COLORS["warning"])}'
                            f'<div><strong>{branch["branch"]}:</strong> Low stock level. '
                            f'Current stock: {branch["current_stock"]:,.0f} units '
                            f'({branch["days_remaining"]:.1f} days remaining). '
                            f'Consider reordering soon.</div></div>',
                            unsafe_allow_html=True,
                        )

                # Distribution charts
                section_header("layers", "Stock Distribution")
                col1, col2 = st.columns(2)

                with col1:
                    status_counts = branch_stock_df['status'].value_counts()
                    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
                    pie_colors = [STATUS_PALETTE.get(s, COLORS["text_sec"]) for s in status_counts.index]
                    wedges, texts, autotexts = ax_pie.pie(
                        status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                        colors=pie_colors, startangle=90,
                        wedgeprops=dict(linewidth=1, edgecolor=COLORS["card"]),
                    )
                    for t in autotexts:
                        t.set_fontsize(10)
                        t.set_color(COLORS["text"])
                    ax_pie.set_title('Branches by Stock Status', fontsize=12, fontweight='bold', pad=12)
                    st.pyplot(fig_pie)

                with col2:
                    bdf_pct = branch_stock_df.copy()
                    bdf_pct['stock_pct'] = (bdf_pct['current_stock'] / total_stock * 100).round(1)
                    bdf_pct = bdf_pct.sort_values('stock_pct', ascending=False)

                    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
                    ax_bar.barh(bdf_pct['branch'], bdf_pct['stock_pct'], color=COLORS["primary"], alpha=0.75)
                    ax_bar.set_xlabel('Percentage of Total Stock (%)', fontsize=11)
                    ax_bar.set_ylabel('Branch', fontsize=11)
                    ax_bar.set_title('Stock Distribution Across Branches', fontsize=12, fontweight='bold', pad=12)
                    st.pyplot(fig_bar)

            else:
                st.warning(f"No branch data available for {selected_med}")

    # ---- Assign remaining tabs ----
    hist_tab = tab4 if has_branches else tab3
    detail_tab = tab5 if has_branches else tab4

    # ===== Historical Trends =====
    with hist_tab:
        section_header("activity", "Historical Demand Trends")

        hist_values = series['y'].values
        if apply_log:
            hist_values = np.expm1(hist_values)

        monthly_data = series.copy()
        monthly_data['y'] = hist_values
        monthly_data['month'] = monthly_data.index.month
        monthly_data['year'] = monthly_data.index.year
        monthly_agg = monthly_data.groupby(['year', 'month'])['y'].mean().reset_index()
        monthly_agg['date'] = pd.to_datetime(monthly_agg[['year', 'month']].assign(day=1))

        fig, axes = plt.subplots(2, 1, figsize=(12, 9))

        axes[0].plot(series.index, hist_values, color=COLORS["chart_line"], linewidth=1.8)
        axes[0].set_xlabel('Date', fontsize=11)
        axes[0].set_ylabel('Demand (units)', fontsize=11)
        axes[0].set_title('Historical Demand Over Time', fontsize=13, fontweight='bold', pad=12)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

        axes[1].bar(monthly_agg['date'], monthly_agg['y'], width=20, alpha=0.75, color=COLORS["primary"])
        axes[1].set_xlabel('Month', fontsize=11)
        axes[1].set_ylabel('Average Demand (units)', fontsize=11)
        axes[1].set_title('Monthly Average Demand', fontsize=13, fontweight='bold', pad=12)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        col1, col2 = st.columns(2)

        with col1:
            section_header("bar_chart", "Demand Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{hist_values.mean():,.0f}",
                    f"{np.median(hist_values):,.0f}",
                    f"{hist_values.std():,.0f}",
                    f"{hist_values.min():,.0f}",
                    f"{hist_values.max():,.0f}",
                ],
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with col2:
            section_header("calendar", "Recent Demand")
            recent_data = pd.DataFrame({
                'Date': series.index[-10:].strftime('%Y-%m-%d'),
                'Demand': hist_values[-10:],
            })
            recent_data['Demand'] = recent_data['Demand'].round(0)
            st.dataframe(recent_data, use_container_width=True, hide_index=True)

    # ===== Detailed View =====
    with detail_tab:
        section_header("search", "Detailed Medication Information")

        med_details = med_data.iloc[0] if len(med_data) > 0 else None

        if med_details is not None:
            col1, col2 = st.columns(2)

            with col1:
                section_header("hospital", "Medication Details")
                detail_fields = {
                    'Drug Name': 'drug_name',
                    'Manufacturer Country': 'manufacturer_country',
                    'License Holder': 'license_holder',
                    'Distribution Region': 'distribution_region',
                    'Facility Type': 'facility_type',
                }
                for label, field in detail_fields.items():
                    if field in med_details:
                        st.write(f"**{label}:** {med_details[field]}")

            with col2:
                section_header("bar_chart", "Operational Metrics")
                metric_fields = {
                    'Reorder Level': 'reorder_level',
                    'Lead Time (days)': 'lead_time_days',
                    'Delivery Frequency (days)': 'delivery_frequency_days',
                    'Supplier Reliability': 'supplier_reliability_score',
                }
                for label, field in metric_fields.items():
                    if field in med_details:
                        value = med_details[field]
                        if isinstance(value, (int, float)):
                            st.write(f"**{label}:** {value:,.0f}" if isinstance(value, float) else f"**{label}:** {value:,}")
                        else:
                            st.write(f"**{label}:** {value}")

        section_header("clipboard", "Raw Data Preview")
        st.dataframe(med_data.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
