"""
MedSupply AI - Uganda
Medicine Supply Chain Dashboard. Simple, clear layout for everyday use.
No icons: text-only navigation and labels.

Structure:
  - Session state holds theme (Light/Dark) and is used by inject_css() and chart helpers.
  - Sidebar renders first and returns the active page; main() then calls the right render_*.
  - CSS is injected with st.markdown(..., unsafe_allow_html=True) so our classes apply.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# ---------------------------------------------------------------------------
# Page config – runs once when the app loads
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedSupply AI - Uganda",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Theme: light or dark (stored in session state so it persists across reruns)
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# ---------------------------------------------------------------------------
# API Integration – fetch live predictions from FastAPI backend
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8090/api"


@st.cache_data(ttl=60)  # Cache for 60 seconds so we don't spam the API
def fetch_predictions():
    """
    Try to pull live predictions from the FastAPI backend.
    Falls back to SAMPLE_FACILITIES if the API is not running.
    """
    try:
        resp = requests.get(f"{API_BASE}/predictions", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data:  # API returned predictions
                return data, True
    except Exception:
        pass
    return None, False


@st.cache_data(ttl=60)
def fetch_predictions_summary():
    """Fetch aggregated risk summary from the API."""
    try:
        resp = requests.get(f"{API_BASE}/predictions/summary", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Shared data – demo/sample data used across pages (replace with real data later)
# ---------------------------------------------------------------------------
SAMPLE_FACILITIES = [
    {"Facility": "Kampala General Hospital", "Drug": "Artemether-Lumefantrine",
     "Current Stock": 6757, "Predicted Demand": 4523, "OOQ Recommended": 5838,
     "Stockout Risk": "Medium", "Action": "Order Recommended"},
    {"Facility": "Mbarara Clinic", "Drug": "Omeprazole Capsules",
     "Current Stock": 1401, "Predicted Demand": 2120, "OOQ Recommended": 0,
     "Stockout Risk": "Low", "Action": "Adequate Stock"},
    {"Facility": "Gulu Health Center", "Drug": "Metformin Tablets",
     "Current Stock": 892, "Predicted Demand": 1543, "OOQ Recommended": 2156,
     "Stockout Risk": "High", "Action": "Urgent Order"},
]

# Options for the facility dropdown on the Facility Analysis page
FACILITIES_LIST = ["Kampala General Hospital", "Mbarara Clinic", "Gulu Health Center", "Jinja Pharmacy"]

# Default date range shown in the header (display only; not used for filtering yet)
DEFAULT_END = datetime.now().date()
DEFAULT_START = DEFAULT_END - timedelta(days=30)


def get_chart_layout(is_dark):
    """
    Build a Plotly layout dict so all charts match the app theme (light/dark).
    Sets font color for title, axes, legend, and hover so graph text is always readable.
    """
    bg = "#1a1d23" if is_dark else "#ffffff"
    grid = "rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.08)"
    # Light theme: dark text so labels are readable on light background
    font_color = "#e4e6eb" if is_dark else "#1e1b4b"
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=font_color),
        margin=dict(t=40, b=40, l=50, r=30),
        xaxis=dict(
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            tickfont=dict(color=font_color),
            title=dict(font=dict(color=font_color)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            tickfont=dict(color=font_color),
            title=dict(font=dict(color=font_color)),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=font_color)),
        hoverlabel=dict(bgcolor=bg, font_color=font_color),
        annotations=[],  # in case we add annotations later
    )


def get_chart_colors(is_dark):
    """Return a list of hex colors for chart series (bars, pie segments). Same palette in both themes."""
    if is_dark:
        return ["#818cf8", "#a78bfa", "#c4b5fd", "#6366f1"]
    return ["#4f46e5", "#6366f1", "#818cf8", "#a5b4fc"]


# ---------------------------------------------------------------------------
# CSS: layout, sidebar, cards, table, theme
# We inject CSS via st.markdown(..., unsafe_allow_html=True) so the app looks
# consistent. Theme variables (sb_bg, card_bg, text_primary, etc.) change per theme.
# ---------------------------------------------------------------------------
def inject_css(theme):
    is_dark = theme == "Dark"
    # Theme color variables (sb=sidebar, main=page background, card=metric/result boxes)
    if is_dark:
        sb_bg = "#111827"
        sb_item_bg = "#374151"
        sb_item_color = "#e5e7eb"
        main_bg = "#0f1116"
        card_bg = "#1a1d23"
        card_border = "1px solid rgba(255,255,255,0.08)"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        table_header_bg = "#374151"
        table_header_color = "#f1f5f9"
    else:
        sb_bg = "#e9e0f7"
        sb_item_bg = "#d4c4f0"
        sb_item_color = "#1e1b4b"
        main_bg = "#f5f3ff"
        card_bg = "#ffffff"
        card_border = "1px solid #c4b5fd"
        text_primary = "#1e1b4b"
        text_secondary = "#5b21b6"
        table_header_bg = "#ddd6fe"
        table_header_color = "#1e1b4b"

    # Shared accent and status colors (same in both themes)
    accent = "#6366f1"
    success = "#22c55e"
    warning = "#eab308"
    danger = "#ef4444"
    highlight_strong = "#c4b5fd" if is_dark else "#6d28d9"

    # Light theme only: extra CSS so Streamlit widgets don’t use white-on-white text.
    # We target data-testid and class names that Streamlit adds to the DOM.
    light_theme_overrides = ""
    if not is_dark:
        dark_text = "#1e1b4b"
        light_theme_overrides = f"""
    /* Force dark text everywhere in main content (light theme) */
    .main .stMarkdown, .main p, .main span, .main div, .main label,
    .main a, .main strong, .main b, .main small, .main li,
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label, [data-testid="stAppViewContainer"] div,
    [data-testid="stAppViewContainer"] a, [data-testid="stAppViewContainer"] strong,
    [data-testid="stAppViewContainer"] li, [data-testid="stAppViewContainer"] small {{
        color: {dark_text} !important;
    }}
    [data-testid="stAppViewContainer"] .stMarkdown p,
    [data-testid="stAppViewContainer"] .stMarkdown div {{
        color: {dark_text} !important;
    }}
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 {{
        color: {dark_text} !important;
    }}
    [data-testid="stDataFrame"], [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] * {{
        color: {dark_text} !important;
    }}
    .main [data-testid="stDataFrame"] td, .main [data-testid="stDataFrame"] th {{
        color: {dark_text} !important;
    }}
    .stSelectbox label, .stSlider label, .stNumberInput label,
    .stSelectbox span, .stSlider span, .stNumberInput span,
    .stSelectbox div, .stNumberInput div {{
        color: {dark_text} !important;
    }}
    .stButton button {{ background: #7c3aed !important; color: white !important; border-radius: 8px; }}
    [data-testid="stAlert"] {{ color: {dark_text} !important; }}
    [data-testid="stAlert"] * {{ color: {dark_text} !important; }}
    .stSuccess, .stWarning, .stInfo {{ color: {dark_text} !important; }}
    .stSuccess *, .stWarning *, .stInfo * {{ color: {dark_text} !important; }}
    [data-testid="stVerticalBlock"] {{ color: {dark_text} !important; }}
    [data-testid="stVerticalBlock"] * {{ color: {dark_text} !important; }}
    .element-container {{ color: {dark_text} !important; }}
    .element-container * {{ color: {dark_text} !important; }}
    section.main .block-container {{ color: {dark_text} !important; }}
    section.main .block-container * {{ color: {dark_text} !important; }}
    /* OOQ Calculator: inputs and values readable; +/- buttons visible */
    .main input {{ color: {dark_text} !important; background-color: #ffffff !important; }}
    .main input::placeholder {{ color: #6b7280 !important; }}
    [data-testid="stNumberInput"] {{
        background: #ffffff !important;
        border-radius: 8px;
        border: 1px solid #c4b5fd;
        overflow: visible !important;
    }}
    [data-testid="stNumberInput"] input {{ color: {dark_text} !important; background: #fff !important; }}
    [data-testid="stNumberInput"] label {{ color: {dark_text} !important; }}
    [data-testid="stNumberInput"] button {{
        background: #e9e0f7 !important;
        color: #1e1b4b !important;
        border: 1px solid #c4b5fd !important;
        visibility: visible !important;
    }}
    [data-testid="stNumberInput"] div[data-baseweb="input"] {{ background: #fff !important; }}
    [data-testid="stSlider"] label {{ color: {dark_text} !important; }}
    [data-testid="stSlider"] [data-baseweb="slider"] {{ color: {dark_text} !important; }}
    .ooq-result .big, .ooq-result .small {{ color: {dark_text} !important; }}
    /* Table: light purple only, no dark */
    [data-testid="stDataFrame"] table {{ background: #fff !important; }}
    [data-testid="stDataFrame"] thead th {{ background: #ddd6fe !important; color: #1e1b4b !important; }}
    [data-testid="stDataFrame"] tbody td {{ background: #fff !important; color: #1e1b4b !important; }}
    [data-testid="stDataFrame"] div {{ background: transparent !important; }}
    /* Selectbox (e.g. facility): light purple, not dark */
    [data-testid="stSelectbox"] div {{ background: #fff !important; color: {dark_text} !important; }}
    [data-testid="stSelectbox"] label {{ color: {dark_text} !important; }}
    [data-testid="stSelectbox"] [data-baseweb="select"] {{
        background: #ffffff !important;
        border: 1px solid #c4b5fd !important;
        color: {dark_text} !important;
    }}
    """

    # Inject one big <style> block; f-string fills in theme variables (e.g. {main_bg}, {text_primary})
    st.markdown(
        f"""
<style>
    /* Base */
    .stApp {{ background: {main_bg}; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }}

    /* Hide Streamlit branding for cleaner look */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {sb_bg};
        border-right: 1px solid rgba(0,0,0,0.06);
    }}
    [data-testid="stSidebar"] .stMarkdown {{
        color: {text_primary};
        font-weight: 600;
        font-size: 1.15rem;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] {{ background: transparent; }}
    [data-testid="stSidebar"] label {{
        color: {sb_item_color} !important;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin: 2px 0;
    }}
    [data-testid="stSidebar"] label:hover {{ background: {sb_item_bg}; }}
    [data-testid="stSidebar"] label[data-checked="true"] {{
        background: {accent} !important;
        color: white !important;
    }}
    [data-testid="stSidebar"] .stRadio > div {{ flex-direction: column; gap: 2px; }}

    /* Main content header – larger, clearer */
    .main-title {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 0.25rem;
    }}
    .main-subtitle {{
        font-size: 1rem;
        color: {text_secondary};
        margin-bottom: 1.5rem;
    }}

    /* Date range text (no calendar) */
    .date-range {{
        font-size: 1rem;
        color: {text_secondary};
        padding: 0.5rem 0;
    }}

    /* Metric cards */
    .metric-card {{
        background: {card_bg};
        border: {card_border};
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .metric-card .label {{
        font-size: 0.85rem;
        color: {text_secondary};
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 0.35rem;
    }}
    .metric-card .value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {text_primary};
        line-height: 1.2;
    }}
    .metric-card .delta {{ font-size: 0.9rem; margin-top: 0.35rem; }}
    .metric-card .delta.up {{ color: {success}; }}
    .metric-card .delta.down {{ color: {danger}; }}

    /* Highlight box – light purple tint in light theme */
    .highlight-box {{
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.35);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: {text_primary};
    }}
    .highlight-box strong {{ color: {highlight_strong}; }}

    /* Section titles – bigger, no dark box */
    .section-title {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {text_primary};
        margin: 1.25rem 0 0.75rem 0;
        background: transparent !important;
    }}

    /* Table styling – light purple in light theme, no dark boxes */
    .dataframe, [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: {card_border};
        background: {card_bg} !important;
    }}
    .dataframe th, .dataframe thead th,
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] thead th {{
        background: {table_header_bg} !important;
        color: {table_header_color} !important;
        font-weight: 600;
        padding: 0.75rem 1rem !important;
    }}
    .dataframe td, .dataframe tbody td,
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] tbody td {{
        padding: 0.65rem 1rem !important;
        color: {text_primary};
        background: {card_bg} !important;
    }}
    .status-low {{ color: {success}; font-weight: 500; }}
    .status-medium {{ color: {warning}; font-weight: 500; }}
    .status-high {{ color: {danger}; font-weight: 500; }}

    /* OOQ result box – high contrast so text is always readable */
    .ooq-result {{
        background: {card_bg};
        border: {card_border};
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
    }}
    .ooq-result .big {{ font-size: 2rem; font-weight: 700; color: {text_primary}; }}
    .ooq-result .small {{ font-size: 0.9rem; color: {text_secondary}; margin-top: 0.5rem; }}

    .theme-row {{ display: flex; gap: 0.5rem; align-items: center; margin-top: 1rem; }}
    .theme-row span {{ color: {text_secondary}; font-size: 0.9rem; }}
    {light_theme_overrides}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: brand, navigation, theme
# The sidebar is rendered first; the returned page name drives which main content we show.
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown("## MedSupply AI")
    st.sidebar.markdown("Uganda · Medicine Supply")
    st.sidebar.markdown("---")

    # Radio acts as a menu; key="nav" lets Streamlit remember the selection across reruns
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Facility Analysis", "Model Performance", "OOQ Calculator", "Pipeline Status"],
        label_visibility="collapsed",
        key="nav",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Theme**")
    # Keep the radio in sync with session state (e.g. after user switches theme)
    theme_index = 1 if st.session_state.theme == "Dark" else 0
    new_theme = st.sidebar.radio(
        "Light or Dark", ["Light", "Dark"], index=theme_index, label_visibility="collapsed", key="theme_radio"
    )
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()  # Rerun so inject_css and charts pick up the new theme

    st.sidebar.markdown("---")
    st.sidebar.markdown("Need help? Contact support from your facility.")

    return page


# ---------------------------------------------------------------------------
# Main header (title + date range)
# Shown at the top of every page. Date range is display-only (no date picker widget).
# ---------------------------------------------------------------------------
def render_header(page_name):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f'<p class="main-title">{page_name}</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="main-subtitle">MedSupply AI · Machine learning for medicine supply and stock decisions</p>',
            unsafe_allow_html=True,
        )
    with c2:
        start_str = DEFAULT_START.strftime("%d %b %Y")
        end_str = DEFAULT_END.strftime("%d %b %Y")
        st.markdown(
            f'<p class="date-range">{start_str} – {end_str}</p>',
            unsafe_allow_html=True,
        )
    return DEFAULT_START, DEFAULT_END


# ---------------------------------------------------------------------------
# Dashboard page – overview metrics, facility table, and two charts
# ---------------------------------------------------------------------------
def render_dashboard():
    inject_css(st.session_state.theme)
    is_dark = st.session_state.theme == "Dark"
    render_header("Dashboard")

    # Top row: four metric cards (label, big value, small delta line)
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Demand forecast accuracy", "99.94%", "Up 0.06% MAPE", "up"),
        ("OOQ prediction error", "0.07%", "Target was 7.3%", "up"),
        ("Stockout prediction F1", "0.349", "Up 14.4% improved", "up"),
        ("Facilities analyzed", "15,000", "Across Uganda", None),
    ]
    for col, (label, value, delta, direction) in zip([col1, col2, col3, col4], metrics):
        with col:
            # direction "up" or "down" adds a CSS class for green/red delta text; None = no class
            delta_cls = f" delta {direction}" if direction else ""
            # Build one metric card as HTML so we can use our CSS classes (metric-card, .label, .value, .delta)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                f'<div class="delta{delta_cls}">{delta}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Target achieved highlight
    st.markdown(
        '<div class="highlight-box">'
        "<strong>Primary target achieved:</strong> OOQ prediction MPE of 0.07%, "
        "about 104 times better than the 7.3% target.</div>",
        unsafe_allow_html=True,
    )

    # Section: Real-time facility predictions
    # -----------------------------------------------------------------------
    # INTEGRATION POINT: Try loading live predictions from the API.
    # If the FastAPI backend is running, we show real ML predictions.
    # Otherwise we gracefully fall back to the SAMPLE_FACILITIES demo data.
    # -----------------------------------------------------------------------
    live_data, api_available = fetch_predictions()

    if api_available and live_data:
        st.markdown(
            '<p class="section-title">Real-time facility predictions (live from pipeline)</p>',
            unsafe_allow_html=True,
        )
        df_pred = pd.DataFrame(live_data)
        display_cols = {
            "drug_name": "Drug",
            "distribution_region": "Region",
            "current_stock": "Current Stock",
            "predicted_demand": "Predicted Demand",
            "recommended_order_qty": "OOQ Recommended",
            "stockout_risk_level": "Stockout Risk",
        }
        # Only show columns that exist
        available_display = {k: v for k, v in display_cols.items() if k in df_pred.columns}
        df_show = df_pred[list(available_display.keys())].rename(columns=available_display)

        # Round numeric columns for readability
        for c in ["Current Stock", "Predicted Demand", "OOQ Recommended"]:
            if c in df_show.columns:
                df_show[c] = df_show[c].round(0).astype(int)

        st.dataframe(df_show.head(20), use_container_width=True, hide_index=True)

        # Build chart data from live predictions
        chart_names = df_show["Drug"].head(6).tolist() if "Drug" in df_show.columns else []
        chart_stock = df_show["Current Stock"].head(6).tolist() if "Current Stock" in df_show.columns else []
        chart_demand = df_show["Predicted Demand"].head(6).tolist() if "Predicted Demand" in df_show.columns else []
        chart_risks = df_pred["stockout_risk_level"].tolist() if "stockout_risk_level" in df_pred.columns else []
    else:
        st.markdown(
            '<p class="section-title">Real-time facility predictions</p>',
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(SAMPLE_FACILITIES)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if not api_available:
            st.caption("Showing demo data. Start the API server for live predictions: uvicorn api.main:app")

        # Chart data from sample facilities
        chart_names = [f["Facility"] for f in SAMPLE_FACILITIES]
        chart_stock = [f["Current Stock"] for f in SAMPLE_FACILITIES]
        chart_demand = [f["Predicted Demand"] for f in SAMPLE_FACILITIES]
        chart_risks = [f["Stockout Risk"] for f in SAMPLE_FACILITIES]

    # Two charts side by side: bar chart (stock vs demand) and donut (risk distribution)
    col_left, col_right = st.columns(2)
    layout = get_chart_layout(is_dark)
    colors = get_chart_colors(is_dark)

    with col_left:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Current stock",
                x=chart_names,
                y=chart_stock,
                marker_color=colors[0],
            )
        )
        fig.add_trace(
            go.Bar(
                name="Predicted demand",
                x=chart_names,
                y=chart_demand,
                marker_color=colors[1],
            )
        )
        # fc = font color from layout so title/axes stay readable in light/dark theme
        fc = layout["font"]["color"]
        fig.update_layout(
            **layout,
            title=dict(text="Current stock vs predicted demand", font=dict(size=14, color=fc)),
            barmode="group",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Donut: count how many facilities are Low/Medium/High risk, then show as pie
        risk_counts = pd.Series(chart_risks).value_counts()
        fc = layout["font"]["color"]
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=risk_counts.index.tolist(),
                    values=risk_counts.values.tolist(),
                    hole=0.55,
                    marker_colors=colors[: len(risk_counts)],
                    textinfo="label+percent",
                    textposition="outside",
                    outsidetextfont=dict(color=fc),
                    insidetextfont=dict(color=fc),
                )
            ]
        )
        fig.update_layout(
            **layout,
            title=dict(text="Stockout risk distribution", font=dict(size=14, color=fc)),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Facility Analysis page – one facility’s inventory and recommendations
# ---------------------------------------------------------------------------
def render_facility_analysis():
    inject_css(st.session_state.theme)
    render_header("Facility analysis")

    selected = st.selectbox("Select facility", FACILITIES_LIST, label_visibility="visible")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Inventory status</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-card">'
            '<div class="label">Current stock level</div>'
            '<div class="value">6,757 units</div>'
            '<div class="delta down">Down 12% from last month</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-card">'
            '<div class="label">Weekly consumption</div>'
            '<div class="value">1,130 units</div>'
            '<div class="delta up">Up 5% trend</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-card">'
            '<div class="label">Stockout probability</div>'
            '<div class="value">35.6%</div>'
            '<div class="delta">Medium risk</div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<p class="section-title">Recommendations</p>', unsafe_allow_html=True)
        st.success("Optimal order quantity: 5,838 units.")
        st.warning("Monitor Artemether-Lumefantrine (high demand season).")
        st.info("Suggestion: Use FEFO to cut expiries by about 15%.")


# ---------------------------------------------------------------------------
# Model Performance page – three bar charts (MAPE, F1, MPE)
# ---------------------------------------------------------------------------
def render_model_performance():
    inject_css(st.session_state.theme)
    is_dark = st.session_state.theme == "Dark"
    render_header("Model performance")

    layout = get_chart_layout(is_dark)
    colors = get_chart_colors(is_dark)

    col1, col2, col3 = st.columns(3)

    with col1:
        fc = layout["font"]["color"]
        fig = go.Figure(
            go.Bar(
                x=["Random Forest", "XGBoost"],
                y=[0.0006, 0.0102],
                marker_color=[colors[0], colors[1]],
                text=["0.06%", "1.02%"],
                textposition="outside",
                textfont=dict(color=fc),
            )
        )
        fig.update_layout(
            **layout,
            title=dict(text="Demand forecasting MAPE", font=dict(size=14, color=fc)),
            xaxis_title="Model",
            yaxis_title="MAPE",
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fc = layout["font"]["color"]
        fig = go.Figure(
            go.Bar(
                x=["Before", "After"],
                y=[0.305, 0.349],
                marker_color=[colors[2], colors[0]],
                text=["0.305", "0.349"],
                textposition="outside",
                textfont=dict(color=fc),
            )
        )
        fig.update_layout(
            **layout,
            title=dict(text="Stockout prediction F1 improvement", font=dict(size=14, color=fc)),
            xaxis_title="",
            yaxis_title="F1 score",
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fc = layout["font"]["color"]
        fig = go.Figure(
            go.Bar(
                x=["Target", "Achieved"],
                y=[7.3, 0.07],
                marker_color=[colors[2], colors[0]],
                text=["7.3%", "0.07%"],
                textposition="outside",
                textfont=dict(color=fc),
            )
        )
        fig.update_layout(
            **layout,
            title=dict(text="OOQ prediction MPE", font=dict(size=14, color=fc)),
            xaxis_title="",
            yaxis_title="MPE %",
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="highlight-box">'
        "Random Forest gives the best demand accuracy (0.06% MAPE). "
        "OOQ error is 0.07%, well below the 7.3% target.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# OOQ Calculator page – inputs and simple OOQ formula
# ---------------------------------------------------------------------------
def render_ooq_calculator():
    inject_css(st.session_state.theme)
    is_dark = st.session_state.theme == "Dark"
    render_header("Optimal Order Quantity Calculator")

    st.markdown(
        "Enter your current stock on hand, weekly demand, and lead time. "
        "The calculator will recommend how much to order and the safety stock needed."
    )

    col1, col2 = st.columns(2)
    with col1:
        current_stock = st.number_input("Current stock on hand (units)", min_value=0, value=1000, key="ooq_stock")
        weekly_demand = st.number_input("Average weekly demand (units)", min_value=0, value=500, key="ooq_demand")
        lead_time = st.number_input("Lead time (days)", min_value=1, value=21, key="ooq_lead")
    with col2:
        service_level = st.slider("Service level target (e.g. 0.95 = 95%)", 0.80, 0.99, 0.95, key="ooq_sl")
        demand_volatility = st.slider("Demand volatility (0.1 = low, 0.5 = high)", 0.1, 0.5, 0.2, key="ooq_vol")

    if st.button("Calculate recommended order quantity"):
        lead_time_weeks = lead_time / 7
        # Safety stock: z * (demand * volatility) * sqrt(lead_time_weeks); 1.65 ≈ 95% service level
        safety_stock = 1.65 * (weekly_demand * demand_volatility) * np.sqrt(lead_time_weeks)
        # OOQ = demand over lead time + safety stock - current stock (never negative)
        ooq = max(0, (weekly_demand * lead_time_weeks) + safety_stock - current_stock)

        st.markdown(
            f'<div class="ooq-result">'
            f'<div class="big">{ooq:.0f} units</div>'
            f'<div class="small">Recommended order quantity (OOQ)</div>'
            f'<div class="small" style="margin-top:0.75rem">Safety stock: {safety_stock:.0f} units</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Stacked bar: total demand over lead time + safety stock (visual breakdown of the order)
        layout = get_chart_layout(is_dark)
        colors = get_chart_colors(is_dark)
        period_demand = weekly_demand * lead_time_weeks
        fig = go.Figure(
            data=[
                go.Bar(name="Demand over lead time", x=["Quantity"], y=[period_demand], marker_color=colors[0]),
                go.Bar(name="Safety stock", x=["Quantity"], y=[safety_stock], marker_color=colors[1]),
            ]
        )
        fc = layout["font"]["color"]
        fig.update_layout(
            **layout,
            barmode="stack",
            height=220,
            title=dict(text="Order composition (demand over lead time + safety stock)", font=dict(size=14, color=fc)),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# App entry – sidebar decides which page to show; only that page’s content is rendered
# ---------------------------------------------------------------------------
def render_pipeline_status():
    """
    Pipeline Status page – shows data pipeline health, lets operators
    trigger runs, view logs, and check model versions.

    INTEGRATION POINT: All data comes from the FastAPI /api/* endpoints.
    Falls back to a helpful message if the API is not reachable.
    """
    inject_css(st.session_state.theme)
    render_header("Pipeline Status")

    api_up = False
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=2)
        api_up = resp.status_code == 200
    except Exception:
        pass

    if not api_up:
        st.warning(
            "API server is not running. Start it with:\n\n"
            "```\nuvicorn api.main:app --host 0.0.0.0 --port 8000\n```"
        )
        return

    st.success("API server is online")

    # --- Data statistics ---
    st.markdown('<p class="section-title">Data Volume</p>', unsafe_allow_html=True)
    try:
        stats = requests.get(f"{API_BASE}/data/stats", timeout=3).json()
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, key) in zip(
            [c1, c2, c3, c4],
            [
                ("Raw rows", "raw_stock_rows"),
                ("Cleaned rows", "cleaned_rows"),
                ("Weekly aggregated", "weekly_aggregated_rows"),
                ("Cached predictions", "cached_predictions"),
            ],
        ):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{label}</div>'
                    f'<div class="value">{stats.get(key, 0):,}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
    except Exception:
        st.error("Could not fetch data stats.")

    # --- Active models ---
    st.markdown('<p class="section-title">Active Models</p>', unsafe_allow_html=True)
    try:
        models = requests.get(f"{API_BASE}/models/active", timeout=3).json()
        for name, version in models.items():
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{name}</div>'
                f'<div class="value">{version}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.info("No trained models yet. Run the pipeline to train.")

    # --- Pipeline controls ---
    st.markdown('<p class="section-title">Pipeline Controls</p>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Run Full Pipeline"):
            with st.spinner("Running pipeline..."):
                try:
                    r = requests.post(f"{API_BASE}/pipeline/run?force_retrain=false", timeout=300)
                    data = r.json()
                    if r.status_code == 200:
                        stages = data.get("result", {}).get("stages", {})
                        summary_lines = []
                        for stage_name, stage_info in stages.items():
                            status = stage_info.get("status", "unknown")
                            rows = stage_info.get("rows", "")
                            icon = "[OK]" if status == "success" else "[FAIL]"
                            row_text = f" — {rows:,} rows" if isinstance(rows, int) else ""
                            summary_lines.append(f"{icon} **{stage_name.replace('_', ' ').title()}**: {status}{row_text}")
                        st.success("Pipeline completed successfully!")
                        for line in summary_lines:
                            st.markdown(line)
                    else:
                        st.error(f"Pipeline failed: {data.get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
    with col_b:
        if st.button("Force Retrain Models"):
            with st.spinner("Retraining..."):
                try:
                    r = requests.post(f"{API_BASE}/pipeline/retrain", timeout=300)
                    data = r.json()
                    if r.status_code == 200:
                        st.success("Models retrained successfully!")
                        result = data.get("result", {})
                        if isinstance(result, dict):
                            model_info = result.get("model_name", "")
                            if model_info:
                                st.markdown(f"**Model**: {model_info}")
                            metrics = result.get("metrics", {})
                            if isinstance(metrics, dict):
                                for k, v in metrics.items():
                                    st.markdown(f"**{k}**: {v:.4f}" if isinstance(v, float) else f"**{k}**: {v}")
                    else:
                        st.error(f"Retrain failed: {data.get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Retrain error: {e}")
    with col_c:
        if st.button("Ingest New Data"):
            with st.spinner("Ingesting..."):
                try:
                    r = requests.post(f"{API_BASE}/pipeline/ingest", timeout=60)
                    data = r.json()
                    if r.status_code == 200:
                        rows = data.get("rows_ingested", 0)
                        st.success(f"Data ingestion complete! {rows:,} rows ingested.")
                    else:
                        st.error(f"Ingestion failed: {data.get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Ingestion error: {e}")

    # --- Recent pipeline runs ---
    st.markdown('<p class="section-title">Recent Pipeline Runs</p>', unsafe_allow_html=True)
    try:
        history = requests.get(f"{API_BASE}/pipeline/history?limit=10", timeout=3).json()
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        else:
            st.info("No pipeline runs recorded yet.")
    except Exception:
        st.info("No pipeline history available.")


def main():
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Facility Analysis":
        render_facility_analysis()
    elif page == "Model Performance":
        render_model_performance()
    elif page == "Pipeline Status":
        render_pipeline_status()
    else:
        render_ooq_calculator()

    st.markdown("---")
    st.markdown(
        "**MedSupply AI** · Uganda pharmaceutical supply chain · Final year project 2024"
    )


if __name__ == "__main__":
    main()  # Run: streamlit run medsupply_dashboard.py
