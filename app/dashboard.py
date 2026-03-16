"""
Pharmacy Dashboard — Pharmacist-focused view for reorder decisions,
demand trends, stock context, and procurement planning.

Run instructions:
  1. Generate outputs first: run notebooks 01 → 05 so that
     outputs/monthly_demand.csv and outputs/next_3_month_forecast.csv exist.
  2. Install: pip install -r app/requirements.txt
  3. From pharmacy_forecasting/: streamlit run app/dashboard.py
     Or from app/: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from io import StringIO

from app.processing import refresh_all
from app.validation import (
    validate_opening_stock_upload,
    validate_sales_upload,
    validate_stock_receipts_upload,
)

# ---------------------------------------------------------------------------
# Visual design helpers (no impact on calculations)
# ---------------------------------------------------------------------------
def inject_global_css():
    st.markdown(
        """
<style>
  /* Layout */
  .block-container { padding-top: 3rem; padding-bottom: 2.0rem; max-width: 1200px; }
  [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }

  /* Header */
  .pf-header {
    padding: 1.1rem 1.2rem;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    background: rgba(255,255,255,0.03);
    margin-bottom: 1.0rem;
  }
  .pf-title { font-size: 2.0rem; font-weight: 800; line-height: 1.1; margin: 0; }
  .pf-subtitle { font-size: 0.95rem; opacity: 0.85; margin-top: 0.35rem; }
  .pf-divider { margin-top: 0.9rem; border-top: 1px solid rgba(255,255,255,0.10); }

  /* Cards */
  .pf-card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.9rem 1.0rem;
    background: rgba(255,255,255,0.03);
    height: 100%;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
  }
  .pf-card-label { font-size: 0.9rem; opacity: 0.82; margin: 0 0 0.30rem 0; letter-spacing: 0.02em; }
  .pf-card-value { font-size: 1.55rem; font-weight: 800; margin: 0; }
  .pf-card-help { font-size: 0.78rem; opacity: 0.70; margin-top: 0.35rem; }

  /* Badges / pills */
  .pf-badge {
    display: inline-block;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.10);
    line-height: 1.1;
    white-space: nowrap;
  }
  .pf-high { background: rgba(255, 77, 79, 0.18); color: #ff6b6b; border-color: rgba(255, 77, 79, 0.28); }
  .pf-med  { background: rgba(255, 173, 51, 0.16); color: #ffb020; border-color: rgba(255, 173, 51, 0.28); }
  .pf-low  { background: rgba(46, 204, 113, 0.14); color: #2ecc71; border-color: rgba(46, 204, 113, 0.25); }
  .pf-neutral { background: rgba(99, 110, 114, 0.18); color: rgba(255,255,255,0.80); }

  /* Section titles */
  .pf-section-title { font-size: 1.10rem; font-weight: 750; margin: 0.25rem 0 0.2rem 0; }
  .pf-section-help { font-size: 0.86rem; opacity: 0.78; margin: 0 0 0.65rem 0; }

  /* Table tweaks */
  .stDataFrame { border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; overflow: hidden; }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_product_header():
    st.markdown(
        """
<div class="pf-header">
  <div class="pf-title">Pharmacy Dashboard</div>
  <div class="pf-subtitle">Reorder suggestions, demand trends, and procurement planning</div>
  <div class="pf-divider"></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_last_refresh_chip():
    ts = load_last_refresh_timestamp()
    if ts:
        st.caption(f"Last refresh (UTC): {ts}")


def format_int(n):
    try:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "—"
        return f"{int(round(float(n))):,}"
    except Exception:
        return "—"


def format_float(n, decimals=1):
    try:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "—"
        return f"{float(n):,.{decimals}f}"
    except Exception:
        return "—"


def badge_html(label: str, level: str):
    level = (level or "").upper()
    css = "pf-neutral"
    if level == "HIGH" or level == "CRITICAL":
        css = "pf-high"
    elif level == "MEDIUM" or level == "LOW":
        css = "pf-med"
    elif level == "SAFE":
        css = "pf-low"
    elif level == "LOW_RISK":
        css = "pf-low"
    return f'<span class="pf-badge {css}">{label}</span>'


def render_card(label: str, value: str, icon: str = "", help_text: str = ""):
    icon_html = f"{icon} " if icon else ""
    help_html = f'<div class="pf-card-help">{help_text}</div>' if help_text else ""
    st.markdown(
        f"""
<div class="pf-card">
  <div class="pf-card-label">{icon_html}{label}</div>
  <div class="pf-card-value">{value}</div>
  {help_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, help_text: str):
    st.markdown(f'<div class="pf-section-title">{title}</div>', unsafe_allow_html=True)
    if help_text:
        st.markdown(f'<div class="pf-section-help">{help_text}</div>', unsafe_allow_html=True)


def style_stockout_table(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    view = df.copy()
    styler = view.style
    # Right-align numeric-ish columns
    right_cols = [c for c in view.columns if c in ("Current stock", "Forecast (next month)")]
    if right_cols:
        styler = styler.set_properties(subset=right_cols, **{"text-align": "right"})
    # Row highlight for HIGH risk (uses original df)
    if "Stockout risk" in view.columns:
        def _row_bg(row):
            risk = str(row.get("Stockout risk", "")).upper()
            if risk == "HIGH":
                return ["background-color: rgba(255, 77, 79, 0.10)"] * len(row)
            if risk == "MEDIUM":
                return ["background-color: rgba(255, 173, 51, 0.06)"] * len(row)
            return [""] * len(row)
        styler = styler.apply(_row_bg, axis=1)
        def _risk_text_color(val):
            v = str(val).upper()
            if v == "HIGH":
                return "color: #ff6b6b; font-weight: 800;"
            if v == "MEDIUM":
                return "color: #ffb020; font-weight: 800;"
            if v == "LOW":
                return "color: #2ecc71; font-weight: 800;"
            return ""
        styler = styler.applymap(_risk_text_color, subset=["Stockout risk"])
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("font-weight", "700"), ("text-align", "left")]},
            {"selector": "td", "props": [("padding", "10px 10px")]},
        ]
    )
    return styler


def style_expiry_table(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    view = df.copy()
    styler = view.style
    right_cols = [c for c in view.columns if c in ("Days to expiry",)]
    if right_cols:
        styler = styler.set_properties(subset=right_cols, **{"text-align": "right"})
    if "Expiry risk" in view.columns:
        def _row_bg(row):
            risk = str(row.get("Expiry risk", "")).upper()
            if risk == "HIGH":
                return ["background-color: rgba(255, 77, 79, 0.10)"] * len(row)
            if risk == "MEDIUM":
                return ["background-color: rgba(255, 173, 51, 0.06)"] * len(row)
            return [""] * len(row)
        styler = styler.apply(_row_bg, axis=1)
        def _risk_text_color(val):
            v = str(val).upper()
            if v == "HIGH":
                return "color: #ff6b6b; font-weight: 800;"
            if v == "MEDIUM":
                return "color: #ffb020; font-weight: 800;"
            if v == "LOW":
                return "color: #2ecc71; font-weight: 800;"
            return ""
        styler = styler.applymap(_risk_text_color, subset=["Expiry risk"])
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("font-weight", "700"), ("text-align", "left")]},
            {"selector": "td", "props": [("padding", "10px 10px")]},
        ]
    )
    return styler


def style_reorder_table(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    view = df.copy()
    styler = view.style
    # Emphasize Suggested reorder column
    if "Suggested reorder (units)" in view.columns:
        styler = styler.set_properties(
            subset=["Suggested reorder (units)"], **{"font-weight": "800"}
        )
    # Right-align numeric columns
    num_cols = [c for c in view.columns if c in ("Current stock", "Avg demand (last 3 months)", "Forecast (next month)", "Suggested reorder (units)")]
    if num_cols:
        styler = styler.set_properties(subset=num_cols, **{"text-align": "right"})
    # Slight highlight for large reorder values
    if "Suggested reorder (units)" in df.columns:
        def _reorder_bg(row):
            try:
                v = float(row.get("Suggested reorder (units)", 0))
            except Exception:
                v = 0
            if v >= 1000:
                return ["background-color: rgba(46, 204, 113, 0.06)"] * len(row)
            return [""] * len(row)
        styler = styler.apply(_reorder_bg, axis=1)
    if "Priority" in view.columns:
        def _prio_color(val):
            v = str(val).upper()
            if v == "HIGH":
                return "color: #ff6b6b; font-weight: 800;"
            if v == "MEDIUM":
                return "color: #ffb020; font-weight: 800;"
            if v == "LOW":
                return "color: #2ecc71; font-weight: 800;"
            return ""
        styler = styler.applymap(_prio_color, subset=["Priority"])
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("font-weight", "700"), ("text-align", "left")]},
            {"selector": "td", "props": [("padding", "10px 10px")]},
        ]
    )
    return styler


def style_days_supply_table(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    view = df.copy()
    styler = view.style
    num_cols = [c for c in view.columns if c in ("Current stock", "Avg daily demand", "Days of supply")]
    if num_cols:
        styler = styler.set_properties(subset=num_cols, **{"text-align": "right"})
    if "Status" in view.columns:
        def _status_color(val):
            txt = str(val).upper()
            if "CRITICAL" in txt:
                return "color: #ff6b6b; font-weight: 800;"
            if "LOW" in txt and "CRITICAL" not in txt:
                return "color: #ffb020; font-weight: 800;"
            if "SAFE" in txt:
                return "color: #2ecc71; font-weight: 800;"
            return ""
        styler = styler.applymap(_status_color, subset=["Status"])
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("font-weight", "700"), ("text-align", "left")]},
            {"selector": "td", "props": [("padding", "10px 10px")]},
        ]
    )
    return styler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def get_base_path():
    cwd = Path(__file__).resolve().parent
    return cwd.parent if cwd.name == "app" else cwd


BASE = get_base_path()
DATA_DIR = BASE / "data"
OUTPUTS_DIR = BASE / "outputs"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_monthly_demand():
    path = OUTPUTS_DIR / "monthly_demand.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df


@st.cache_data
def load_forecast():
    path = OUTPUTS_DIR / "next_3_month_forecast.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["forecast_month"] = pd.to_datetime(df["forecast_month"])
    return df


@st.cache_data
def load_stock_status():
    """Processed stock balance per drug from Notebook 1."""
    path = OUTPUTS_DIR / "stock_status.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Basic cleaning
    if "drug_id" in df.columns:
        df["drug_id"] = df["drug_id"].astype(str)
    for col in ["opening_stock_units", "total_received", "total_dispensed", "current_stock"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data
def load_last_refresh_timestamp():
    path = OUTPUTS_DIR / "last_refresh.json"
    if not path.exists():
        return None
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("last_refresh_utc")
    except Exception:
        return None


@st.cache_data
def load_stock_receipts():
    path = DATA_DIR / "stock_receipts.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["stock_received_date"] = pd.to_datetime(df["stock_received_date"])
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    return df


@st.cache_data
def load_sales_transactions():
    path = DATA_DIR / "sales_transactions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@st.cache_data
def load_drug_names():
    path = DATA_DIR / "sales_transactions.csv"
    if not path.exists():
        return pd.DataFrame(columns=["drug_id", "drug_name", "category"])
    df = pd.read_csv(path)
    return df[["drug_id", "drug_name", "category"]].drop_duplicates("drug_id").reset_index(drop=True)


def build_drug_lookup(sales_drugs):
    id_to_name = {}
    id_to_cat = {}
    if sales_drugs.empty:
        return id_to_name, id_to_cat
    for _, r in sales_drugs.iterrows():
        id_to_name[r["drug_id"]] = r["drug_name"]
        id_to_cat[r["drug_id"]] = r.get("category", "")
    return id_to_name, id_to_cat


# ---------------------------------------------------------------------------
# Current stock: total received - total dispensed, clamped to 0
# ---------------------------------------------------------------------------
@st.cache_data
def compute_total_received_by_drug(stock_df):
    if stock_df.empty or "quantity_received" not in stock_df.columns:
        return pd.Series(dtype=float)
    return stock_df.groupby("drug_id")["quantity_received"].sum()


@st.cache_data
def compute_total_dispensed_by_drug(sales_df):
    if sales_df.empty or "quantity_dispensed" not in sales_df.columns:
        return pd.Series(dtype=float)
    return sales_df.groupby("drug_id")["quantity_dispensed"].sum()


def get_current_stock_by_drug(stock_df, sales_df):
    """Estimated current stock per drug = total received - total dispensed, min 0."""
    received = compute_total_received_by_drug(stock_df)
    dispensed = compute_total_dispensed_by_drug(sales_df)
    all_ids = set(received.index) | set(dispensed.index)
    out = {}
    for did in all_ids:
        r = received.get(did, 0)
        d = dispensed.get(did, 0)
        out[did] = max(0, float(r) - float(d))
    return out


# ---------------------------------------------------------------------------
# Next month forecast and forecast next 3 months total
# ---------------------------------------------------------------------------
def get_next_month_forecast(forecast_df):
    if forecast_df.empty:
        return {}
    first = forecast_df["forecast_month"].min()
    sub = forecast_df[forecast_df["forecast_month"] == first]
    return dict(zip(sub["drug_id"], sub["predicted_demand"].astype(float)))


def get_forecast_next_3_months_total(forecast_df):
    if forecast_df.empty:
        return {}
    return forecast_df.groupby("drug_id")["predicted_demand"].sum().to_dict()


# ---------------------------------------------------------------------------
# Last 3 months demand aggregation
# ---------------------------------------------------------------------------
def get_avg_last_3_months(monthly_df, drug_ids=None):
    if monthly_df.empty:
        return {}
    m = monthly_df.sort_values("month")
    if drug_ids is None:
        drug_ids = m["drug_id"].unique()
    out = {}
    for did in drug_ids:
        sub = m[m["drug_id"] == did].tail(3)
        out[did] = sub["monthly_demand"].mean() if len(sub) else 0.0
    return out


def get_total_demand_last_3_months(monthly_df, drug_ids=None):
    if monthly_df.empty:
        return {}
    m = monthly_df.sort_values("month")
    if drug_ids is None:
        drug_ids = m["drug_id"].unique()
    out = {}
    for did in drug_ids:
        sub = m[m["drug_id"] == did].tail(3)
        out[did] = sub["monthly_demand"].sum() if len(sub) else 0.0
    return out


# ---------------------------------------------------------------------------
# Days of supply: current_stock / (avg_monthly_last_3 / 30)
# ---------------------------------------------------------------------------
def calc_days_of_supply(current_stock, avg_monthly_demand):
    if avg_monthly_demand is None or avg_monthly_demand <= 0:
        return None
    avg_daily = avg_monthly_demand / 30.0
    if avg_daily <= 0:
        return None
    return current_stock / avg_daily


def get_days_of_supply_status(days):
    if days is None or (isinstance(days, float) and np.isnan(days)):
        return "—", "—"
    if days < 7:
        return "CRITICAL", "🔴"
    if days < 14:
        return "LOW", "🟡"
    return "SAFE", "🟢"


# ---------------------------------------------------------------------------
# Stockout risk: compare current stock to next month forecast
# ---------------------------------------------------------------------------
def get_stockout_risk(current_stock, next_month_forecast):
    if next_month_forecast is None or next_month_forecast <= 0:
        return "LOW"
    ratio = current_stock / next_month_forecast
    if ratio < 1.0:
        return "HIGH"
    if ratio < 1.5:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Expiry risk from stock_receipts (nearest upcoming expiry per drug)
# ---------------------------------------------------------------------------
def get_expiry_risk_by_drug(stock_df):
    if stock_df.empty or "expiry_date" not in stock_df.columns:
        return {}
    today = pd.Timestamp.now().normalize()
    stock_df = stock_df.dropna(subset=["expiry_date"])
    stock_df = stock_df[stock_df["expiry_date"] >= today]
    if stock_df.empty:
        return {}
    nearest = stock_df.loc[stock_df.groupby("drug_id")["expiry_date"].idxmin()]
    out = {}
    for _, r in nearest.iterrows():
        did = r["drug_id"]
        exp = r["expiry_date"]
        days = (pd.Timestamp(exp) - today).days
        if days <= 60:
            risk = "HIGH"
        elif days <= 90:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        out[did] = {"expiry_date": exp, "days_to_expiry": days, "expiry_risk": risk}
    return out


# ---------------------------------------------------------------------------
# Restock interval (avg days between receipts) per drug
# ---------------------------------------------------------------------------
def get_avg_restock_interval_days(stock_df):
    if stock_df.empty:
        return {}
    out = {}
    for did in stock_df["drug_id"].unique():
        sub = stock_df[stock_df["drug_id"] == did].sort_values("stock_received_date")
        dates = pd.to_datetime(sub["stock_received_date"]).sort_values()
        if len(dates) >= 2:
            gaps = dates.diff().dt.days.dropna()
            out[did] = int(round(gaps.mean()))
        else:
            out[did] = 30
    return out


# ---------------------------------------------------------------------------
# Reorder quantity: base = next_month_forecast + 25% buffer; subtract current stock
# ---------------------------------------------------------------------------
def calc_suggested_reorder_qty(next_month_forecast, current_stock, buffer_pct=0.25):
    if next_month_forecast is None:
        next_month_forecast = 0.0
    with_buffer = next_month_forecast * (1 + buffer_pct)
    qty = max(0.0, with_buffer - (current_stock or 0))
    return int(round(qty))


# ---------------------------------------------------------------------------
# Demand change: (forecast_next - avg_last_3) / avg_last_3 * 100
# ---------------------------------------------------------------------------
def get_demand_change_pct(avg_last_3, forecast_next):
    if avg_last_3 is None or avg_last_3 <= 0:
        return None
    return ((forecast_next - avg_last_3) / avg_last_3) * 100


def get_demand_change_label(pct):
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "—", "No change"
    if pct > 5:
        return "↑", "Expected demand is rising"
    if pct < -5:
        return "↓", "Expected demand is falling"
    return "→", "Expected demand is stable"


# ---------------------------------------------------------------------------
# Stock context for selected drug
# ---------------------------------------------------------------------------
def get_stock_context(drug_id, stock_df, current_stock_by_drug=None):
    if stock_df.empty or drug_id not in stock_df["drug_id"].values:
        return None
    sub = stock_df[stock_df["drug_id"] == drug_id].sort_values("stock_received_date")
    if sub.empty:
        return None
    last = sub.iloc[-1]
    last_date = last["stock_received_date"]
    last_qty = last["quantity_received"]
    dates = pd.to_datetime(sub["stock_received_date"]).sort_values()
    avg_interval = int(round(dates.diff().dt.days.dropna().mean())) if len(dates) >= 2 else None
    days_since = (pd.Timestamp.now().normalize() - pd.Timestamp(last_date)).days
    due_soon = avg_interval is not None and days_since >= avg_interval
    ctx = {
        "last_restock_date": last_date,
        "last_quantity_received": last_qty,
        "avg_restock_interval_days": avg_interval,
        "days_since_restock": days_since,
        "restock_due_soon": due_soon,
    }
    if current_stock_by_drug is not None:
        ctx["current_stock"] = current_stock_by_drug.get(drug_id, 0)
    return ctx


# ---------------------------------------------------------------------------
# Build reorder table (with current stock, demand change, suggested reorder)
# ---------------------------------------------------------------------------
def build_reorder_table(
    monthly_df,
    forecast_df,
    id_to_name,
    current_stock_by_drug,
    next_month_forecast_by_drug,
    avg_last_3_by_drug,
):
    if monthly_df.empty or forecast_df.empty:
        return pd.DataFrame()
    monthly_sorted = monthly_df.sort_values(["drug_id", "month"])
    drug_ids = monthly_sorted["drug_id"].unique()
    next_fc = next_month_forecast_by_drug or {}
    all_next = pd.Series(next_fc)
    p75 = all_next.quantile(0.75) if len(all_next) else 0
    p25 = all_next.quantile(0.25) if len(all_next) else 0

    rows = []
    for drug_id in drug_ids:
        sub = monthly_sorted[monthly_sorted["drug_id"] == drug_id]
        last_3 = sub.tail(3)
        avg_last_3 = last_3["monthly_demand"].mean() if len(last_3) else 0.0
        expected_next = next_fc.get(drug_id, 0.0)
        current_stock = current_stock_by_drug.get(drug_id, 0)
        suggested = calc_suggested_reorder_qty(expected_next, current_stock, 0.25)

        high_demand = expected_next >= p75 if pd.notna(p75) else False
        low_demand = expected_next <= p25 if pd.notna(p25) else False
        last_2 = sub.tail(2)["monthly_demand"]
        rising_trend = (
            (len(last_2) >= 2 and last_2.iloc[-1] > last_2.iloc[-2])
            or (len(last_2) == 1 and last_2.iloc[-1] > avg_last_3)
        )
        forecast_decreasing = avg_last_3 > 0 and expected_next < avg_last_3
        if high_demand or rising_trend:
            priority = "HIGH"
        elif forecast_decreasing and low_demand:
            priority = "LOW"
        else:
            priority = "MEDIUM"

        pct = get_demand_change_pct(avg_last_3, expected_next)
        _, change_label = get_demand_change_label(pct)

        rows.append({
            "drug_id": drug_id,
            "Drug name": id_to_name.get(drug_id, drug_id),
            "Current stock": int(current_stock),
            "Avg demand (last 3 months)": round(avg_last_3, 1),
            "Forecast (next month)": round(expected_next, 1),
            "Demand trend": change_label,
            "Suggested reorder (units)": suggested,
            "Priority": priority,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Days of supply table
# ---------------------------------------------------------------------------
def build_days_of_supply_table(drug_ids, id_to_name, current_stock_by_drug, avg_last_3_by_drug):
    rows = []
    for did in drug_ids:
        stock = current_stock_by_drug.get(did, 0)
        avg_m = avg_last_3_by_drug.get(did)
        days = calc_days_of_supply(stock, avg_m)
        status, emoji = get_days_of_supply_status(days)
        avg_daily = (avg_m / 30.0) if avg_m and avg_m > 0 else None
        rows.append({
            "drug_id": did,
            "Drug name": id_to_name.get(did, did),
            "Current stock": int(stock),
            "Avg daily demand": round(avg_daily, 1) if avg_daily is not None else "—",
            "Days of supply": round(days, 1) if days is not None else "—",
            "Status": f"{emoji} {status}" if emoji != "—" else "—",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stockout risk table
# ---------------------------------------------------------------------------
def build_stockout_risk_table(drug_ids, id_to_name, current_stock_by_drug, next_month_forecast_by_drug):
    rows = []
    for did in drug_ids:
        stock = current_stock_by_drug.get(did, 0)
        fc = next_month_forecast_by_drug.get(did)
        risk = get_stockout_risk(stock, fc)
        rows.append({
            "drug_id": did,
            "Drug name": id_to_name.get(did, did),
            "Current stock": int(stock),
            "Forecast (next month)": round(fc, 1) if fc is not None else "—",
            "Stockout risk": risk,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Category-level monthly demand from sales (month = proper datetime, freq MS)
# ---------------------------------------------------------------------------
@st.cache_data
def load_category_monthly_demand():
    sales = load_sales_transactions()
    if sales.empty or "transaction_date" not in sales.columns or "category" not in sales.columns:
        return pd.DataFrame(columns=["month", "category", "demand"])
    category_demand = (
        sales.groupby(
            [pd.Grouper(key="transaction_date", freq="MS"), "category"],
            as_index=False,
        )["quantity_dispensed"]
        .sum()
        .rename(columns={"transaction_date": "month", "quantity_dispensed": "demand"})
    )
    category_demand["month"] = pd.to_datetime(category_demand["month"]).dt.normalize()
    return category_demand


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_demand_forecast(monthly_df, forecast_df, drug_id, drug_name, months_limit=12):
    sub_m = monthly_df[monthly_df["drug_id"] == drug_id].sort_values("month").tail(months_limit or 12)
    sub_f = forecast_df[forecast_df["drug_id"] == drug_id].sort_values("forecast_month")
    fig = go.Figure()
    if sub_m.empty and sub_f.empty:
        fig.add_annotation(text="No data for this drug", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    if not sub_m.empty:
        fig.add_trace(
            go.Scatter(
                x=sub_m["month"], y=sub_m["monthly_demand"],
                name="Demand (actual)", mode="lines+markers",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=6),
            )
        )
    if not sub_f.empty:
        fig.add_trace(
            go.Scatter(
                x=sub_f["forecast_month"], y=sub_f["predicted_demand"],
                name="Forecast", mode="lines+markers",
                line=dict(color="#ff7f0e", width=3, dash="dash"),
                marker=dict(size=7, symbol="diamond"),
            )
        )
    fig.update_layout(
        title=dict(text=f"Demand & forecast — {drug_name}", x=0.5, xanchor="center", y=0.98, yref="paper", font=dict(size=16)),
        xaxis_title="Month", yaxis_title="Demand (units)",
        legend=dict(orientation="h", yanchor="top", y=0.92, xanchor="center", x=0.5),
        margin=dict(t=110, b=40, l=40, r=20), height=440,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def plot_top5_demand(total_demand_last_3, id_to_name, top_n=5):
    items = sorted(total_demand_last_3.items(), key=lambda x: -x[1])[:top_n]
    if not items:
        return go.Figure()
    names = [id_to_name.get(did, did) for did, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#2ca02c"))
    fig.update_layout(
        title="Top 5 drugs by demand (last 3 months)",
        xaxis_title="Total demand (units)",
        yaxis_title="", height=300, margin=dict(l=120),
    )
    return fig


def plot_category_demand_over_time(cat_monthly_df, months_limit=None):
    if cat_monthly_df.empty or "month" not in cat_monthly_df.columns or "demand" not in cat_monthly_df.columns:
        return go.Figure()
    df = cat_monthly_df.copy()
    df["month"] = pd.to_datetime(df["month"]).dt.normalize()
    df = df.sort_values("month")
    fig = px.line(
        df,
        x="month",
        y="demand",
        color="category",
        markers=True,
        title="Demand by category over time",
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Demand (units)",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    return fig


# ---------------------------------------------------------------------------
# Summary stats for cards
# ---------------------------------------------------------------------------
def get_summary_stats(
    monthly_df,
    forecast_df,
    reorder_df,
    stockout_risk_table_df,
    expiry_risk_by_drug,
):
    n_drugs = monthly_df["drug_id"].nunique() if not monthly_df.empty else 0
    if monthly_df.empty:
        this_month_total = 0.0
    else:
        last_month = monthly_df["month"].max()
        this_month_total = float(monthly_df[monthly_df["month"] == last_month]["monthly_demand"].sum())
    if forecast_df.empty:
        forecast_next_total = 0.0
    else:
        first_fc = forecast_df["forecast_month"].min()
        forecast_next_total = float(forecast_df[forecast_df["forecast_month"] == first_fc]["predicted_demand"].sum())
    needing_attention = (reorder_df["Priority"] == "HIGH").sum() if not reorder_df.empty else 0
    high_stockout = (
        (stockout_risk_table_df["Stockout risk"] == "HIGH").sum()
        if not stockout_risk_table_df.empty and "Stockout risk" in stockout_risk_table_df.columns
        else 0
    )
    expiring_soon = sum(1 for v in (expiry_risk_by_drug or {}).values() if v.get("expiry_risk") in ("HIGH", "MEDIUM"))
    return {
        "n_drugs": n_drugs,
        "this_month_total": this_month_total,
        "forecast_next_total": forecast_next_total,
        "needing_attention": needing_attention,
        "high_stockout_risk": high_stockout,
        "expiring_soon": expiring_soon,
    }


# ---------------------------------------------------------------------------
# Procurement Planner table
# ---------------------------------------------------------------------------
def build_procurement_planner_table(
    drug_ids,
    id_to_name,
    current_stock_by_drug,
    forecast_next_3_total,
    next_month_forecast_by_drug,
    avg_last_3_by_drug,
    expiry_risk_by_drug,
):
    rows = []
    for did in drug_ids:
        stock = current_stock_by_drug.get(did, 0)
        fc3 = forecast_next_3_total.get(did, 0)
        fc1 = next_month_forecast_by_drug.get(did)
        avg3 = avg_last_3_by_drug.get(did)
        suggested = calc_suggested_reorder_qty(fc1, stock, 0.25)
        risk = get_stockout_risk(stock, fc1)
        days = calc_days_of_supply(stock, avg3)
        days_status, _ = get_days_of_supply_status(days)
        exp = expiry_risk_by_drug.get(did, {})
        exp_risk = exp.get("expiry_risk", "—")
        rows.append({
            "Drug name": id_to_name.get(did, did),
            "Current stock": int(stock),
            "Forecast (next 3 months)": round(fc3, 0),
            "Suggested reorder": suggested,
            "Stockout risk": risk,
            "Days of supply": round(days, 1) if days is not None else "—",
            "Expiry risk": exp_risk,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Expiring soon table
# ---------------------------------------------------------------------------
def build_expiring_soon_table(expiry_risk_by_drug, id_to_name):
    if not expiry_risk_by_drug:
        return pd.DataFrame()
    rows = []
    for did, v in expiry_risk_by_drug.items():
        rows.append({
            "drug_id": did,
            "Drug name": id_to_name.get(did, did),
            "Nearest expiry date": pd.Timestamp(v["expiry_date"]).strftime("%Y-%m-%d"),
            "Days to expiry": v["days_to_expiry"],
            "Expiry risk": v["expiry_risk"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Time range filter
# ---------------------------------------------------------------------------
def filter_monthly_by_time_range(monthly_df, time_range):
    if monthly_df.empty:
        return monthly_df
    max_month = monthly_df["month"].max()
    if time_range == "Last 6 months":
        min_month = max_month - pd.DateOffset(months=5)
        return monthly_df[monthly_df["month"] >= min_month]
    if time_range == "Last 12 months":
        min_month = max_month - pd.DateOffset(months=11)
        return monthly_df[monthly_df["month"] >= min_month]
    return monthly_df


def get_months_limit(time_range):
    if time_range == "Last 6 months":
        return 6
    if time_range == "Last 12 months":
        return 12
    return None


# ---------------------------------------------------------------------------
# Page filtering helpers (category filter)
# ---------------------------------------------------------------------------
def filter_dict_by_allowed_ids(d: dict, allowed_ids: set | None):
    if not allowed_ids:
        return d
    return {k: v for k, v in d.items() if k in allowed_ids}


def filter_df_by_allowed_ids(df: pd.DataFrame, allowed_ids: set | None, id_col: str = "drug_id") -> pd.DataFrame:
    if df is None or df.empty or not allowed_ids or id_col not in df.columns:
        return df
    return df[df[id_col].isin(allowed_ids)].copy()


# ---------------------------------------------------------------------------
# Page 1: Overview — pharmacy-wide only (no selected-drug content)
# ---------------------------------------------------------------------------
def render_overview_page(stats, total_last_3, id_to_name, stockout_risk_df, expiring_df, cat_monthly_df, time_range):
    """Pharmacy-level summary: cards, top 5 chart, all-drug risk tables, category chart."""
    with st.container():
        section_header("Overview", "Pharmacy-wide summary and monitoring.")
        # Summary cards (2 rows, 3 columns each)
        r1 = st.columns(3)
        with r1[0]:
            render_card("Total drugs tracked", format_int(stats["n_drugs"]), icon="💊", help_text="Distinct drugs in the dataset.")
        with r1[1]:
            render_card("This month demand", format_int(stats["this_month_total"]), icon="📦", help_text="Total units dispensed in the latest month.")
        with r1[2]:
            render_card("Forecast next month", format_int(stats["forecast_next_total"]), icon="📈", help_text="Total predicted demand for the next month.")

        r2 = st.columns(3)
        with r2[0]:
            render_card("Drugs needing attention", format_int(stats["needing_attention"]), icon="⚠️", help_text="High priority based on demand level/trend.")
        with r2[1]:
            render_card("High stockout risk", format_int(stats["high_stockout_risk"]), icon="🧯", help_text="Drugs where stock is below next-month forecast.")
        with r2[2]:
            render_card("Expiring soon", format_int(stats["expiring_soon"]), icon="⏳", help_text="Drugs with an expiry within 90 days.")

    # Top 5 drugs by demand
    st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)
    section_header("Top 5 drugs by demand", "Biggest movers by total demand over the last 3 months.")
    fig_top = plot_top5_demand(total_last_3, id_to_name, 5)
    st.plotly_chart(fig_top, use_container_width=True)

    # All-drug tables in expanders to keep page short
    with st.expander("Stockout risk by drug", expanded=True):
        st.caption("Color-coded risk levels for quick action prioritization.")
        styled = style_stockout_table(stockout_risk_df.drop(columns=["drug_id"], errors="ignore"))
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("Expiring soon (all drugs)"):
        st.caption("Nearest expiry per drug. HIGH: within 60 days; MEDIUM: within 90 days.")
        if not expiring_df.empty:
            styled = style_expiry_table(expiring_df.drop(columns=["drug_id"], errors="ignore"))
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("No upcoming expiries in the data.")

    # Category chart (optional, in expander)
    with st.expander("Demand by category over time"):
        months_limit = get_months_limit(time_range)
        if months_limit and not cat_monthly_df.empty:
            max_m = pd.to_datetime(cat_monthly_df["month"]).max()
            cutoff_date = max_m - pd.DateOffset(months=months_limit - 1)
            cat_filtered = cat_monthly_df[pd.to_datetime(cat_monthly_df["month"]) >= cutoff_date].copy()
        else:
            cat_filtered = cat_monthly_df.copy() if not cat_monthly_df.empty else cat_monthly_df
        if cat_filtered.empty:
            st.info("No category demand data available for the selected time range.")
        else:
            fig_cat = plot_category_demand_over_time(cat_filtered, months_limit)
            st.plotly_chart(fig_cat, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2: Drug Detail — selected drug only (no all-drug tables)
# ---------------------------------------------------------------------------
def render_drug_detail_page(
    selected_id,
    selected_name,
    selected_category,
    monthly_df,
    forecast_df,
    stock_df,
    current_stock_by_drug,
    next_month_fc,
    avg_last_3,
    expiry_risk_by_drug,
    time_range,
):
    """Single-drug view: summary cards, demand chart, restock context, trend text, export."""
    section_header("Drug detail", "Focused view for the selected drug.")
    st.markdown(
        f"""
<div class="pf-card" style="padding: 0.95rem 1.0rem; margin-bottom: 0.6rem;">
  <div style="font-size: 1.25rem; font-weight: 800;">{selected_name}</div>
  <div style="opacity: 0.78; font-size: 0.88rem;">Category: {selected_category if selected_category else "—"}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Summary cards in one clean row (5 cards)
    stock_val = int(current_stock_by_drug.get(selected_id, 0))
    fc_next = next_month_fc.get(selected_id)
    days = calc_days_of_supply(stock_val, avg_last_3.get(selected_id))
    days_status, days_emoji = get_days_of_supply_status(days)
    risk = get_stockout_risk(stock_val, fc_next)
    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "")
    exp = expiry_risk_by_drug.get(selected_id, {})
    exp_risk = exp.get("expiry_risk", "—")

    cards = st.columns(5)
    with cards[0]:
        render_card("Current stock", format_int(stock_val), icon="📦", help_text="Estimated: total received − total dispensed.")
    with cards[1]:
        render_card("Next month forecast", format_int(fc_next), icon="📈", help_text="Expected demand next month.")
    with cards[2]:
        render_card("Days of supply", format_float(days, 1) if days is not None else "—", icon="🕒", help_text=f"Status: {days_emoji} {days_status}")
    with cards[3]:
        render_card("Stockout risk", f"{risk_emoji} {risk}", icon="🧯", help_text="Based on stock vs next-month forecast.")
    with cards[4]:
        render_card("Expiry risk", exp_risk, icon="⏳", help_text="Nearest upcoming batch expiry.")

    st.markdown("<div style='height: 0.9rem'></div>", unsafe_allow_html=True)
    section_header("Demand and forecast", "Actual demand vs next 3 month forecast (dashed).")
    months_limit = get_months_limit(time_range) or 12
    monthly_filtered = filter_monthly_by_time_range(monthly_df, time_range)
    fig = plot_demand_forecast(monthly_filtered, forecast_df, selected_id, selected_name, months_limit)
    st.plotly_chart(fig, use_container_width=True)

    # Restock context
    st.markdown("<div style='height: 0.3rem'></div>", unsafe_allow_html=True)
    section_header("Restock context", "Recent supply activity for the selected drug.")
    ctx = get_stock_context(selected_id, stock_df, current_stock_by_drug)
    if ctx:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_card("Last restock date", pd.Timestamp(ctx["last_restock_date"]).strftime("%Y-%m-%d"), icon="📅")
        with c2:
            render_card("Last received qty", format_int(ctx["last_quantity_received"]), icon="📥")
        with c3:
            render_card("Avg restock interval", f"{ctx['avg_restock_interval_days'] or '—'} days", icon="⏱️")
        with c4:
            render_card("Days since restock", format_int(ctx["days_since_restock"]), icon="🧭")
        if ctx["restock_due_soon"]:
            st.info("Restock due soon — consider reordering.")
    else:
        st.info("No restock history for this drug.")

    # Plain-language demand trend
    st.markdown("<div style='height: 0.2rem'></div>", unsafe_allow_html=True)
    section_header("Demand trend", "Plain-language interpretation for quick decision-making.")
    pct = get_demand_change_pct(avg_last_3.get(selected_id), fc_next)
    _, change_label = get_demand_change_label(pct)
    st.info(change_label)

    # Selected-drug export only
    st.markdown("<div style='height: 0.2rem'></div>", unsafe_allow_html=True)
    section_header("Export", "Download a clean CSV for this drug.")
    fc_sel = forecast_df[forecast_df["drug_id"] == selected_id].copy()
    if not fc_sel.empty:
        fc_export = fc_sel[["drug_id", "forecast_month", "predicted_demand"]].rename(
            columns={"forecast_month": "Forecast month", "predicted_demand": "Predicted demand"}
        )
        fc_export.insert(1, "drug_name", selected_name)
        buf = StringIO()
        fc_export.to_csv(buf, index=False)
        st.download_button(
            "Download forecast for this drug (CSV)",
            buf.getvalue(),
            file_name=f"forecast_{selected_id}.csv",
            mime="text/csv",
            key="dl_forecast_drug",
        )


# ---------------------------------------------------------------------------
# Page 3: Procurement Planner — action tables and exports only
# ---------------------------------------------------------------------------
def render_procurement_page(
    reorder_df,
    days_supply_df,
    stockout_risk_df,
):
    """Operational planning: reorder table, stock coverage, filters, exports."""
    # Optional summary cards
    high_priority = (reorder_df["Priority"] == "HIGH").sum() if not reorder_df.empty else 0
    total_suggested = reorder_df["Suggested reorder (units)"].sum() if not reorder_df.empty and "Suggested reorder (units)" in reorder_df.columns else 0
    section_header("Procurement planner", "Action tables for ordering and coverage planning.")
    row = st.columns(3)
    with row[0]:
        render_card("HIGH reorder priority", format_int(high_priority), icon="⚠️", help_text="Drugs prioritized for attention.")
    with row[1]:
        render_card("Total suggested order", format_int(total_suggested), icon="🧾", help_text="Sum of suggested reorder across drugs.")
    with row[2]:
        high_stockout = (stockout_risk_df.get("Stockout risk") == "HIGH").sum() if isinstance(stockout_risk_df, pd.DataFrame) and not stockout_risk_df.empty else 0
        render_card("High stockout risk", format_int(high_stockout), icon="🧯", help_text="Quick signal for urgency.")

    # Sort control
    sort_by = st.selectbox(
        "Sort reorder table by",
        ["Suggested reorder (units)", "Priority", "Drug name"],
        key="proc_sort",
    )
    display_reorder = reorder_df.drop(columns=["drug_id"], errors="ignore")
    if not display_reorder.empty and sort_by == "Suggested reorder (units)":
        display_reorder = display_reorder.sort_values("Suggested reorder (units)", ascending=False)
    elif not display_reorder.empty and sort_by == "Priority":
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        display_reorder = display_reorder.copy()
        display_reorder["_order"] = display_reorder["Priority"].map(order).fillna(3)
        display_reorder = display_reorder.sort_values("_order").drop(columns=["_order"])
    elif not display_reorder.empty:
        display_reorder = display_reorder.sort_values("Drug name")

    st.markdown("<div style='height: 0.9rem'></div>", unsafe_allow_html=True)
    section_header("Suggested reorder", "Suggested order = (next month forecast + 25% buffer) − current stock.")
    styled = style_reorder_table(display_reorder)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("Stock coverage (days of supply)"):
        st.caption("How long current stock would last at recent demand. CRITICAL: under 7 days; LOW: under 14; SAFE: 14+.")
        styled = style_days_supply_table(days_supply_df.drop(columns=["drug_id"], errors="ignore"))
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Export buttons
    st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)
    section_header("Export", "Download tables for procurement actions.")
    col_a, col_b = st.columns(2)
    with col_a:
        if not reorder_df.empty:
            reorder_export = reorder_df.drop(columns=["drug_id"], errors="ignore")
            buf = StringIO()
            reorder_export.to_csv(buf, index=False)
            st.download_button(
                "Download reorder suggestion table (CSV)",
                buf.getvalue(),
                file_name="reorder_suggestions.csv",
                mime="text/csv",
                key="dl_reorder",
            )
    with col_b:
        if not stockout_risk_df.empty:
            buf2 = StringIO()
            stockout_risk_df.drop(columns=["drug_id"], errors="ignore").to_csv(buf2, index=False)
            st.download_button(
                "Download stockout risk table (CSV)",
                buf2.getvalue(),
                file_name="stockout_risk.csv",
                mime="text/csv",
                key="dl_stockout",
            )


def render_upload_refresh_page(data_dir: Path, outputs_dir: Path):
    section_header("Upload & Refresh Data", "Upload the latest full files and refresh forecasts and dashboard outputs.")

    st.markdown(
        """
**Simple, safe update approach**
- Uploaded files are treated as the **latest authoritative full records** and will **replace** the existing raw CSVs in `data/`.
- After processing, forecasts and dashboard recommendations will refresh automatically.

**First-time setup:** upload **Sales**, **Stock receipts**, and **Opening stock**.  
**Regular updates:** upload the latest full **Sales** and **Stock receipts**; **Opening stock** is optional if unchanged.
        """
    )

    with st.expander("Upload files", expanded=True):
        sales_file = st.file_uploader("Sales / dispensing file (CSV)", type=["csv"], key="upl_sales")
        stock_file = st.file_uploader("Stock receipts file (CSV)", type=["csv"], key="upl_stock")
        opening_file = st.file_uploader(
            "Opening stock file (CSV) — optional after initial setup",
            type=["csv"],
            key="upl_opening",
        )

    # Initial setup if any required raw file is missing on disk
    initial_setup_needed = not (
        (data_dir / "sales_transactions.csv").exists()
        and (data_dir / "stock_receipts.csv").exists()
        and (data_dir / "opening_stock.csv").exists()
    )

    st.info(
        "Initial setup is required (missing one or more raw data files)." if initial_setup_needed
        else "Regular update mode: upload the latest full files and refresh."
    )

    # Validate any provided uploads and show previews
    validations = {}
    if sales_file is not None:
        validations["Sales"] = validate_sales_upload(sales_file)
    if stock_file is not None:
        validations["Stock receipts"] = validate_stock_receipts_upload(stock_file)
    if opening_file is not None:
        validations["Opening stock"] = validate_opening_stock_upload(opening_file)

    if validations:
        section_header("Validation preview", "Quick checks before saving and processing.")
        for label, res in validations.items():
            if res.ok:
                st.success(f"{label}: {res.message}")
            else:
                st.error(f"{label}: {res.message}")
            if res.df_preview is not None and not res.df_preview.empty:
                st.dataframe(res.df_preview, use_container_width=True, hide_index=True)

    # Guidance warnings (non-blocking)
    if initial_setup_needed:
        needed = []
        if sales_file is None:
            needed.append("Sales / dispensing")
        if stock_file is None:
            needed.append("Stock receipts")
        if opening_file is None:
            needed.append("Opening stock")
        if needed:
            st.warning(f"To complete first-time setup, please upload: {', '.join(needed)}.")
    else:
        if sales_file is None or stock_file is None:
            st.warning(
                "For a proper refresh, upload both the latest full Sales and Stock receipts files. "
                "Opening stock is optional."
            )

    st.markdown("---")
    process_clicked = st.button("Process uploaded files and refresh dashboard", type="primary")
    if not process_clicked:
        return

    # Hard validation gate
    for res in validations.values():
        if not res.ok:
            st.error("Fix validation errors above before processing.")
            return

    if initial_setup_needed and (sales_file is None or stock_file is None or opening_file is None):
        st.error("Initial setup requires all three files: Sales, Stock receipts, and Opening stock.")
        return

    # Save uploaded files (replace raw CSVs); keep existing file if not uploaded
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        if sales_file is not None:
            (data_dir / "sales_transactions.csv").write_bytes(sales_file.getvalue())
        if stock_file is not None:
            (data_dir / "stock_receipts.csv").write_bytes(stock_file.getvalue())
        if opening_file is not None:
            (data_dir / "opening_stock.csv").write_bytes(opening_file.getvalue())
    except Exception as e:
        st.error(f"Failed to save uploaded files: {e}")
        return

    # Run processing and regenerate outputs
    try:
        with st.spinner("Processing data and regenerating outputs (monthly demand, stock status, forecasts)…"):
            _ = refresh_all(data_dir=data_dir, outputs_dir=outputs_dir)
        st.success("Refresh complete. Dashboard outputs have been updated.")
        st.caption("Navigate to Overview / Procurement Planner / Drug Detail to see updated results.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Refresh failed: {e}")


def main():
    st.set_page_config(
        page_title="Pharmacy Dashboard",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()
    render_product_header()
    render_last_refresh_chip()

    # Load data once
    monthly = load_monthly_demand()
    forecast = load_forecast()
    stock = load_stock_receipts()
    sales = load_sales_transactions()
    sales_drugs = load_drug_names()
    id_to_name, id_to_cat = build_drug_lookup(sales_drugs)

    if monthly.empty or forecast.empty:
        st.warning(
            "Monthly demand or forecast data not found. Run notebooks 01 and 02–05 to generate "
            "`outputs/monthly_demand.csv` and `outputs/next_3_month_forecast.csv`, then restart the dashboard."
        )
        st.stop()

    # Shared computed data
    stock_status_df = load_stock_status()
    if not stock_status_df.empty and "current_stock" in stock_status_df.columns:
        current_stock_by_drug = {
            str(row["drug_id"]): int(row["current_stock"])
            for _, row in stock_status_df.iterrows()
        }
    else:
        # Fallback to inferred stock if processed status is not available
        current_stock_by_drug = get_current_stock_by_drug(stock, sales)
    next_month_fc = get_next_month_forecast(forecast)
    avg_last_3 = get_avg_last_3_months(monthly)
    total_last_3 = get_total_demand_last_3_months(monthly)
    forecast_next_3_total = get_forecast_next_3_months_total(forecast)
    expiry_risk_by_drug = get_expiry_risk_by_drug(stock)

    reorder_df = build_reorder_table(
        monthly, forecast, id_to_name,
        current_stock_by_drug, next_month_fc, avg_last_3,
    )
    drug_ids_list = list(avg_last_3.keys()) if avg_last_3 else []
    days_supply_df = build_days_of_supply_table(
        drug_ids_list, id_to_name, current_stock_by_drug, avg_last_3,
    )
    stockout_risk_df = build_stockout_risk_table(
        list(next_month_fc.keys()) if next_month_fc else [],
        id_to_name, current_stock_by_drug, next_month_fc,
    )
    expiring_df = build_expiring_soon_table(expiry_risk_by_drug, id_to_name)
    stats = get_summary_stats(monthly, forecast, reorder_df, stockout_risk_df, expiry_risk_by_drug)
    cat_monthly_df = load_category_monthly_demand()

    # ---------------- Sidebar control panel ----------------
    st.sidebar.markdown("## Control panel")

    # 1) Navigation (default = Overview)
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Procurement Planner", "Drug Detail", "Upload & Refresh Data"],
        index=0,
        key="page_nav",
    )

    # 2) Drug selection (above Filters; category filter lives here)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Drug selection")
    drug_options = [(id_to_name.get(d, d), d) for d in monthly["drug_id"].unique()]
    drug_options.sort(key=lambda x: x[0])
    categories = ["All"] + sorted({c for c in id_to_cat.values() if c and pd.notna(c)})
    sel_cat = st.sidebar.selectbox("Filter by category", categories, key="filter_category")
    allowed_ids = {k for k, v in id_to_cat.items() if v == sel_cat} if sel_cat != "All" else None
    if allowed_ids:
        drug_options = [(n, i) for n, i in drug_options if i in allowed_ids]
    if not drug_options:
        st.sidebar.warning("No drugs match the selected category.")
        drug_options = [(id_to_name.get(d, d), d) for d in monthly["drug_id"].unique()]
        drug_options.sort(key=lambda x: x[0])
    drug_names_for_select = [x[0] for x in drug_options]
    selected_name = st.sidebar.selectbox("Selected drug", drug_names_for_select, key="drug_select")
    selected_id = next(i for n, i in drug_options if n == selected_name)
    selected_category = id_to_cat.get(selected_id, "")

    # 3) Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")
    time_range = st.sidebar.selectbox(
        "Time range for charts",
        ["Last 6 months", "Last 12 months", "All data"],
        key="time_range",
    )
    st.sidebar.caption("Affects Overview/Planner charts. Forecast always shows next 3 months.")

    # ---------------- Apply category filter to pharmacy-wide outputs ----------------
    total_last_3_f = filter_dict_by_allowed_ids(total_last_3, allowed_ids)
    stockout_risk_df_f = filter_df_by_allowed_ids(stockout_risk_df, allowed_ids, "drug_id")
    expiring_df_f = filter_df_by_allowed_ids(expiring_df, allowed_ids, "drug_id")
    reorder_df_f = filter_df_by_allowed_ids(reorder_df, allowed_ids, "drug_id")
    days_supply_df_f = filter_df_by_allowed_ids(days_supply_df, allowed_ids, "drug_id")

    # ---------------- Render selected page ----------------
    if page == "Upload & Refresh Data":
        render_upload_refresh_page(DATA_DIR, OUTPUTS_DIR)
        return
    if page == "Overview":
        render_overview_page(
            stats, total_last_3_f, id_to_name,
            stockout_risk_df_f, expiring_df_f, cat_monthly_df, time_range,
        )
    elif page == "Procurement Planner":
        render_procurement_page(reorder_df_f, days_supply_df_f, stockout_risk_df_f)
    else:
        render_drug_detail_page(
            selected_id, selected_name, selected_category,
            monthly, forecast, stock,
            current_stock_by_drug, next_month_fc, avg_last_3, expiry_risk_by_drug,
            time_range,
        )


if __name__ == "__main__":
    main()
