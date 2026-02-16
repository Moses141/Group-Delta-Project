"""
Pharmacy database module for stock transactions (purchases and restocks).
Uses SQLite for simple, file-based persistence.
"""
import sqlite3
import os
import pandas as pd
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "pharmacy_stock.db")


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Allow dict-like access
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_id TEXT NOT NULL,
                drug_name TEXT NOT NULL,
                location_key TEXT NOT NULL,
                transaction_type TEXT NOT NULL CHECK(transaction_type IN ('purchase', 'restock')),
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                transaction_date DATE NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_drug 
            ON stock_transactions(drug_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_date 
            ON stock_transactions(transaction_date)
        """)


def record_transaction(drug_id: str, drug_name: str, location_key: str,
                       transaction_type: str, quantity: int, transaction_date, notes: str = "") -> bool:
    """Record a purchase or restock transaction."""
    if transaction_type not in ('purchase', 'restock'):
        return False
    if quantity <= 0:
        return False
    try:
        date_str = transaction_date.strftime('%Y-%m-%d') if hasattr(transaction_date, 'strftime') else str(transaction_date)
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO stock_transactions 
                (drug_id, drug_name, location_key, transaction_type, quantity, transaction_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (drug_id, drug_name, location_key, transaction_type, quantity, date_str, notes))
        return True
    except Exception:
        return False


def get_transactions_for_drug(drug_id: str = None, drug_name: str = None, 
                              limit: int = 100) -> pd.DataFrame:
    """Get recent transactions, optionally filtered by drug."""
    with get_connection() as conn:
        if drug_id:
            rows = conn.execute(
                """SELECT * FROM stock_transactions 
                   WHERE drug_id = ? ORDER BY transaction_date DESC, created_at DESC LIMIT ?""",
                (drug_id, limit)
            ).fetchall()
        elif drug_name:
            rows = conn.execute(
                """SELECT * FROM stock_transactions 
                   WHERE drug_name = ? ORDER BY transaction_date DESC, created_at DESC LIMIT ?""",
                (drug_name, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM stock_transactions 
                   ORDER BY transaction_date DESC, created_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
    
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_net_transactions_by_drug(df_csv: pd.DataFrame) -> dict:
    """
    Compute net stock change from transactions for each drug_id.
    Returns dict: drug_id -> net_change (positive = more stock from restocks - purchases)
    """
    if not os.path.exists(DB_PATH):
        return {}
    
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT drug_id, transaction_type, SUM(quantity) as total FROM stock_transactions GROUP BY drug_id, transaction_type"
        ).fetchall()
    
    net = {}
    for r in rows:
        did = r['drug_id']
        if did not in net:
            net[did] = 0
        if r['transaction_type'] == 'restock':
            net[did] += r['total']
        else:
            net[did] -= r['total']
    
    return net


def get_purchase_demand_by_week(drug_name: str = None, drug_id: str = None, days: int = 365) -> pd.DataFrame:
    """
    Get weekly aggregated purchase quantities (demand) for the last `days` days.
    Returns DataFrame with DatetimeIndex (week end) and 'y' column (quantity).
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    cutoff = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    
    with get_connection() as conn:
        if drug_id:
            rows = conn.execute(
                """SELECT transaction_date, quantity FROM stock_transactions 
                   WHERE drug_id = ? AND transaction_type = 'purchase' AND transaction_date >= ?""",
                (drug_id, cutoff)
            ).fetchall()
        elif drug_name:
            rows = conn.execute(
                """SELECT transaction_date, quantity FROM stock_transactions 
                   WHERE drug_name = ? AND transaction_type = 'purchase' AND transaction_date >= ?""",
                (drug_name, cutoff)
            ).fetchall()
        else:
            return pd.DataFrame()
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame([dict(r) for r in rows])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    weekly = df.set_index('transaction_date')['quantity'].resample('W').sum().to_frame('y')
    return weekly


def get_inventory_locations(df: pd.DataFrame, drug_name: str = None) -> list:
    """
    Get list of (drug_id, drug_name, location_key, label) for UI selection.
    Uses facility_type only (no region split). Picks the most recent drug_id per (drug_name, facility_type)
    so Record Stock transactions match the same drug_id the branch stock display uses.
    """
    if 'drug_id' not in df.columns or 'drug_name' not in df.columns or 'facility_type' not in df.columns:
        return []
    
    sub = df.copy()
    if drug_name:
        sub = sub[sub['drug_name'] == drug_name]
    
    # Use facility_type only (no Northern/Eastern/Western/Central)
    # Most recent row per (drug_name, facility_type) - same logic as get_branch_stock_data
    if 'stock_received_date' in sub.columns:
        sub = sub.copy()
        sub['_sort_date'] = pd.to_datetime(sub['stock_received_date'], errors='coerce')
        sub = sub.sort_values('_sort_date', ascending=False, na_position='last')
    sub = sub.drop_duplicates(subset=['drug_name', 'facility_type'], keep='first')
    
    sub['location_key'] = sub['facility_type'].astype(str)
    sub['label'] = sub['drug_name'] + ' — ' + sub['location_key']
    return sub[['drug_id', 'drug_name', 'location_key', 'label']].to_dict('records')
