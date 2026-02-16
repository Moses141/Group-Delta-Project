"""
Shared data preparation for SARIMA models.
Merges CSV data with database purchase data and filters to last N days.
"""
import pandas as pd
import numpy as np
import os


def prepare_time_series(df, medication, target_col='average_monthly_demand', time_col='stock_received_date',
                       lookback_days=365, use_db=True):
    """
    Prepare demand time series for a medication.
    - Filters to last lookback_days (default 365)
    - Merges CSV data with DB purchase data (purchases = actual demand)
    - Prefers DB purchase data when available for a given week
    """
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    
    # Get CSV data filtered to last N days
    df_med = df[df['drug_name'] == medication].copy() if 'drug_name' in df.columns else df.copy()
    df_med = df_med[df_med[time_col].notna()]
    df_med = df_med[df_med[time_col] >= cutoff]
    df_med = df_med.sort_values(time_col).reset_index(drop=True)
    
    # Resample CSV to weekly
    if len(df_med) > 0:
        csv_weekly = df_med.set_index(time_col)[target_col].resample("W").mean().to_frame("y")
    else:
        csv_weekly = pd.DataFrame(columns=['y'])
    
    # Get DB purchase data (actual demand) for last N days
    db_weekly = pd.DataFrame()
    if use_db:
        try:
            from db_pharmacy import get_purchase_demand_by_week
            db_weekly = get_purchase_demand_by_week(drug_name=medication, days=lookback_days)
        except Exception:
            pass
    
    # Merge: prefer DB purchases (actual demand) when available, else CSV
    if len(db_weekly) > 0 and len(csv_weekly) > 0:
        all_weeks = csv_weekly.index.union(db_weekly.index).unique()
        combined = pd.DataFrame(index=sorted(all_weeks), columns=['y'])
        for idx in combined.index:
            if idx in db_weekly.index and pd.notna(db_weekly.loc[idx, 'y']) and db_weekly.loc[idx, 'y'] > 0:
                combined.loc[idx, 'y'] = db_weekly.loc[idx, 'y']
            elif idx in csv_weekly.index:
                combined.loc[idx, 'y'] = csv_weekly.loc[idx, 'y']
    elif len(db_weekly) > 0:
        combined = db_weekly.copy()
    elif len(csv_weekly) > 0:
        combined = csv_weekly.copy()
    else:
        return pd.DataFrame(), False
    
    # Ensure we only have last lookback_days
    combined = combined[combined.index >= cutoff]
    
    if len(combined) == 0:
        return pd.DataFrame(), False
    
    if len(combined) < 2:
        return combined, False
    
    # Fill gaps (interpolate NaN; keep 0 as valid low-demand)
    combined['y'] = combined['y'].interpolate(limit_direction='both').ffill().bfill()
    combined['y'] = combined['y'].fillna(1)  # avoid zeros for log transform
    
    # Apply log transform if all values positive
    apply_log = (combined['y'] > 0).all()
    if apply_log:
        combined['y'] = np.log1p(combined['y'])
    
    return combined, apply_log
