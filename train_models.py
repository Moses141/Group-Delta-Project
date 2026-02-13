"""
Pre-train SARIMA models for all medications and save them for faster dashboard loading.
This script can be run periodically to update models with new data.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
except ImportError:
    print("statsmodels not installed. Please install it: pip install statsmodels")
    exit(1)

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

def train_sarima_model(series, train_size=0.8):
    """Train SARIMA model on the time series"""
    n = len(series)
    if n < 20:
        return None, None, None
    
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
        print(f"  Error training model: {e}")
        return None, None, None

def main():
    print("=" * 60)
    print("SARIMA Model Training Script")
    print("=" * 60)
    
    # Load data
    data_file = "uganda_drug_supply_synthetic.csv"
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        return
    
    print(f"\nLoading data from {data_file}...")
    df = pd.read_csv(data_file)
    df['stock_received_date'] = pd.to_datetime(df['stock_received_date'], errors='coerce')
    print(f"Loaded {len(df)} records")
    
    # Get medications
    if 'drug_name' not in df.columns:
        print("Error: 'drug_name' column not found in data!")
        return
    
    medications = sorted(df['drug_name'].unique().tolist())
    print(f"\nFound {len(medications)} medications")
    
    # Create models directory
    models_dir = "trained_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Train models for each medication
    trained_count = 0
    failed_count = 0
    models_info = {}
    
    print("\nTraining models...")
    print("-" * 60)
    
    for i, med in enumerate(medications, 1):
        print(f"[{i}/{len(medications)}] Training model for: {med}")
        
        try:
            # Prepare time series
            series, apply_log = prepare_time_series(df, med)
            
            if len(series) < 20:
                print(f"  Insufficient data (only {len(series)} points)")
                failed_count += 1
                continue
            
            # Train model
            model, train_series, train_size_idx = train_sarima_model(series)
            
            if model is None:
                print(f" Model training failed")
                failed_count += 1
                continue
            
            # Save model
            model_file = os.path.join(models_dir, f"{med.replace('/', '_').replace(' ', '_')}.pkl")
            model_data = {
                'model': model,
                'series': series,
                'apply_log': apply_log,
                'train_size_idx': train_size_idx,
                'trained_date': datetime.now().isoformat(),
                'medication': med
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f" Model saved to {model_file}")
            trained_count += 1
            
            # Store info
            models_info[med] = {
                'file': model_file,
                'data_points': len(series),
                'train_points': train_size_idx,
                'apply_log': apply_log
            }
            
        except Exception as e:
            print(f"   Error: {e}")
            failed_count += 1
    
    # Save models index
    index_file = os.path.join(models_dir, "models_index.pkl")
    with open(index_file, 'wb') as f:
        pickle.dump(models_info, f)
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f" Successfully trained: {trained_count} models")
    print(f" Failed: {failed_count} models")
    print(f" Models saved to: {models_dir}/")
    print(f" Index saved to: {index_file}")
    print("\nYou can now use these pre-trained models in the dashboard for faster loading!")

if __name__ == "__main__":
    main()

