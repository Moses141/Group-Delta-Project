import pandas as pd

from ml.feature_engineering import add_lag_features
from ml.feature_spec import FORBIDDEN_COLUMNS, TARGET_COLUMN, allowed_inference_features
from ml.preprocessing import select_feature_columns, time_series_split
from ml.train import train_demand_pipeline


def _sample_df(n=12):
    dates = pd.date_range("2024-01-01", periods=n, freq="W")
    return pd.DataFrame({
        "drug_id": ["D1"] * n,
        "drug_name": ["Drug A"] * n,
        "distribution_region": ["Central"] * n,
        "facility_type": ["Hospital"] * n,
        "stock_received_date": dates,
        "initial_stock_units": list(range(100, 100 + n)),
        "reorder_level": [30] * n,
        "average_monthly_demand": list(range(20, 20 + n)),
        "delivery_frequency_days": [30] * n,
        "lead_time_days": [14] * n,
        "supplier_reliability_score": [0.8] * n,
        "region_disease_outbreaks": [0] * n,
        "season": ["Dry"] * n,
        "transport_accessibility_score": [0.7] * n,
        "power_stability_index": [0.9] * n,
        "staff_availability_index": [0.8] * n,
        "storage_temperature": [24.0] * n,
        "storage_humidity": [55.0] * n,
        "FEFO_policy_implemented": [1] * n,
        "warehouse_capacity_utilization": [0.6] * n,
        "storage_condition_rating": ["Good"] * n,
        "delivery_delay_days": [0] * n,
    })


def test_no_target_leakage():
    df = _sample_df()
    features = select_feature_columns(df)
    assert TARGET_COLUMN not in features
    assert all(col not in features for col in FORBIDDEN_COLUMNS)


def test_time_split_order():
    df = _sample_df(20)
    train, test = time_series_split(df, "stock_received_date", test_size=0.2)
    assert train["stock_received_date"].max() <= test["stock_received_date"].min()
    assert len(train) + len(test) == len(df)


def test_feature_shift_correctness():
    df = _sample_df(8)
    out = add_lag_features(df)
    assert pd.isna(out.loc[0, "lag_1"])
    assert out.loc[2, "lag_1"] == out.loc[1, "average_monthly_demand"]
    assert out.loc[3, "rolling_mean_3"] == out.loc[0:2, "average_monthly_demand"].mean()


def test_inference_schema_match():
    df = _sample_df(30)
    trained = train_demand_pipeline(df)
    expected = set(allowed_inference_features())
    actual = set(trained["feature_columns"])
    assert actual.issubset(expected)
