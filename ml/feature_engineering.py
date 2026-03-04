"""Leakage-safe temporal feature engineering."""

from __future__ import annotations

import pandas as pd

from ml.feature_spec import TIME_COLUMN, TARGET_COLUMN


def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TIME_COLUMN] = pd.to_datetime(out[TIME_COLUMN], errors="coerce")
    out["year"] = out[TIME_COLUMN].dt.year
    out["month"] = out[TIME_COLUMN].dt.month
    out["iso_week"] = out[TIME_COLUMN].dt.isocalendar().week.astype("Int64")
    return out


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    out = df.copy()
    out[TIME_COLUMN] = pd.to_datetime(out[TIME_COLUMN], errors="coerce")
    groups = group_cols or ["drug_id", "distribution_region"]
    sort_cols = groups + [TIME_COLUMN]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    grouped = out.groupby(groups)[target_col]
    out["lag_1"] = grouped.shift(1)
    out["lag_2"] = grouped.shift(2)

    shifted = grouped.shift(1)
    out["rolling_mean_3"] = shifted.groupby(out[groups].apply(tuple, axis=1)).transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    out["rolling_std_3"] = shifted.groupby(out[groups].apply(tuple, axis=1)).transform(
        lambda s: s.rolling(window=3, min_periods=1).std()
    )

    out["demand_growth_rate"] = (out["lag_1"] - out["lag_2"]) / out["lag_2"].replace(0, pd.NA)

    out["rolling_std_3"] = out["rolling_std_3"].fillna(0.0)
    out["demand_growth_rate"] = out["demand_growth_rate"].fillna(0.0)
    return out
