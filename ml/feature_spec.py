"""Feature contract for MedSupply ML workflows.

This module is the single source of truth for:
- target definitions
- allowed feature columns
- columns forbidden due to leakage / non-causal availability
"""

TARGET_COLUMN = "average_monthly_demand"
STOCKOUT_TARGET_COLUMN = "stockout_occurred"
EXPIRY_TARGET_COLUMN = "expiry_rate_percent"

TIME_COLUMN = "stock_received_date"

ID_COLUMNS = [
	"drug_id",
	"drug_name",
	"distribution_region",
	"facility_type",
]

NUMERIC_FEATURES = [
	"initial_stock_units",
	"reorder_level",
	"delivery_frequency_days",
	"lead_time_days",
	"supplier_reliability_score",
	"region_disease_outbreaks",
	"transport_accessibility_score",
	"power_stability_index",
	"staff_availability_index",
	"storage_temperature",
	"storage_humidity",
	"FEFO_policy_implemented",
	"warehouse_capacity_utilization",
	"delivery_delay_days",
	"year",
	"month",
	"iso_week",
	"lag_1",
	"lag_2",
	"rolling_mean_3",
	"rolling_std_3",
	"demand_growth_rate",
]

CATEGORICAL_FEATURES = [
	"distribution_region",
	"facility_type",
	"season",
	"data_record_quality",
	"storage_condition_rating",
	"manufacturer_country",
]

FORBIDDEN_COLUMNS = [
	"average_monthly_demand",
	"stockout_occurred",
	"expiry_rate_percent",
	"predicted_stockout_probability",
	"forecast_error_percent",
	"financial_loss_due_to_expiry_usd",
	"expiry_risk_category",
	"composite_risk_score",
	"inventory_turnover",
	"service_level_estimate",
]

DEMAND_FORECAST_FEATURES = [
	"initial_stock_units",
	"reorder_level",
	"delivery_frequency_days",
	"lead_time_days",
	"supplier_reliability_score",
	"region_disease_outbreaks",
	"transport_accessibility_score",
	"power_stability_index",
	"staff_availability_index",
	"storage_temperature",
	"storage_humidity",
	"FEFO_policy_implemented",
	"warehouse_capacity_utilization",
	"delivery_delay_days",
	"year",
	"month",
	"iso_week",
	"lag_1",
	"lag_2",
	"rolling_mean_3",
	"rolling_std_3",
	"demand_growth_rate",
	"distribution_region",
	"facility_type",
	"season",
]

STOCKOUT_FEATURES = [
	"initial_stock_units",
	"reorder_level",
	"delivery_frequency_days",
	"lead_time_days",
	"supplier_reliability_score",
	"region_disease_outbreaks",
	"transport_accessibility_score",
	"power_stability_index",
	"staff_availability_index",
	"FEFO_policy_implemented",
	"warehouse_capacity_utilization",
	"delivery_delay_days",
	"lag_1",
	"lag_2",
	"rolling_mean_3",
	"rolling_std_3",
	"demand_growth_rate",
	"distribution_region",
	"facility_type",
	"season",
	"storage_condition_rating",
]

EXPIRY_RISK_FEATURES = [
	"initial_stock_units",
	"reorder_level",
	"delivery_frequency_days",
	"lead_time_days",
	"supplier_reliability_score",
	"transport_accessibility_score",
	"power_stability_index",
	"staff_availability_index",
	"storage_temperature",
	"storage_humidity",
	"FEFO_policy_implemented",
	"warehouse_capacity_utilization",
	"delivery_delay_days",
	"distribution_region",
	"facility_type",
	"season",
	"storage_condition_rating",
	"data_record_quality",
]


def allowed_inference_features() -> list[str]:
	"""Return deterministic, contract-compliant demand features."""
	return DEMAND_FORECAST_FEATURES.copy()
