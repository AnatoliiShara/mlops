from dagster import define_asset_job
from .assets import (
    reference_data,
    current_queries, 
    input_drift_report,
    output_drift_report,
    drift_summary_report
)

# Job for daily comprehensive drift detection
daily_drift_detection_job = define_asset_job(
    name="daily_drift_detection",
    description="Comprehensive daily drift detection for book recommendation system",
    selection=[
        reference_data,
        current_queries,
        input_drift_report, 
        output_drift_report,
        drift_summary_report
    ],
    tags={
        "team": "ml-ops",
        "priority": "high",
        "type": "drift_detection",
        "schedule": "daily"
    }
)

# Job for hourly monitoring (lighter checks)
hourly_monitoring_job = define_asset_job(
    name="hourly_monitoring",
    description="Hourly monitoring of recent queries and basic drift indicators",
    selection=[
        current_queries,
        input_drift_report
    ],
    tags={
        "team": "ml-ops", 
        "priority": "medium",
        "type": "monitoring",
        "schedule": "hourly"
    }
)

# Job for manual/on-demand drift detection
manual_drift_check_job = define_asset_job(
    name="manual_drift_check",
    description="Manual drift detection triggered by alerts or on-demand",
    selection=[
        reference_data,
        current_queries,
        input_drift_report,
        output_drift_report,
        drift_summary_report
    ],
    tags={
        "team": "ml-ops",
        "priority": "urgent",
        "type": "manual_check"
    }
)