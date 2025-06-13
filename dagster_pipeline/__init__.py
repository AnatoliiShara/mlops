from dagster import Definitions
from .assets import (
    reference_data,
    current_queries,
    input_drift_report,
    output_drift_report,
    drift_summary_report
)
from .jobs import daily_drift_detection_job, hourly_monitoring_job
from .schedules import daily_drift_schedule, hourly_monitoring_schedule
from .sensors import drift_alert_sensor, performance_monitoring_sensor
from .resources import postgres_resource, prometheus_resource, signoz_resource

# Define all assets, jobs, schedules, and sensors
defs = Definitions(
    assets=[
        reference_data,
        current_queries,
        input_drift_report,
        output_drift_report,
        drift_summary_report,
    ],
    jobs=[
        daily_drift_detection_job,
        hourly_monitoring_job,
    ],
    schedules=[
        daily_drift_schedule,
        hourly_monitoring_schedule,
    ],
    sensors=[
        drift_alert_sensor,
        performance_monitoring_sensor,
    ],
    resources={
        "postgres": postgres_resource,
        "prometheus": prometheus_resource,
        "signoz": signoz_resource,
    },
)