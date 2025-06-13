from dagster import ScheduleDefinition, sensor, SensorEvaluationContext, RunRequest, SkipReason
from .jobs import daily_drift_detection_job, hourly_monitoring_job, manual_drift_check_job
import json
import os
from datetime import datetime

# Daily drift detection schedule (runs every day at 2 AM)
daily_drift_schedule = ScheduleDefinition(
    job=daily_drift_detection_job,
    cron_schedule="0 2 * * *",  # 2 AM every day
    name="daily_drift_detection_schedule",
    description="Run comprehensive drift detection daily at 2 AM",
    tags={
        "team": "ml-ops",
        "schedule_type": "daily",
        "automation": "scheduled"
    }
)

# Hourly monitoring schedule (runs every hour during business hours)
hourly_monitoring_schedule = ScheduleDefinition(
    job=hourly_monitoring_job,
    cron_schedule="0 9-18 * * 1-5",  # Every hour from 9 AM to 6 PM, Monday to Friday
    name="hourly_monitoring_schedule", 
    description="Run hourly monitoring during business hours (9 AM - 6 PM, weekdays)",
    tags={
        "team": "ml-ops",
        "schedule_type": "hourly",
        "automation": "scheduled"
    }
)

@sensor(job=daily_drift_detection_job, minimum_interval_seconds=300)  # Check every 5 minutes
def drift_alert_sensor(context: SensorEvaluationContext):
    """
    Sensor that triggers drift detection when significant changes are detected.
    Monitors for:
    1. Sudden increase in query volume
    2. Error rate spikes  
    3. Manual triggers from monitoring systems
    """
    
    try:
        # Try different paths for cache file
        cache_paths = [
            '/data/data/cache/user_queries_finetuned_gradio.json',
            './data/cache/user_queries_test.json',
            '/app/data/cache/user_queries_test.json'
        ]
        
        cache_path = None
        for path in cache_paths:
            if os.path.exists(path):
                cache_path = path
                break
        
        # Check if recent activity warrants drift check
        if cache_path:
            with open(cache_path, 'r', encoding='utf-8') as f:
                recent_data = json.load(f)
            
            # Count queries in last hour
            current_time = datetime.now()
            recent_queries = 0
            error_queries = 0
            
            for entry in recent_data[-100:]:  # Check last 100 entries
                try:
                    entry_time = datetime.strptime(entry['time'], "%Y-%m-%d %H:%M:%S")
                    time_diff = (current_time - entry_time).total_seconds()
                    
                    if time_diff <= 3600:  # Last hour
                        recent_queries += 1
                        # Check for error indicators
                        results = entry.get('results', [])
                        if not results or len(results) == 0:
                            error_queries += 1
                        
                except Exception:
                    continue
            
            # Trigger if high activity (more than 15 queries in last hour)
            if recent_queries > 15:
                context.log.info(f"High activity detected: {recent_queries} queries in last hour")
                return RunRequest(
                    run_key=f"drift_alert_volume_{current_time.strftime('%Y%m%d_%H%M%S')}",
                    tags={
                        "trigger": "high_activity",
                        "query_count": str(recent_queries),
                        "alert_type": "volume_spike",
                        "priority": "medium"
                    }
                )
            
            # Trigger if high error rate (more than 30% of queries return no results)
            if recent_queries > 5 and (error_queries / recent_queries) > 0.3:
                context.log.warning(f"High error rate detected: {error_queries}/{recent_queries} queries with no results")
                return RunRequest(
                    run_key=f"drift_alert_errors_{current_time.strftime('%Y%m%d_%H%M%S')}",
                    tags={
                        "trigger": "high_error_rate",
                        "error_rate": f"{(error_queries/recent_queries)*100:.1f}%",
                        "alert_type": "quality_degradation",
                        "priority": "high"
                    }
                )
        
        # Check for manual trigger files
        trigger_paths = [
            '/data/data/cache/drift_check_trigger.txt',
            './data/cache/drift_check_trigger.txt',
            '/app/data/cache/drift_check_trigger.txt'
        ]
        
        for trigger_path in trigger_paths:
            if os.path.exists(trigger_path):
                try:
                    with open(trigger_path, 'r') as f:
                        trigger_reason = f.read().strip()
                    
                    # Remove trigger file
                    os.remove(trigger_path)
                    
                    context.log.info(f"Manual drift check triggered: {trigger_reason}")
                    return RunRequest(
                        run_key=f"manual_trigger_{current_time.strftime('%Y%m%d_%H%M%S')}",
                        tags={
                            "trigger": "manual",
                            "reason": trigger_reason,
                            "alert_type": "manual_check",
                            "priority": "urgent"
                        }
                    )
                except Exception as e:
                    context.log.error(f"Error processing manual trigger: {e}")
        
        return SkipReason("No significant activity or triggers detected")
        
    except Exception as e:
        context.log.error(f"Error in drift alert sensor: {e}")
        return SkipReason(f"Sensor error: {str(e)}")

@sensor(job=hourly_monitoring_job, minimum_interval_seconds=600)  # Check every 10 minutes
def performance_monitoring_sensor(context: SensorEvaluationContext):
    """
    Sensor that monitors system performance and triggers checks when needed.
    Integrates with Prometheus metrics if available.
    """
    
    try:
        current_time = datetime.now()
        
        # Check for performance degradation indicators
        # This could be extended to check Prometheus metrics
        
        # Check if it's business hours (9 AM - 6 PM, weekdays)
        if current_time.weekday() < 5 and 9 <= current_time.hour <= 18:
            
            # Check for performance trigger files
            perf_trigger_paths = [
                '/data/data/cache/performance_check_trigger.txt',
                './data/cache/performance_check_trigger.txt',
                '/app/data/cache/performance_check_trigger.txt'
            ]
            
            for trigger_path in perf_trigger_paths:
                if os.path.exists(trigger_path):
                    try:
                        with open(trigger_path, 'r') as f:
                            trigger_reason = f.read().strip()
                        
                        # Remove trigger file
                        os.remove(trigger_path)
                        
                        context.log.info(f"Performance monitoring triggered: {trigger_reason}")
                        return RunRequest(
                            run_key=f"performance_trigger_{current_time.strftime('%Y%m%d_%H%M%S')}",
                            tags={
                                "trigger": "performance",
                                "reason": trigger_reason,
                                "alert_type": "performance_check",
                                "priority": "medium"
                            }
                        )
                    except Exception as e:
                        context.log.error(f"Error processing performance trigger: {e}")
            
            # Check for sustained high activity (every 2 hours during business hours)
            if current_time.hour % 2 == 0 and current_time.minute < 10:
                cache_paths = [
                    '/data/data/cache/user_queries_finetuned_gradio.json',
                    './data/cache/user_queries_test.json',
                    '/app/data/cache/user_queries_test.json'
                ]
                
                for cache_path in cache_paths:
                    if os.path.exists(cache_path):
                        try:
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # Count queries in last 2 hours
                            cutoff_time = current_time.replace(hour=current_time.hour-2, minute=0, second=0, microsecond=0)
                            recent_count = 0
                            
                            for entry in data[-200:]:  # Check last 200 entries
                                try:
                                    entry_time = datetime.strptime(entry['time'], "%Y-%m-%d %H:%M:%S")
                                    if entry_time >= cutoff_time:
                                        recent_count += 1
                                except:
                                    continue
                            
                            # Trigger if sustained activity (>30 queries in 2 hours)
                            if recent_count > 30:
                                context.log.info(f"Sustained high activity: {recent_count} queries in last 2 hours")
                                return RunRequest(
                                    run_key=f"sustained_activity_{current_time.strftime('%Y%m%d_%H%M')}",
                                    tags={
                                        "trigger": "sustained_activity",
                                        "query_count_2h": str(recent_count),
                                        "alert_type": "activity_monitoring",
                                        "priority": "low"
                                    }
                                )
                            break
                        except Exception as e:
                            context.log.warning(f"Error checking sustained activity: {e}")
                            continue
        
        return SkipReason("No performance triggers detected or outside business hours")
        
    except Exception as e:
        context.log.error(f"Error in performance monitoring sensor: {e}")
        return SkipReason(f"Performance sensor error: {str(e)}")

@sensor(job=manual_drift_check_job, minimum_interval_seconds=60)  # Check every minute
def critical_alert_sensor(context: SensorEvaluationContext):
    """
    High-priority sensor for critical system alerts that require immediate drift analysis.
    """
    
    try:
        current_time = datetime.now()
        
        # Check for critical alert files
        critical_paths = [
            '/data/data/cache/critical_drift_alert.txt',
            './data/cache/critical_drift_alert.txt',
            '/app/data/cache/critical_drift_alert.txt'
        ]
        
        for alert_path in critical_paths:
            if os.path.exists(alert_path):
                try:
                    with open(alert_path, 'r') as f:
                        alert_data = f.read().strip()
                    
                    # Remove alert file
                    os.remove(alert_path)
                    
                    context.log.warning(f"Critical drift alert triggered: {alert_data}")
                    return RunRequest(
                        run_key=f"critical_alert_{current_time.strftime('%Y%m%d_%H%M%S')}",
                        tags={
                            "trigger": "critical_alert",
                            "alert_data": alert_data,
                            "alert_type": "critical_system_alert",
                            "priority": "critical"
                        }
                    )
                except Exception as e:
                    context.log.error(f"Error processing critical alert: {e}")
        
        # Check for system errors in recent logs (if available)
        # This could be extended to integrate with external monitoring systems
        
        return SkipReason("No critical alerts detected")
        
    except Exception as e:
        context.log.error(f"Error in critical alert sensor: {e}")
        return SkipReason(f"Critical sensor error: {str(e)}")