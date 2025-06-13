from dagster import ConfigurableResource
import psycopg2
import requests
import os
from typing import Dict, Any, Optional
import logging

class PostgresResource(ConfigurableResource):
    """PostgreSQL database resource for Dagster storage and drift data"""
    
    host: str
    port: int
    database: str
    username: str
    password: str
    
    def get_connection(self):
        """Get PostgreSQL connection"""
        try:
            return psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                connect_timeout=10
            )
        except Exception as e:
            logging.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def execute_query(self, query: str, params=None):
        """Execute a query and return results"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    if query.strip().lower().startswith('select'):
                        return cursor.fetchall()
                    conn.commit()
                    return cursor.rowcount
        except Exception as e:
            logging.error(f"Query execution failed: {e}")
            raise

    def store_drift_result(self, drift_data: Dict[str, Any]):
        """Store drift detection results in database"""
        
        # Create table if not exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS drift_results (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            drift_type VARCHAR(50) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            has_drift BOOLEAN NOT NULL,
            drift_score FLOAT NOT NULL,
            threshold_value FLOAT,
            confidence FLOAT,
            details JSONB,
            alert_level VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_results(timestamp);
        CREATE INDEX IF NOT EXISTS idx_drift_type ON drift_results(drift_type);
        CREATE INDEX IF NOT EXISTS idx_drift_alert_level ON drift_results(alert_level);
        """
        
        try:
            self.execute_query(create_table_query)
            
            insert_query = """
            INSERT INTO drift_results (timestamp, drift_type, metric_name, has_drift, drift_score, threshold_value, confidence, details, alert_level)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            timestamp = drift_data.get('timestamp')
            alert_level = drift_data.get('overall_summary', {}).get('alert_level', 'UNKNOWN')
            
            # Store input drift metrics
            for metric_name, metric_data in drift_data.get('input_drift_analysis', {}).get('details', {}).items():
                self.execute_query(insert_query, (
                    timestamp,
                    'input',
                    metric_name,
                    metric_data.get('has_drift', False),
                    metric_data.get('drift_score', 0.0),
                    metric_data.get('threshold', 0.0),
                    metric_data.get('confidence', 0.0),
                    str(metric_data),
                    alert_level
                ))
            
            # Store output drift metrics
            for metric_name, metric_data in drift_data.get('output_drift_analysis', {}).get('details', {}).items():
                self.execute_query(insert_query, (
                    timestamp,
                    'output',
                    metric_name,
                    metric_data.get('has_drift', False),
                    metric_data.get('drift_score', 0.0),
                    metric_data.get('threshold', 0.0),
                    metric_data.get('confidence', 0.0),
                    str(metric_data),
                    alert_level
                ))
                
            logging.info(f"Stored drift results with alert level: {alert_level}")
            
        except Exception as e:
            logging.error(f"Failed to store drift results: {e}")
            raise

    def get_drift_history(self, days: int = 30) -> list:
        """Get drift detection history for the last N days"""
        query = """
        SELECT timestamp, drift_type, metric_name, has_drift, drift_score, alert_level
        FROM drift_results 
        WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
        ORDER BY timestamp DESC
        """
        
        try:
            return self.execute_query(query, (days,))
        except Exception as e:
            logging.error(f"Failed to get drift history: {e}")
            return []

class PrometheusResource(ConfigurableResource):
    """Prometheus integration for metrics collection and alerting"""
    
    base_url: str
    timeout: int = 10
    
    def query_metric(self, query: str) -> Dict[str, Any]:
        """Query Prometheus for metrics"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/query",
                params={"query": query},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Prometheus query failed: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            logging.error(f"Unexpected error in Prometheus query: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_search_metrics(self) -> Dict[str, float]:
        """Get book recommendation system metrics"""
        metrics = {}
        
        metric_queries = {
            "search_latency_p95": 'histogram_quantile(0.95, rate(book_search_duration_seconds_bucket[5m]))',
            "search_latency_p50": 'histogram_quantile(0.50, rate(book_search_duration_seconds_bucket[5m]))',
            "error_rate_per_minute": 'rate(book_errors_total[5m]) * 60',
            "request_rate_per_minute": 'rate(book_search_requests_total[5m]) * 60',
            "user_satisfaction": 'book_user_satisfaction_rate',
            "system_memory_usage": 'book_system_memory_usage_bytes'
        }
        
        for metric_name, query in metric_queries.items():
            try:
                result = self.query_metric(query)
                if result.get("status") == "success":
                    data = result.get("data", {}).get("result", [])
                    if data:
                        metrics[metric_name] = float(data[0]["value"][1])
                    else:
                        metrics[metric_name] = 0.0
                else:
                    metrics[metric_name] = 0.0
            except Exception as e:
                logging.warning(f"Failed to get metric {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        return metrics
    
    def check_alert_conditions(self) -> Dict[str, Any]:
        """Check for alert conditions in Prometheus metrics"""
        alerts = {
            "high_latency": False,
            "high_error_rate": False,
            "low_satisfaction": False,
            "high_memory_usage": False
        }
        
        try:
            metrics = self.get_search_metrics()
            
            # Check alert conditions
            if metrics.get("search_latency_p95", 0) > 5.0:  # 5 seconds
                alerts["high_latency"] = True
                
            if metrics.get("error_rate_per_minute", 0) > 5.0:  # 5 errors per minute
                alerts["high_error_rate"] = True
                
            if metrics.get("user_satisfaction", 1.0) < 0.7:  # Below 70%
                alerts["low_satisfaction"] = True
                
            if metrics.get("system_memory_usage", 0) > 3 * 1024 * 1024 * 1024:  # 3GB
                alerts["high_memory_usage"] = True
                
        except Exception as e:
            logging.error(f"Failed to check alert conditions: {e}")
        
        return alerts
    
    def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert to Prometheus Alertmanager (if configured)"""
        try:
            # This would integrate with Alertmanager API in production
            logging.info(f"Alert would be sent: {alert_data}")
            # Placeholder for actual alerting implementation
        except Exception as e:
            logging.error(f"Failed to send alert: {e}")

class SigNozResource(ConfigurableResource):
    """SigNoz integration for tracing and advanced analytics"""
    
    base_url: str
    timeout: int = 10
    
    def get_trace_metrics(self, time_range: str = "1h") -> Dict[str, Any]:
        """Get trace metrics from SigNoz"""
        try:
            # This would integrate with SigNoz API in production
            # For demonstration, return mock data based on system activity
            
            # Try to get real activity data
            cache_paths = [
                '/data/data/cache/user_queries_finetuned_gradio.json',
                './data/cache/user_queries_test.json',
                '/app/data/cache/user_queries_test.json'
            ]
            
            total_traces = 0
            error_traces = 0
            
            for cache_path in cache_paths:
                if os.path.exists(cache_path):
                    try:
                        import json
                        from datetime import datetime, timedelta
                        
                        with open(cache_path, 'r') as f:
                            data = json.load(f)
                        
                        # Count recent traces (last hour)
                        cutoff = datetime.now() - timedelta(hours=1)
                        for entry in data[-100:]:
                            try:
                                entry_time = datetime.strptime(entry['time'], "%Y-%m-%d %H:%M:%S")
                                if entry_time >= cutoff:
                                    total_traces += 1
                                    if not entry.get('results') or len(entry.get('results', [])) == 0:
                                        error_traces += 1
                            except:
                                continue
                        break
                    except:
                        continue
            
            return {
                "total_traces": total_traces,
                "error_traces": error_traces,
                "error_rate": error_traces / total_traces if total_traces > 0 else 0,
                "avg_latency": 1.2,  # Mock value
                "services": ["book-recommendation-system"],
                "time_range": time_range
            }
            
        except Exception as e:
            logging.error(f"Failed to get SigNoz trace metrics: {e}")
            return {
                "total_traces": 0,
                "error_traces": 0,
                "error_rate": 0,
                "avg_latency": 0,
                "services": [],
                "status": "error",
                "error": str(e)
            }
    
    def analyze_trace_patterns(self) -> Dict[str, Any]:
        """Analyze trace patterns for anomalies"""
        try:
            trace_metrics = self.get_trace_metrics()
            
            # Simple anomaly detection based on trace data
            anomalies_detected = False
            if trace_metrics.get("error_rate", 0) > 0.2:  # >20% error rate
                anomalies_detected = True
            
            performance_trend = "stable"
            if trace_metrics.get("avg_latency", 0) > 3.0:
                performance_trend = "degrading"
            elif trace_metrics.get("avg_latency", 0) < 0.5:
                performance_trend = "improving"
            
            error_patterns = []
            if trace_metrics.get("error_rate", 0) > 0.1:
                error_patterns.append("High error rate detected")
            
            return {
                "anomalies_detected": anomalies_detected,
                "performance_trends": performance_trend,
                "error_patterns": error_patterns,
                "trace_analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze trace patterns: {e}")
            return {
                "anomalies_detected": True,
                "performance_trends": "unknown",
                "error_patterns": ["Analysis failed"],
                "error": str(e)
            }

# Resource configurations for different environments
postgres_resource = PostgresResource(
    host=os.getenv("DAGSTER_POSTGRES_HOST", "localhost"),
    port=int(os.getenv("DAGSTER_POSTGRES_PORT", "5432")),
    database=os.getenv("DAGSTER_POSTGRES_DB", "dagster"),
    username=os.getenv("DAGSTER_POSTGRES_USER", "dagster"),
    password=os.getenv("DAGSTER_POSTGRES_PASSWORD", "dagster")
)

prometheus_resource = PrometheusResource(
    base_url=os.getenv("PROMETHEUS_URL", "http://prometheus:9090"),
    timeout=15
)

signoz_resource = SigNozResource(
    base_url=os.getenv("SIGNOZ_URL", "http://signoz:3301"),
    timeout=15
)