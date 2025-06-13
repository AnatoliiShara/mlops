from dagster import asset, AssetExecutionContext, MetadataValue
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys

# Add the parent directory to sys.path to import drift_detection
sys.path.append('/app')
try:
    from drift_detection.detector import DriftDetector, DriftReporter
except ImportError:
    # Fallback for local development
    sys.path.append('.')
    from drift_detection.detector import DriftDetector, DriftReporter

@asset(group_name="data_collection")
def reference_data(context: AssetExecutionContext) -> Dict[str, Any]:
    """
    Load reference data for drift comparison.
    This represents the "baseline" behavior of the system.
    """
    # Try both Docker and local paths
    project_root = os.getenv('PROJECT_ROOT', '/data')
    cache_paths = [
        os.path.join(project_root, 'data/cache/user_queries_finetuned_gradio.json'),
        './data/cache/user_queries_test.json',
        '/app/data/cache/user_queries_test.json'
    ]
    
    cache_path = None
    for path in cache_paths:
        if os.path.exists(path):
            cache_path = path
            break
    
    reference_data = {
        "queries": [],
        "results": [],
        "timestamp": datetime.now().isoformat(),
        "period": "reference"
    }
    
    try:
        # Load historical data as reference (older than 7 days)
        if cache_path:
            context.log.info(f"Loading reference data from: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                historical_data = json.load(f)
            
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for entry in historical_data:
                try:
                    entry_date = datetime.strptime(entry['time'], "%Y-%m-%d %H:%M:%S")
                    if entry_date < cutoff_date:
                        reference_data["queries"].append(entry['user_query'])
                        reference_data["results"].append(entry.get('results', []))
                except Exception as e:
                    context.log.warning(f"Error parsing entry: {e}")
                    continue
        
        # If no reference data, create synthetic reference with Ukrainian focus
        if not reference_data["queries"]:
            context.log.warning("No historical reference data found, using synthetic Ukrainian data")
            reference_data["queries"] = [
                "фентезі книга про дракона та магію",
                "детективний роман у стилі Агати Крісті",
                "наукова фантастика про космічні подорожі",
                "романтична історія з щасливим кінцем",
                "історична книга про середньовіччя",
                "трилер з неочікуваним фіналом",
                "українська класична література",
                "книга про війну та мир",
                "біографія відомої особистості",
                "кулінарна книга з простими рецептами",
                "філософська література про сенс життя",
                "дитяча книга про пригоди",
                "поезія сучасних авторів",
                "книга по саморозвитку",
                "історія України"
            ]
            reference_data["results"] = [
                [{"id": i, "title": f"Reference Book {i}", "genre": "various", "description": "Sample"}] 
                for i in range(len(reference_data["queries"]))
            ]
        
        context.add_output_metadata({
            "num_reference_queries": MetadataValue.int(len(reference_data["queries"])),
            "num_reference_results": MetadataValue.int(len(reference_data["results"])),
            "reference_period": MetadataValue.text("Data older than 7 days"),
            "sample_queries": MetadataValue.json(reference_data["queries"][:3]),
            "data_source": MetadataValue.text(cache_path or "synthetic")
        })
        
        context.log.info(f"Loaded {len(reference_data['queries'])} reference queries")
        return reference_data
        
    except Exception as e:
        context.log.error(f"Error loading reference data: {e}")
        # Return minimal reference data on error
        return {
            "queries": ["фентезі книга", "детективна історія", "наукова фантастика"],
            "results": [[], [], []],
            "timestamp": datetime.now().isoformat(),
            "period": "reference_fallback"
        }

@asset(group_name="data_collection", deps=[reference_data])
def current_queries(context: AssetExecutionContext) -> Dict[str, Any]:
    """
    Load current/recent data for drift comparison.
    This represents the current behavior of the system.
    """
    # Try both Docker and local paths
    project_root = os.getenv('PROJECT_ROOT', '/data')
    cache_paths = [
        os.path.join(project_root, 'data/cache/user_queries_finetuned_gradio.json'),
        './data/cache/user_queries_test.json',
        '/app/data/cache/user_queries_test.json'
    ]
    
    cache_path = None
    for path in cache_paths:
        if os.path.exists(path):
            cache_path = path
            break
    
    current_data = {
        "queries": [],
        "results": [],
        "timestamp": datetime.now().isoformat(),
        "period": "current"
    }
    
    try:
        if cache_path:
            context.log.info(f"Loading current data from: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                recent_data = json.load(f)
            
            # Get data from last 24 hours
            cutoff_date = datetime.now() - timedelta(hours=24)
            
            for entry in recent_data:
                try:
                    entry_date = datetime.strptime(entry['time'], "%Y-%m-%d %H:%M:%S")
                    if entry_date >= cutoff_date:
                        current_data["queries"].append(entry['user_query'])
                        current_data["results"].append(entry.get('results', []))
                except Exception as e:
                    context.log.warning(f"Error parsing recent entry: {e}")
                    continue
        
        # If no recent data, create synthetic current data with slight variations for drift detection
        if not current_data["queries"]:
            context.log.warning("No recent data found, using synthetic current data with variations")
            current_data["queries"] = [
                "fantasy novel about magical dragons and wizards",  # English variation
                "mystery detective thriller book",
                "sci-fi space exploration adventure",
                "romance love story novel",
                "historical medieval fiction book",
                "suspense thriller with plot twists",
                "сучасна українська література",  # Different Ukrainian focus
                "книга про сучасну війну",  # Contemporary war theme
                "біографія політичної особистості",  # Political biography
                "кулінарні рецепти здорової їжі",  # Healthy cooking
                "психологічна література",  # Psychology books
                "дитячі казки і оповідання",  # Children's tales
                "сучасна поезія",  # Modern poetry  
                "бізнес книги про успіх",  # Business success
                "подорожі по світу"  # Travel literature
            ]
            current_data["results"] = [
                [{"id": i+100, "title": f"Current Book {i}", "genre": "various", "description": "Sample current"}] 
                for i in range(len(current_data["queries"]))
            ]
        
        context.add_output_metadata({
            "num_current_queries": MetadataValue.int(len(current_data["queries"])),
            "num_current_results": MetadataValue.int(len(current_data["results"])),
            "current_period": MetadataValue.text("Last 24 hours"),
            "sample_queries": MetadataValue.json(current_data["queries"][:3]),
            "data_source": MetadataValue.text(cache_path or "synthetic")
        })
        
        context.log.info(f"Loaded {len(current_data['queries'])} current queries")
        return current_data
        
    except Exception as e:
        context.log.error(f"Error loading current data: {e}")
        return {
            "queries": ["fantasy magic", "detective case", "space story"],
            "results": [[], [], []],
            "timestamp": datetime.now().isoformat(),
            "period": "current_fallback"
        }

@asset(group_name="drift_detection", deps=[reference_data, current_queries])
def input_drift_report(context: AssetExecutionContext, reference_data: Dict, current_queries: Dict) -> Dict[str, Any]:
    """
    Detect drift in input queries using comprehensive drift detection methods.
    """
    detector = DriftDetector()
    
    ref_queries = reference_data["queries"]
    curr_queries = current_queries["queries"]
    
    context.log.info(f"Analyzing input drift: {len(ref_queries)} reference vs {len(curr_queries)} current queries")
    
    try:
        # Perform drift detection
        drift_results = detector.detect_input_drift(ref_queries, curr_queries)
        
        # Count metrics with drift
        drift_count = sum(1 for metric in drift_results.values() if metric.has_drift)
        total_metrics = len(drift_results)
        
        # Create summary
        input_drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_period": reference_data.get("period", "unknown"),
            "current_period": current_queries.get("period", "unknown"),
            "metrics_analyzed": total_metrics,
            "metrics_with_drift": drift_count,
            "drift_ratio": drift_count / total_metrics if total_metrics > 0 else 0,
            "status": "DRIFT_DETECTED" if drift_count > 0 else "NO_DRIFT",
            "drift_details": {
                name: {
                    "has_drift": metric.has_drift,
                    "drift_score": round(metric.drift_score, 4),
                    "threshold": metric.threshold,
                    "confidence": round(metric.confidence, 4),
                    "reference_value": round(metric.reference_value, 4),
                    "current_value": round(metric.current_value, 4)
                }
                for name, metric in drift_results.items()
            }
        }
        
        # Add metadata for Dagster UI
        semantic_metric = drift_results.get('semantic_similarity')
        context.add_output_metadata({
            "drift_detected": MetadataValue.bool(drift_count > 0),
            "drift_ratio": MetadataValue.float(drift_count / total_metrics if total_metrics > 0 else 0),
            "metrics_with_drift": MetadataValue.int(drift_count),
            "total_metrics": MetadataValue.int(total_metrics),
            "semantic_similarity": MetadataValue.float(semantic_metric.current_value if semantic_metric else 0),
            "reference_queries_count": MetadataValue.int(len(ref_queries)),
            "current_queries_count": MetadataValue.int(len(curr_queries)),
            "alert_level": MetadataValue.text("HIGH" if drift_count > 2 else "MEDIUM" if drift_count > 0 else "LOW")
        })
        
        context.log.info(f"Input drift analysis complete: {drift_count}/{total_metrics} metrics show drift")
        return input_drift_summary
        
    except Exception as e:
        context.log.error(f"Error in input drift detection: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "ERROR",
            "error": str(e),
            "drift_details": {}
        }

@asset(group_name="drift_detection", deps=[reference_data, current_queries])
def output_drift_report(context: AssetExecutionContext, reference_data: Dict, current_queries: Dict) -> Dict[str, Any]:
    """
    Detect drift in output recommendations using comprehensive analysis methods.
    """
    detector = DriftDetector()
    
    ref_results = reference_data["results"]
    curr_results = current_queries["results"]
    
    context.log.info(f"Analyzing output drift: {len(ref_results)} reference vs {len(curr_results)} current result sets")
    
    try:
        # Perform drift detection
        drift_results = detector.detect_output_drift(ref_results, curr_results)
        
        # Count metrics with drift
        drift_count = sum(1 for metric in drift_results.values() if metric.has_drift)
        total_metrics = len(drift_results)
        
        # Create summary
        output_drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_period": reference_data.get("period", "unknown"),
            "current_period": current_queries.get("period", "unknown"),
            "metrics_analyzed": total_metrics,
            "metrics_with_drift": drift_count,
            "drift_ratio": drift_count / total_metrics if total_metrics > 0 else 0,
            "status": "DRIFT_DETECTED" if drift_count > 0 else "NO_DRIFT",
            "drift_details": {
                name: {
                    "has_drift": metric.has_drift,
                    "drift_score": round(metric.drift_score, 4),
                    "threshold": metric.threshold,
                    "confidence": round(metric.confidence, 4),
                    "reference_value": round(metric.reference_value, 4),
                    "current_value": round(metric.current_value, 4)
                }
                for name, metric in drift_results.items()
            }
        }
        
        # Add metadata for Dagster UI
        genre_metric = drift_results.get('genre_distribution')
        context.add_output_metadata({
            "drift_detected": MetadataValue.bool(drift_count > 0),
            "drift_ratio": MetadataValue.float(drift_count / total_metrics if total_metrics > 0 else 0),
            "metrics_with_drift": MetadataValue.int(drift_count),
            "total_metrics": MetadataValue.int(total_metrics),
            "genre_drift_score": MetadataValue.float(genre_metric.drift_score if genre_metric else 0),
            "reference_results_count": MetadataValue.int(len(ref_results)),
            "current_results_count": MetadataValue.int(len(curr_results)),
            "alert_level": MetadataValue.text("HIGH" if drift_count > 2 else "MEDIUM" if drift_count > 0 else "LOW")
        })
        
        context.log.info(f"Output drift analysis complete: {drift_count}/{total_metrics} metrics show drift")
        return output_drift_summary
        
    except Exception as e:
        context.log.error(f"Error in output drift detection: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "ERROR",
            "error": str(e),
            "drift_details": {}
        }

@asset(group_name="reporting", deps=[input_drift_report, output_drift_report])
def drift_summary_report(context: AssetExecutionContext, input_drift_report: Dict, output_drift_report: Dict) -> Dict[str, Any]:
    """
    Generate comprehensive drift summary report combining input and output analysis.
    """
    reporter = DriftReporter()
    
    try:
        # Convert reports back to DriftMetrics for the reporter
        input_drift_metrics = {}
        output_drift_metrics = {}
        
        # Create simplified metric objects for the reporter
        for name, details in input_drift_report.get("drift_details", {}).items():
            class SimpleDriftMetric:
                def __init__(self, details):
                    self.metric_name = name
                    self.has_drift = details["has_drift"]
                    self.drift_score = details["drift_score"]
                    self.threshold = details["threshold"]
                    self.confidence = details["confidence"]
                    self.reference_value = details["reference_value"]
                    self.current_value = details["current_value"]
            
            input_drift_metrics[name] = SimpleDriftMetric(details)
        
        for name, details in output_drift_report.get("drift_details", {}).items():
            class SimpleDriftMetric:
                def __init__(self, details):
                    self.metric_name = name
                    self.has_drift = details["has_drift"]
                    self.drift_score = details["drift_score"]
                    self.threshold = details["threshold"]
                    self.confidence = details["confidence"]
                    self.reference_value = details["reference_value"]
                    self.current_value = details["current_value"]
            
            output_drift_metrics[name] = SimpleDriftMetric(details)
        
        # Generate comprehensive report
        summary_report = reporter.generate_summary_report(
            input_drift_metrics, 
            output_drift_metrics,
            datetime.now()
        )
        
        # Add execution metadata
        total_drift_count = (input_drift_report.get("metrics_with_drift", 0) + 
                           output_drift_report.get("metrics_with_drift", 0))
        total_metrics = (input_drift_report.get("metrics_analyzed", 0) + 
                        output_drift_report.get("metrics_analyzed", 0))
        
        alert_level = summary_report["overall_summary"]["alert_level"]
        
        context.add_output_metadata({
            "overall_drift_detected": MetadataValue.bool(total_drift_count > 0),
            "alert_level": MetadataValue.text(alert_level),
            "total_drift_ratio": MetadataValue.float(summary_report["overall_summary"]["drift_ratio"]),
            "input_drift_count": MetadataValue.int(input_drift_report.get("metrics_with_drift", 0)),
            "output_drift_count": MetadataValue.int(output_drift_report.get("metrics_with_drift", 0)),
            "recommendations_count": MetadataValue.int(len(summary_report["recommendations"])),
            "next_check": MetadataValue.text(summary_report["next_check_recommended"]),
            "status": MetadataValue.text(summary_report["overall_summary"]["status"])
        })
        
        context.log.info(f"Drift summary generated: {alert_level} alert level, {total_drift_count} metrics with drift")
        
        # Save report to file for external access
        try:
            # Try different paths for saving
            report_paths = [
                '/data/data/drift_reports',
                './data/drift_reports',
                '/app/data/drift_reports'
            ]
            
            report_dir = None
            for path in report_paths:
                try:
                    os.makedirs(path, exist_ok=True)
                    report_dir = path
                    break
                except:
                    continue
            
            if report_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = os.path.join(report_dir, f"drift_report_{timestamp}.json")
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_report, f, indent=2, ensure_ascii=False)
                    
                context.log.info(f"Drift report saved to: {report_file}")
            else:
                context.log.warning("Could not create drift reports directory")
                
        except Exception as e:
            context.log.warning(f"Could not save drift report to file: {e}")
        
        return summary_report
        
    except Exception as e:
        context.log.error(f"Error generating drift summary: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_summary": {
                "status": "ERROR",
                "error": str(e),
                "alert_level": "CRITICAL",
                "total_metrics_checked": 0,
                "metrics_with_drift": 0,
                "drift_ratio": 0
            },
            "input_drift_analysis": {"metrics_checked": 0, "drift_detected": 0, "details": {}},
            "output_drift_analysis": {"metrics_checked": 0, "drift_detected": 0, "details": {}},
            "recommendations": ["⚠️ Drift detection failed. Check system logs and data availability."]
        }