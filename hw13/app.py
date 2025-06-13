import gradio as gr
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import os
import json
from datetime import datetime
import nltk
import time
import logging
from functools import wraps
import threading

# === OpenTelemetry & SigNoz ===
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import psutil

# === Prometheus Metrics ===
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from flask import Flask, Response

# Download NLTK data
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

# === PROMETHEUS METRICS SETUP ===
# Counters
prometheus_search_requests = Counter(
    'book_search_requests_total',
    'Total number of search requests',
    ['query_length_bucket', 'results_found_bucket']
)

prometheus_feedback_submissions = Counter(
    'book_feedback_submissions_total',
    'Total feedback submissions',
    ['satisfied', 'has_additional_feedback']
)

prometheus_errors = Counter(
    'book_errors_total',
    'Total number of errors',
    ['operation', 'error_type']
)

# Histograms
prometheus_search_duration = Histogram(
    'book_search_duration_seconds',
    'Duration of search operations',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

prometheus_bm25_duration = Histogram(
    'book_bm25_search_duration_seconds',
    'Duration of BM25 search',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

prometheus_faiss_duration = Histogram(
    'book_faiss_search_duration_seconds',
    'Duration of FAISS search',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

prometheus_rerank_duration = Histogram(
    'book_rerank_duration_seconds',
    'Duration of CrossEncoder reranking',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

prometheus_results_count = Histogram(
    'book_search_results_count',
    'Number of results returned',
    buckets=[0, 1, 3, 5, 7, 10]
)

# Gauges
prometheus_system_memory = Gauge(
    'book_system_memory_usage_bytes',
    'System memory usage in bytes'
)

prometheus_user_satisfaction = Gauge(
    'book_user_satisfaction_rate',
    'User satisfaction rate (0-1)'
)

prometheus_active_sessions = Gauge(
    'book_active_sessions',
    'Number of active user sessions'
)

# === MONITORING SETUP ===
def setup_telemetry():
    """Initialize OpenTelemetry for SigNoz"""
    
    # Resource identification
    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "book-recommendation-system"),
        "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
        "service.instance.id": f"instance-{os.getpid()}",
        "deployment.environment": "development"
    })
    
    # Tracing setup
    trace_provider = TracerProvider(resource=resource)
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        insecure=True
    )
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(trace_provider)
    
    # Metrics setup  
    metric_reader = PeriodicExportingMetricReader(
        exporter=OTLPMetricExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            insecure=True
        ),
        export_interval_millis=5000
    )
    metrics.set_meter_provider(MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    ))
    
    # Auto-instrumentation
    LoggingInstrumentor().instrument(set_logging_format=True)
    RequestsInstrumentor().instrument()
    
    print("✅ OpenTelemetry configured for SigNoz")
    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Initialize telemetry
tracer, meter = setup_telemetry()

# === CUSTOM OTEL METRICS ===
# Counters
search_requests_counter = meter.create_counter(
    name="search_requests_total",
    description="Total number of search requests",
    unit="1"
)

feedback_counter = meter.create_counter(
    name="feedback_submissions_total", 
    description="Total feedback submissions",
    unit="1"
)

errors_counter = meter.create_counter(
    name="errors_total",
    description="Total number of errors",
    unit="1"
)

# Histograms
search_duration_histogram = meter.create_histogram(
    name="search_duration_seconds",
    description="Duration of search operations",
    unit="s"
)

bm25_duration_histogram = meter.create_histogram(
    name="bm25_search_duration_seconds",
    description="Duration of BM25 search",
    unit="s"
)

faiss_duration_histogram = meter.create_histogram(
    name="faiss_search_duration_seconds", 
    description="Duration of FAISS search",
    unit="s"
)

rerank_duration_histogram = meter.create_histogram(
    name="rerank_duration_seconds",
    description="Duration of CrossEncoder reranking",
    unit="s"
)

# Gauges  
system_memory_gauge = meter.create_gauge(
    name="system_memory_usage_bytes",
    description="System memory usage in bytes"
)

results_count_histogram = meter.create_histogram(
    name="search_results_count",
    description="Number of results returned",
    unit="1"
)

user_satisfaction_gauge = meter.create_gauge(
    name="user_satisfaction_rate",
    description="User satisfaction rate",
    unit="1"
)

# === PROMETHEUS HTTP SERVER ===
def start_prometheus_server():
    """Start Prometheus metrics HTTP server"""
    port = int(os.getenv("PROMETHEUS_PORT", 8080))
    try:
        start_http_server(port)
        print(f"✅ Prometheus metrics server started on port {port}")
    except Exception as e:
        print(f"⚠️ Failed to start Prometheus server: {e}")

# === TRACING DECORATORS ===
def trace_function(span_name, record_exception=True):
    """Decorator to trace functions with dual monitoring"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add function attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.args_count", len(args))
                    
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    span.set_attribute("function.duration", duration)
                    span.set_attribute("function.success", True)
                    
                    return result
                    
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                        span.set_attribute("function.success", False)
                        span.set_attribute("error.message", str(e))
                        
                        # Record in both systems
                        errors_counter.add(1, {"function": func.__name__, "error_type": type(e).__name__})
                        prometheus_errors.labels(operation=func.__name__, error_type=type(e).__name__).inc()
                    raise
        return wrapper
    return decorator

# === PATHS FOR DATA ===
PROJECT_ROOT = os.getenv('PROJECT_ROOT', '/data')

# Check if we're running in Docker with mounted data or local development
if os.path.exists('/data'):
    # Docker environment with mounted data
    CSV_PATH = os.path.join(PROJECT_ROOT, 'data/processed/bookye_books_10000_with_id.csv')
    INDEX_PATH = os.path.join(PROJECT_ROOT, 'data/embeddings/faiss_index_finetuned.idx')
    EMB_PATH = os.path.join(PROJECT_ROOT, 'data/embeddings/bookye_books_10000_embeddings_finetuned.parquet')
    ID_MAPPING_PATH = os.path.join(PROJECT_ROOT, 'data/embeddings/faiss_id_mapping_finetuned.json')
    CACHE_PATH = os.path.join(PROJECT_ROOT, 'data/cache/user_queries_finetuned_gradio.json')
    FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/finetuned/final_model_20250530_075616')
else:
    # Local development - use sample data
    CSV_PATH = './data/sample_books.csv'
    CACHE_PATH = './data/cache/user_queries_test.json'
    INDEX_PATH = None  # Will create mock data
    EMB_PATH = None
    ID_MAPPING_PATH = None
    FINETUNED_MODEL_PATH = None

# === PARAMETERS ===
TOP_K = 5
BM25_N = 30
FAISS_N = 30  
RRF_N = 20
CROSS_ENCODER_MODEL = "cross-encoder/stsb-distilroberta-base"

# === GLOBAL VARIABLES ===
data_df = None
faiss_index = None
id_list = None
emb_matrix = None
bm25 = None
bm25_tokenized = None
sbert = None
cross_encoder = None
id_to_idx = None
idx_to_id = None

# === SYSTEM METRICS COLLECTION ===
def collect_system_metrics():
    """Collect system metrics for both monitoring systems"""
    try:
        memory_info = psutil.virtual_memory()
        memory_bytes = memory_info.used
        
        # Update both monitoring systems
        system_memory_gauge.set(memory_bytes)
        prometheus_system_memory.set(memory_bytes)
        
    except Exception as e:
        logging.warning(f"Failed to collect system metrics: {e}")

# === INITIALIZATION ===
@trace_function("system_initialization")
def initialize_system():
    """Initialize all system components with monitoring"""
    global data_df, faiss_index, id_list, emb_matrix, bm25, bm25_tokenized, sbert, cross_encoder, id_to_idx, idx_to_id
    
    with tracer.start_as_current_span("load_data") as span:
        try:
            print("🚀 Initializing components...")
            
            # Load CSV data
            print(f"Loading CSV from: {CSV_PATH}")
            if os.path.exists(CSV_PATH):
                data_df = pd.read_csv(CSV_PATH)
                if 'id' not in data_df.columns:
                    data_df['id'] = range(len(data_df))
                data_df.set_index('id', inplace=True)
                data_df['combined_text'] = data_df['title'].astype(str) + " SEP " + data_df['description'].astype(str) + " SEP " + data_df['genre'].astype(str)
            else:
                # Create sample data if file doesn't exist
                print("⚠️ CSV file not found, creating sample data")
                sample_data = {
                    'id': range(10),
                    'title': [
                        'Гаррі Поттер і філософський камінь',
                        'Шерлок Холмс: Знак чотирьох',
                        'Дюна',
                        'Війна і мир',
                        'Майстер і Маргарита',
                        'Кобзар',
                        'Драгоценная',
                        'Чорний лебідь',
                        'Алхімік',
                        'Три товариша'
                    ],
                    'description': [
                        'Магічна історія про хлопчика-чарівника',
                        'Детективні пригоди великого сищика',
                        'Науково-фантастичний епос про пустельну планету',
                        'Історичний роман про наполеонівські війни',
                        'Містичний роман про диявола в Москві',
                        'Збірка поезій Тараса Шевченка',
                        'Роман про афроамериканку у 1980-х',
                        'Книга про непередбачувані події',
                        'Філософська повість про пастуха',
                        'Історія про дружбу під час війни'
                    ],
                    'genre': [
                        'fantasy',
                        'detective',
                        'science_fiction',
                        'historical',
                        'mystical',
                        'poetry',
                        'drama',
                        'non_fiction',
                        'philosophy',
                        'war'
                    ]
                }
                data_df = pd.DataFrame(sample_data)
                data_df.set_index('id', inplace=True)
                data_df['combined_text'] = data_df['title'].astype(str) + " SEP " + data_df['description'].astype(str) + " SEP " + data_df['genre'].astype(str)
            
            span.set_attribute("data.books_count", len(data_df))
            print(f"✅ Loaded {len(data_df)} books from CSV")
            
            # Create BM25 index (always available)
            print("Creating BM25 index...")
            tokenized = [word_tokenize(str(doc).lower()) for doc in data_df['combined_text'].tolist()]
            bm25 = BM25Okapi(tokenized)
            bm25_tokenized = tokenized
            print("✅ BM25 index created")
            
            # Try to load FAISS and models if available
            if FINETUNED_MODEL_PATH and os.path.exists(FINETUNED_MODEL_PATH):
                print(f"Loading fine-tuned model from: {FINETUNED_MODEL_PATH}")
                sbert = SentenceTransformer(FINETUNED_MODEL_PATH)
                span.set_attribute("model.embedding_dimension", sbert.get_sentence_embedding_dimension())
                print(f"✅ Fine-tuned model loaded")
                
                # Load FAISS index if available
                if INDEX_PATH and os.path.exists(INDEX_PATH):
                    faiss_index = faiss.read_index(INDEX_PATH)
                    span.set_attribute("faiss.vectors_count", faiss_index.ntotal)
                    print(f"✅ FAISS index loaded: {faiss_index.ntotal} vectors")
                    
                    # Load embeddings
                    if EMB_PATH and os.path.exists(EMB_PATH):
                        emb_df = pd.read_parquet(EMB_PATH)
                        emb_cols = [c for c in emb_df.columns if c.startswith('emb')]
                        id_list = emb_df['id'].tolist()
                        emb_matrix = emb_df[emb_cols].values.astype('float32')
                        span.set_attribute("embeddings.dimension", len(emb_cols))
                        print(f"✅ Embeddings loaded: {len(id_list)} vectors")
                        
                        # Load ID mapping
                        if ID_MAPPING_PATH and os.path.exists(ID_MAPPING_PATH):
                            with open(ID_MAPPING_PATH, 'r', encoding='utf-8') as f:
                                mapping_data = json.load(f)
                            id_mapping = {int(k): int(v) for k, v in mapping_data['mapping'].items()}
                            id_to_idx = {v: k for k, v in id_mapping.items()}
                            idx_to_id = id_mapping
                            print(f"✅ ID mapping loaded: {len(id_mapping)} records")
                        else:
                            id_to_idx = {id_: idx for idx, id_ in enumerate(id_list)}
                            idx_to_id = {v: k for k, v in id_to_idx.items()}
                            print("⚠️ Using default ID mapping")
                
                # Load CrossEncoder
                print(f"Loading CrossEncoder: {CROSS_ENCODER_MODEL}")
                cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
                print(f"✅ CrossEncoder loaded")
            else:
                print("⚠️ Full ML pipeline not available, using BM25-only mode")
                sbert = None
                faiss_index = None
                cross_encoder = None
            
            print("🎉 System initialization completed!")
            return True
            
        except Exception as e:
            span.record_exception(e)
            print(f"❌ Initialization error: {e}")
            print(f"Error details: {str(e)}")
            
            # Record errors in both systems
            errors_counter.add(1, {"component": "initialization", "error_type": type(e).__name__})
            prometheus_errors.labels(operation="initialization", error_type=type(e).__name__).inc()
            return False

# === SIMPLIFIED SEARCH FOR DEMO ===
@trace_function("simple_search")
def simple_search(query, top_k=TOP_K):
    """Simplified search using only BM25 (fallback when full pipeline unavailable)"""
    search_start_time = time.time()
    
    try:
        with tracer.start_as_current_span("bm25_only_search") as span:
            span.set_attribute("query.text", query)
            span.set_attribute("query.top_k", top_k)
            
            # BM25 search
            query_tok = word_tokenize(query.lower())
            bm25_scores = bm25.get_scores(query_tok)
            bm25_ranked_idx = np.argsort(bm25_scores)[::-1][:top_k]
            
            # Get results
            results = []
            for idx in bm25_ranked_idx:
                row = data_df.iloc[idx]
                results.append({
                    "id": int(row.name),
                    "title": row['title'],
                    "description": row['description'],
                    "genre": row['genre']
                })
            
            search_time = time.time() - search_start_time
            
            # Record metrics
            search_duration_histogram.record(search_time)
            prometheus_search_duration.observe(search_time)
            
            span.set_attribute("search.total_duration", search_time)
            span.set_attribute("search.results_count", len(results))
            
            return results, search_time
            
    except Exception as e:
        search_time = time.time() - search_start_time
        errors_counter.add(1, {"operation": "simple_search", "error_type": type(e).__name__})
        prometheus_errors.labels(operation="simple_search", error_type=type(e).__name__).inc()
        print(f"❌ Search error: {e}")
        return [], search_time

# === RRF FUSION ===
@trace_function("rrf_fusion")
def rrf_fusion(bm25_ids, faiss_ids, r=60, top_n=RRF_N):
    """Reciprocal Rank Fusion for combining BM25 and FAISS results"""
    with tracer.start_as_current_span("rrf_scoring") as span:
        scores = {}
        for rank, id_ in enumerate(bm25_ids):
            scores[id_] = scores.get(id_, 0) + 1 / (r + rank + 1)
        for rank, id_ in enumerate(faiss_ids):
            scores[id_] = scores.get(id_, 0) + 1 / (r + rank + 1)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        span.set_attribute("rrf.candidates_count", len(scores))
        span.set_attribute("rrf.output_count", min(top_n, len(ranked)))
        
        return [id_ for id_, score in ranked[:top_n]]

# === HYBRID SEARCH PIPELINE ===
@trace_function("hybrid_search") 
def hybrid_search(query, top_k=TOP_K):
    """Hybrid search: BM25 + FAISS + RRF + CrossEncoder (if available)"""
    
    # If full pipeline not available, use simple search
    if not sbert or not faiss_index:
        return simple_search(query, top_k)
    
    search_start_time = time.time()
    
    try:
        with tracer.start_as_current_span("search_pipeline") as span:
            span.set_attribute("query.text", query)
            span.set_attribute("query.top_k", top_k)
            
            # Collect system metrics
            collect_system_metrics()
            
            # 1. BM25 search
            with tracer.start_as_current_span("bm25_search") as bm25_span:
                bm25_start = time.time()
                query_tok = word_tokenize(query.lower())
                bm25_scores = bm25.get_scores(query_tok)
                bm25_ranked_idx = np.argsort(bm25_scores)[::-1][:BM25_N]
                bm25_ids = [data_df.iloc[idx].name for idx in bm25_ranked_idx]
                bm25_duration = time.time() - bm25_start
                
                bm25_span.set_attribute("bm25.results_count", len(bm25_ids))
                bm25_span.set_attribute("bm25.duration", bm25_duration)
                
                # Record in both systems
                bm25_duration_histogram.record(bm25_duration)
                prometheus_bm25_duration.observe(bm25_duration)

            # 2. FAISS semantic search
            with tracer.start_as_current_span("faiss_search") as faiss_span:
                faiss_start = time.time()
                query_emb = sbert.encode([query], normalize_embeddings=True)
                _, faiss_idx = faiss_index.search(query_emb, FAISS_N)
                
                faiss_ids = []
                for i in faiss_idx[0]:
                    if i in idx_to_id:
                        faiss_ids.append(idx_to_id[i])
                    else:
                        faiss_ids.append(id_list[i])
                        
                faiss_duration = time.time() - faiss_start
                faiss_span.set_attribute("faiss.results_count", len(faiss_ids))
                faiss_span.set_attribute("faiss.duration", faiss_duration)
                
                # Record in both systems
                faiss_duration_histogram.record(faiss_duration)
                prometheus_faiss_duration.observe(faiss_duration)

            # 3. RRF Fusion
            rrf_ids = rrf_fusion(bm25_ids, faiss_ids, top_n=RRF_N)

            # 4. CrossEncoder reranking (if available)
            if cross_encoder:
                with tracer.start_as_current_span("crossencoder_rerank") as rerank_span:
                    rerank_start = time.time()
                    texts = [data_df.loc[_id]['combined_text'] for _id in rrf_ids]
                    cross_inputs = list(zip([query]*len(texts), texts))
                    cross_scores = cross_encoder.predict(cross_inputs)
                    reranked = sorted(zip(rrf_ids, cross_scores), key=lambda x: x[1], reverse=True)
                    final_ids = [id_ for id_, score in reranked[:top_k]]
                    rerank_duration = time.time() - rerank_start
                    
                    rerank_span.set_attribute("rerank.candidates_count", len(rrf_ids))
                    rerank_span.set_attribute("rerank.final_count", len(final_ids))
                    rerank_span.set_attribute("rerank.duration", rerank_duration)
                    
                    # Record in both systems
                    rerank_duration_histogram.record(rerank_duration)
                    prometheus_rerank_duration.observe(rerank_duration)
            else:
                final_ids = rrf_ids[:top_k]

            # Format results
            results = []
            for idx in final_ids:
                row = data_df.loc[idx]
                results.append({
                    "id": int(idx),
                    "title": row['title'],
                    "description": row['description'],
                    "genre": row['genre']
                })
            
            search_time = time.time() - search_start_time
            
            # Categorize metrics for better analysis
            query_length_bucket = "short" if len(query) < 20 else "medium" if len(query) < 50 else "long"
            results_found_bucket = "none" if len(results) == 0 else "few" if len(results) < 3 else "many"
            
            # Record metrics in both systems
            search_duration_histogram.record(search_time)
            results_count_histogram.record(len(results))
            search_requests_counter.add(1, {
                "query_length": len(query),
                "results_found": len(results)
            })
            
            prometheus_search_duration.observe(search_time)
            prometheus_results_count.observe(len(results))
            prometheus_search_requests.labels(
                query_length_bucket=query_length_bucket,
                results_found_bucket=results_found_bucket
            ).inc()
            
            span.set_attribute("search.total_duration", search_time)
            span.set_attribute("search.results_count", len(results))
            span.set_attribute("search.success", True)
            
            return results, search_time
    
    except Exception as e:
        search_time = time.time() - search_start_time
        
        # Record errors in both systems
        errors_counter.add(1, {"operation": "hybrid_search", "error_type": type(e).__name__})
        prometheus_errors.labels(operation="hybrid_search", error_type=type(e).__name__).inc()
        
        print(f"❌ Search error: {e}")
        return [], search_time

# === LOGGING ===
@trace_function("save_query")
def save_query(query, top_k, results, satisfied, feedback=""):
    """Save user query with monitoring"""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        entry = {
            "user_query": query,
            "top_k": top_k,
            "results": results,
            "user_satisfied": satisfied,
            "user_feedback": feedback,
            "model_type": "hybrid" if sbert else "bm25_only",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        else:
            cache = []
            
        cache.append(entry)
        
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)
            
        # Update satisfaction metrics in both systems
        if satisfied is not None:
            try:
                satisfaction_rate = sum(1 for q in cache if q.get('user_satisfied') == True) / len(cache)
                user_satisfaction_gauge.set(satisfaction_rate)
                prometheus_user_satisfaction.set(satisfaction_rate)
            except:
                pass
                
    except Exception as e:
        print(f"⚠️ Query save error: {e}")
        errors_counter.add(1, {"operation": "save_query", "error_type": type(e).__name__})
        prometheus_errors.labels(operation="save_query", error_type=type(e).__name__).inc()

# === GRADIO INTERFACE ===
@trace_function("search_books_endpoint")
def search_books(query, num_results):
    """Main search function for Gradio with dual monitoring"""
    if not query.strip():
        return "⚠️ Please enter a search query!", None, None
    
    results, search_time = hybrid_search(query, num_results)
    
    if not results:
        return "😔 No relevant books found. Try a different query.", None, None
    
    # Format results
    output_text = f"### 📚 Found {len(results)} books in {search_time:.2f} seconds:\n\n"
    
    for i, res in enumerate(results, 1):
        output_text += f"**{i}. {res['title']}**\n"
        output_text += f"📖 *Description:* {res['description']}\n"
        output_text += f"🏷️ *Genre:* {res['genre']}\n"
        output_text += f"---\n\n"
    
    # Save query
    save_query(query, num_results, results, None)
    
    return output_text, results, query

@trace_function("submit_feedback_endpoint")
def submit_feedback(results, query, satisfied, additional_feedback):
    """Process user feedback with dual monitoring"""
    if results is None:
        return "⚠️ Please perform a search first!"
    
    # Record feedback metrics in both systems
    satisfied_bool = satisfied == "Yes ✅"
    has_feedback = bool(additional_feedback.strip())
    
    feedback_counter.add(1, {
        "satisfied": str(satisfied_bool),
        "has_additional_feedback": str(has_feedback)
    })
    prometheus_feedback_submissions.labels(
        satisfied=str(satisfied_bool),
        has_additional_feedback=str(has_feedback)
    ).inc()
    
    # Save feedback
    save_query(query, len(results), results, satisfied_bool, additional_feedback)
    
    if satisfied == "Yes ✅":
        return "✅ Thank you for your feedback! Glad we could help."
    else:
        if additional_feedback.strip():
            # Perform new search with refinement
            new_query = f"{query} {additional_feedback}"
            new_results, search_time = hybrid_search(new_query, len(results))
            
            if new_results:
                output_text = f"### 🔄 Refined results ({search_time:.2f} sec):\n\n"
                for i, res in enumerate(new_results, 1):
                    output_text += f"**{i}. {res['title']}**\n"
                    output_text += f"📖 *Description:* {res['description']}\n"
                    output_text += f"🏷️ *Genre:* {res['genre']}\n"
                    output_text += f"---\n\n"
                return output_text
            else:
                return "😔 Unfortunately, couldn't find better results."
        else:
            return "💡 Try adding more details to refine your search."

def get_statistics():
    """Return system statistics"""
    pipeline_mode = "Full ML Pipeline" if sbert and faiss_index else "BM25 Only"
    
    stats = f"""### 📊 System Statistics
    
- 📚 **Books in database:** {len(data_df) if data_df is not None else 0}
- 🤖 **Pipeline mode:** {pipeline_mode}
- 🔍 **FAISS vectors:** {faiss_index.ntotal if faiss_index is not None else "N/A"}
- 📏 **Model dimension:** {sbert.get_sentence_embedding_dimension() if sbert is not None else "N/A"}
- 🖥️ **Memory usage:** {psutil.virtual_memory().percent:.1f}%
- 🔬 **Monitoring:** SigNoz + OpenTelemetry + Prometheus + Grafana + Dagster
"""
    
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            stats += f"- 🔢 **Total queries:** {len(cache)}\n"
            
            # Calculate satisfaction
            satisfied_count = sum(1 for q in cache if q.get('user_satisfied') == True)
            if len(cache) > 0:
                satisfaction_rate = (satisfied_count / len(cache)) * 100
                stats += f"- 😊 **Satisfaction:** {satisfaction_rate:.1f}%\n"
        except:
            stats += "- 🔢 **Total queries:** 0\n"
    else:
        stats += "- 🔢 **Total queries:** 0\n"
    
    return stats

# === MAIN GRADIO APP ===
@trace_function("create_gradio_interface")
def create_gradio_interface():
    """Create Gradio interface with complete monitoring stack"""
    
    with gr.Blocks(title="📚 Ukrainian Book Recommendations + Complete MLOps", theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown("""
        # 📚 Hybrid Search Book Recommendations
        ### 🤖 Complete MLOps Stack: Monitoring + Drift Detection + Orchestration
        
        Find your next favorite book! The system uses advanced ML pipeline with comprehensive monitoring.
        
        **🔍 Complete Monitoring Stack:**
        - [Dagster Orchestration](http://localhost:3070) - Pipeline & Drift Detection
        - [Grafana Dashboards](http://localhost:3000) - Visual Metrics & Alerts  
        - [SigNoz Tracing](http://localhost:3301) - Traces & Advanced Analytics
        - [Prometheus Metrics](http://localhost:9090) - Raw Metrics
        """)
        
        # State storage
        results_state = gr.State(None)
        query_state = gr.State(None)
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=2):
                # Search section
                with gr.Group():
                    gr.Markdown("### 🔍 Search Books")
                    query_input = gr.Textbox(
                        label="Describe the book you want to find",
                        placeholder="For example: detective about murder in a locked room",
                        lines=3
                    )
                    
                    with gr.Row():
                        num_results = gr.Slider(
                            minimum=1, 
                            maximum=10, 
                            value=5, 
                            step=1, 
                            label="Number of results"
                        )
                        search_btn = gr.Button("🔍 Find Books", variant="primary")
                
                # Results
                results_output = gr.Markdown(label="Search Results")
                
                # Feedback section
                with gr.Group():
                    gr.Markdown("### 💬 Your Feedback")
                    satisfaction = gr.Radio(
                        ["Yes ✅", "No ❌"],
                        label="Are you satisfied with the results?",
                        value="Yes ✅"
                    )
                    feedback_input = gr.Textbox(
                        label="Add details to improve search (optional)",
                        placeholder="For example: I want more modern detectives",
                        lines=2,
                        visible=True
                    )
                    submit_btn = gr.Button("📤 Submit Feedback", variant="secondary")
                
                feedback_output = gr.Markdown(label="Feedback Response")
            
            # Sidebar
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ℹ️ ML Pipeline")
                    pipeline_status = "🟢 Full Pipeline Active" if sbert and faiss_index else "🟡 BM25 Only Mode"
                    gr.Markdown(f"""
                    **Status:** {pipeline_status}
                    
                    **Components:**
                    - 🔍 BM25 lexical search ✅
                    - 🤖 Fine-tuned SBERT {'✅' if sbert else '❌'}
                    - 🧠 FAISS semantic search {'✅' if faiss_index else '❌'}
                    - 🔄 RRF fusion {'✅' if faiss_index else '❌'}
                    - 🎯 CrossEncoder reranking {'✅' if cross_encoder else '❌'}
                    """)
                
                # MLOps Stack info
                with gr.Group():
                    gr.Markdown("### 🚀 MLOps Stack")
                    gr.Markdown("""
                    **Complete Observability:**
                    - 🔄 **Dagster**: Pipeline orchestration & drift detection
                    - 📊 **Grafana**: Visual dashboards & alerting
                    - 🔍 **SigNoz**: Distributed tracing & analytics
                    - 📈 **Prometheus**: Metrics collection & storage
                    - 📝 **OpenTelemetry**: Instrumentation
                    
                    **Drift Detection:**
                    - 🎯 Input query analysis
                    - 📚 Output recommendation patterns
                    - ⏰ Automated daily checks
                    - 🔔 Real-time alerting
                    
                    **Access Dashboards:**
                    - [Dagster UI](http://localhost:3070)
                    - [Grafana](http://localhost:3000) (admin/admin)
                    - [SigNoz](http://localhost:3301)
                    - [Prometheus](http://localhost:9090)
                    """)
                
                # Statistics
                stats_output = gr.Markdown(get_statistics())
                refresh_btn = gr.Button("🔄 Refresh Statistics", size="sm")
        
        # Examples
        gr.Examples(
            examples=[
                ["fantasy about dragon and magic"],
                ["detective in Agatha Christie style"],
                ["science fiction about space"],
                ["romantic story with happy ending"],
                ["thriller with unexpected finale"],
                ["українська класика"],
                ["книга про війну"],
                ["філософська література"]
            ],
            inputs=query_input
        )
        
        # Event handlers
        search_btn.click(
            fn=search_books,
            inputs=[query_input, num_results],
            outputs=[results_output, results_state, query_state]
        )
        
        submit_btn.click(
            fn=submit_feedback,
            inputs=[results_state, query_state, satisfaction, feedback_input],
            outputs=feedback_output
        )
        
        refresh_btn.click(
            fn=get_statistics,
            outputs=stats_output
        )
        
        # Footer
        gr.Markdown("""
        ---
        💡 **Tip:** The more detailed your description, the better the results!
        
        🚀 **Complete MLOps Stack**: Monitoring + Drift Detection + Orchestration + Alerting
        """)
    
    return demo

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting Ukrainian Book Recommendation System with Complete MLOps Stack")
    print("=" * 80)
    
    # Start Prometheus metrics server
    start_prometheus_server()
    
    # Initialize system
    if initialize_system():
        # Create and launch interface
        demo = create_gradio_interface()
        
        print("🎯 Complete MLOps Stack Access Points:")
        print("📚 Book Recommendation App: http://localhost:7860")
        print("🔄 Dagster Orchestration: http://localhost:3070")
        print("📊 Grafana Dashboards: http://localhost:3000 (admin/admin)")
        print("🔍 SigNoz Tracing: http://localhost:3301")
        print("📈 Prometheus Metrics: http://localhost:9090")
        print("🔢 Raw Metrics Endpoint: http://localhost:8080/metrics")
        
        # Launch with monitoring
        demo.launch(
            share=False,
            server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
            server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
            show_error=True,
            favicon_path=None
        )
    else:
        print("❌ Failed to initialize system!")
        exit(1)