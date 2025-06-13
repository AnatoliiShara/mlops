"""
FastAPI application with Prometheus metrics for HPA
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from contextlib import asynccontextmanager
import time
import psutil
import torch
import numpy as np
import pandas as pd
import faiss
import os
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
active_requests = Gauge('http_requests_active', 'Active HTTP requests')
model_inference_duration = Histogram('model_inference_duration_seconds', 'Model inference duration')
cpu_usage = Gauge('process_cpu_usage_percent', 'Process CPU usage')
memory_usage = Gauge('process_memory_usage_bytes', 'Process memory usage')

# Global variables for models
embedder = None
cross_encoder = None
faiss_index = None
bm25_model = None
df_books = None
embeddings = None
texts = None

# Request/Response models
class BookQuery(BaseModel):
    query: str
    top_k: int = 5

class BookRecommendation(BaseModel):
    rank: int
    title: str
    description: str
    genre: Optional[str]
    rating: Optional[float]
    score: float

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[BookRecommendation]
    processing_time: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global embedder, cross_encoder, faiss_index, bm25_model, df_books, embeddings, texts
    
    logger.info("Loading models and data...")
    
    # Load data
    data_path = os.getenv('DATA_PATH', '/app/data/ukr_books_dataset.csv')
    df_books = pd.read_csv(data_path)
    df_books = df_books.dropna(subset=["title", "description"])
    df_books["combined_text"] = df_books["title"] + " " + df_books["description"]
    texts = df_books["combined_text"].tolist()
    
    # Load models
    embedder_model = os.getenv('EMBEDDER_MODEL', 'sentence-transformers/all-MiniLM-L12-v2')
    cross_encoder_model = os.getenv('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    embedder = SentenceTransformer(embedder_model)
    cross_encoder = CrossEncoder(cross_encoder_model)
    
    # Generate or load embeddings
    embeddings_path = os.getenv('EMBEDDINGS_PATH', '/app/models/embeddings.npy')
    if os.path.exists(embeddings_path):
        logger.info("Loading pre-computed embeddings...")
        embeddings = np.load(embeddings_path)
    else:
        logger.info("Computing embeddings...")
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.save(embeddings_path, embeddings)
    
    # Build FAISS index
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True))
    
    # Build BM25 model
    tokenized_texts = [text.split() for text in texts]
    bm25_model = BM25Okapi(tokenized_texts)
    
    logger.info("Models loaded successfully!")
    
    yield
    
    # Cleanup (if needed)
    logger.info("Shutting down...")

# Create app
app = FastAPI(
    title="Book Recommender API",
    description="Book recommendation service with HPA support",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def track_metrics(request, call_next):
    """Track request metrics"""
    active_requests.inc()
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    request_duration.observe(duration)
    active_requests.dec()
    
    return response

@app.get("/health")
async def health_check():
    """Kubernetes liveness probe endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    if embedder is None or faiss_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready", "timestamp": time.time()}

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # Update system metrics
    cpu_usage.set(psutil.cpu_percent(interval=0.1))
    memory_usage.set(psutil.Process().memory_info().rss)
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(query: BookQuery):
    """Get book recommendations"""
    start_time = time.time()
    
    with model_inference_duration.time():
        # Encode query
        query_embedding = embedder.encode([query.query], convert_to_numpy=True)[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # FAISS search
        faiss_scores, faiss_ids = faiss_index.search(
            np.expand_dims(query_embedding, 0), 
            min(20, len(texts))
        )
        
        # BM25 search
        bm25_scores = bm25_model.get_scores(query.query.split())
        
        # RRF fusion
        k = 60
        faiss_rank = np.argsort(-faiss_scores[0])
        bm25_rank = np.argsort(-bm25_scores)
        
        rrf_scores = np.zeros(len(texts))
        for rank_list in [faiss_ids[0], bm25_rank[:20]]:
            for r, idx in enumerate(rank_list):
                if idx < len(texts):
                    rrf_scores[idx] += 1 / (k + r + 1)
        
        # Get top candidates
        top_indices = np.argsort(-rrf_scores)[:query.top_k * 2]
        candidate_texts = [texts[i] for i in top_indices]
        
        # Cross-encoder reranking
        pairs = [[query.query, text] for text in candidate_texts]
        ce_scores = cross_encoder.predict(pairs)
        
        # Get final top-k
        ce_ranking = np.argsort(-ce_scores)[:query.top_k]
        
        # Prepare recommendations
        recommendations = []
        for rank, idx in enumerate(ce_ranking):
            global_idx = top_indices[idx]
            book = df_books.iloc[global_idx]
            
            recommendations.append(BookRecommendation(
                rank=rank + 1,
                title=book['title'],
                description=book['description'][:200] + "..." if len(book['description']) > 200 else book['description'],
                genre=book.get('genre'),
                rating=float(book['rating']) if pd.notna(book.get('rating')) else None,
                score=float(ce_scores[idx])
            ))
    
    processing_time = time.time() - start_time
    
    return RecommendationResponse(
        query=query.query,
        recommendations=recommendations,
        processing_time=processing_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)