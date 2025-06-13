"""
Async Consumer Worker for processing book recommendations
"""
import asyncio
import json
import time
import os
import sys
import signal
from datetime import datetime
from typing import Dict, List, Optional
import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import faiss
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.schemas.models import JobStatus, BookRecommendation
from src.utils.search_utils import hybrid_search_with_rerank

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
jobs_processed = Counter('jobs_processed_total', 'Total jobs processed', ['status'])
processing_duration = Histogram('job_processing_duration_seconds', 'Job processing duration')
active_workers = Gauge('active_workers', 'Number of active workers')

class BookRecommenderWorker:
    """Async worker for processing book recommendation jobs"""
    
    def __init__(self, worker_id: str = None):
        self.worker_id = worker_id or f"worker-{os.getpid()}"
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # ML models
        self.embedder: Optional[SentenceTransformer] = None
        self.cross_encoder: Optional[CrossEncoder] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25_model: Optional[BM25Okapi] = None
        
        # Data
        self.df_books: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.texts: Optional[List[str]] = None
        
        # Control
        self.shutdown = False
        self.tasks = set()
        
    async def initialize(self):
        """Initialize connections and load models"""
        logger.info(f"Initializing worker {self.worker_id}")
        
        try:
            # Start Prometheus metrics server
            metrics_port = int(os.getenv('METRICS_PORT', 9090))
            start_http_server(metrics_port)
            active_workers.inc()
            
            # Initialize Kafka Consumer
            kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
            self.consumer = AIOKafkaConsumer(
                'book-recommendations',
                bootstrap_servers=kafka_servers,
                group_id='book-recommender-workers',
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                enable_auto_commit=False,
                auto_offset_reset='earliest',
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                max_poll_records=1,
                max_poll_interval_ms=300000  # 5 minutes
            )
            
            # Initialize Redis
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Start consumer
            await self.consumer.start()
            await self.redis_client.ping()
            
            logger.info("Connections established, loading models...")
            
            # Load models and data
            await self._load_models()
            
            logger.info(f"Worker {self.worker_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize worker: {e}")
            raise
    
    async def _load_models(self):
        """Load ML models and prepare data"""
        # Load data
        data_path = os.getenv('DATA_PATH', '/app/data/ukr_books_dataset.csv')
        self.df_books = pd.read_csv(data_path)
        self.df_books = self.df_books.dropna(subset=["title", "description"])
        self.df_books["combined_text"] = self.df_books["title"] + " " + self.df_books["description"]
        self.texts = self.df_books["combined_text"].tolist()
        
        # Load models
        embedder_model = os.getenv('EMBEDDER_MODEL', 'sentence-transformers/all-MiniLM-L12-v2')
        cross_encoder_model = os.getenv('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.embedder = SentenceTransformer(embedder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        # Load or compute embeddings
        embeddings_path = os.getenv('EMBEDDINGS_PATH', '/app/models/embeddings.npy')
        if os.path.exists(embeddings_path):
            logger.info("Loading pre-computed embeddings...")
            self.embeddings = np.load(embeddings_path)
        else:
            logger.info("Computing embeddings...")
            self.embeddings = self.embedder.encode(
                self.texts, 
                convert_to_numpy=True, 
                show_progress_bar=True,
                normalize_embeddings=True
            )
            os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            np.save(embeddings_path, self.embeddings)
        
        # Build FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)
        
        # Build BM25 model
        tokenized_texts = [text.split() for text in self.texts]
        self.bm25_model = BM25Okapi(tokenized_texts)
        
        logger.info("Models loaded successfully")
    
    async def process_message(self, message: Dict):
        """Process a single recommendation job"""
        job_id = message['job_id']
        start_time = time.time()
        
        logger.info(f"Processing job {job_id}")
        
        try:
            # Update job status to processing
            await self._update_job_status(job_id, JobStatus.PROCESSING)
            
            # Remove from pending queue
            await self.redis_client.lrem("queue:pending", 1, job_id)
            
            # Process with timeout
            result = await asyncio.wait_for(
                self._perform_search(message),
                timeout=120.0  # 2 minute timeout
            )
            
            # Store result
            processing_time = time.time() - start_time
            result_data = {
                "job_id": job_id,
                "status": JobStatus.COMPLETED.value,
                "query": message['query'],
                "recommendations": result,
                "processing_time": processing_time,
                "worker_id": self.worker_id,
                "created_at": message['timestamp'],
                "completed_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                f"result:{job_id}",
                3600,  # 1 hour TTL
                json.dumps(result_data)
            )
            
            # Update job status
            await self._update_job_status(job_id, JobStatus.COMPLETED)
            
            # Update metrics
            jobs_processed.labels(status='success').inc()
            processing_duration.observe(processing_time)
            await self.redis_client.incr("metrics:completed_last_minute")
            
            logger.info(f"Job {job_id} completed in {processing_time:.2f}s")
            
        except asyncio.TimeoutError:
            logger.error(f"Job {job_id} timed out")
            await self._handle_job_failure(job_id, "Processing timeout")
            jobs_processed.labels(status='timeout').inc()
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            await self._handle_job_failure(job_id, str(e))
            jobs_processed.labels(status='error').inc()
    
    async def _perform_search(self, message: Dict) -> List[Dict]:
        """Perform the actual book search"""
        query = message['query']
        top_k = message.get('top_k', 5)
        
        # Perform hybrid search
        results = hybrid_search_with_rerank(
            query=query,
            bm25_model=self.bm25_model,
            faiss_index=self.faiss_index,
            embeddings=self.embeddings,
            texts=self.texts,
            df=self.df_books,
            cross_encoder=self.cross_encoder,
            top_k=top_k,
            bm25_k=20,
            faiss_k=20,
            rrf_k=60
        )
        
        # Format recommendations
        recommendations = []
        for rank, (idx, score) in enumerate(results, 1):
            book = self.df_books.iloc[idx]
            recommendations.append({
                "rank": rank,
                "title": book['title'],
                "description": book['description'][:200] + "..." if len(book['description']) > 200 else book['description'],
                "genre": book.get('genre'),
                "rating": float(book['rating']) if pd.notna(book.get('rating')) else None,
                "score": float(score)
            })
        
        return recommendations
    
    async def _update_job_status(self, job_id: str, status: JobStatus):
        """Update job status in Redis"""
        job_data = await self.redis_client.get(f"job:{job_id}")
        if job_data:
            job_info = json.loads(job_data)
            old_status = job_info.get("status")
            job_info["status"] = status.value
            job_info["updated_at"] = datetime.utcnow().isoformat()
            if status == JobStatus.PROCESSING:
                job_info["worker_id"] = self.worker_id
            
            await self.redis_client.setex(
                f"job:{job_id}",
                3600,
                json.dumps(job_info)
            )
            
            # Update metrics
            if old_status:
                await self.redis_client.decr(f"metrics:jobs:{old_status}")
            await self.redis_client.incr(f"metrics:jobs:{status.value}")
    
    async def _handle_job_failure(self, job_id: str, error: str):
        """Handle job failure"""
        await self._update_job_status(job_id, JobStatus.FAILED)
        
        # Store error info
        job_data = await self.redis_client.get(f"job:{job_id}")
        if job_data:
            job_info = json.loads(job_data)
            job_info["error"] = error
            job_info["failed_at"] = datetime.utcnow().isoformat()
            
            await self.redis_client.setex(
                f"job:{job_id}",
                3600,
                json.dumps(job_info)
            )
    
    async def run(self):
        """Main worker loop"""
        await self.initialize()
        
        logger.info(f"Worker {self.worker_id} starting main loop")
        
        try:
            async for msg in self.consumer:
                if self.shutdown:
                    break
                
                # Process message in background
                task = asyncio.create_task(self.process_message(msg.value))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
                
                # Commit offset after creating task
                await self.consumer.commit()
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            await self.shutdown_worker()
    
    async def shutdown_worker(self):
        """Graceful shutdown"""
        logger.info(f"Shutting down worker {self.worker_id}")
        self.shutdown = True
        
        # Wait for ongoing tasks
        if self.tasks:
            logger.info(f"Waiting for {len(self.tasks)} tasks to complete")
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close connections
        if self.consumer:
            await self.consumer.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        active_workers.dec()
        logger.info(f"Worker {self.worker_id} shut down successfully")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    asyncio.create_task(worker.shutdown_worker())

# Global worker instance for signal handling
worker = None

async def main():
    """Main entry point"""
    global worker
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run worker
    worker_id = os.getenv('WORKER_ID')
    worker = BookRecommenderWorker(worker_id)
    
    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())