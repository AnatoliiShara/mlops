"""
Async Producer API for book recommendations
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import json
import uuid
import asyncio
from datetime import datetime
import logging
import os
from typing import Optional

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
import redis.asyncio as redis

from src.schemas.models import (
    BookQuery, JobSubmission, JobResult, 
    JobStatus, HealthStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
job_submitted = Counter('jobs_submitted_total', 'Total jobs submitted')
job_status_gauge = Gauge('jobs_by_status', 'Current jobs by status', ['status'])
kafka_send_duration = Histogram('kafka_send_duration_seconds', 'Kafka send duration')
redis_operation_duration = Histogram('redis_operation_duration_seconds', 'Redis operation duration')

# Global connections
producer: Optional[AIOKafkaProducer] = None
redis_client: Optional[redis.Redis] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global producer, redis_client
    
    try:
        # Initialize Kafka Producer
        kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        producer = AIOKafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            compression_type='gzip',
            acks='all',
            retries=3,
            retry_backoff_ms=500,
            request_timeout_ms=30000,
            max_in_flight_requests_per_connection=5
        )
        await producer.start()
        logger.info(f"Kafka producer connected to {kafka_servers}")
        
        # Initialize Redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            retry_on_error=[ConnectionError, TimeoutError],
            health_check_interval=30
        )
        await redis_client.ping()
        logger.info(f"Redis connected to {redis_host}:{redis_port}")
        
        # Create Redis indices for better performance
        try:
            await redis_client.execute_command(
                'FT.CREATE', 'idx:jobs', 'ON', 'JSON', 
                'PREFIX', '1', 'job:', 
                'SCHEMA', '$.status', 'AS', 'status', 'TAG',
                '$.created_at', 'AS', 'created_at', 'NUMERIC', 'SORTABLE'
            )
        except redis.ResponseError:
            pass  # Index already exists
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize connections: {e}")
        raise
    finally:
        if producer:
            await producer.stop()
            logger.info("Kafka producer stopped")
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")

# Create FastAPI app
app = FastAPI(
    title="Book Recommender Async Producer API",
    description="Async job submission for book recommendations",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    kafka_status = "unknown"
    redis_status = "unknown"
    
    try:
        # Check Kafka
        if producer and producer._sender.sender_task and not producer._sender.sender_task.done():
            kafka_status = "healthy"
        else:
            kafka_status = "unhealthy"
    except:
        kafka_status = "error"
    
    try:
        # Check Redis
        await redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    overall_status = "healthy" if kafka_status == "healthy" and redis_status == "healthy" else "degraded"
    
    return HealthStatus(
        status=overall_status,
        kafka=kafka_status,
        redis=redis_status,
        timestamp=datetime.utcnow()
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # Update job status metrics
    try:
        for status in JobStatus:
            count = await redis_client.get(f"metrics:jobs:{status.value}") or 0
            job_status_gauge.labels(status=status.value).set(int(count))
    except:
        pass
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/recommend", response_model=JobSubmission)
async def submit_recommendation(query: BookQuery, background_tasks: BackgroundTasks):
    """Submit a book recommendation job"""
    job_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    
    # Prepare Kafka message
    message = {
        "job_id": job_id,
        "query": query.query,
        "top_k": query.top_k,
        "user_id": query.user_id or "anonymous",
        "timestamp": timestamp.isoformat(),
        "retry_count": 0
    }
    
    try:
        # Send to Kafka with timing
        with kafka_send_duration.time():
            await producer.send_and_wait(
                topic="book-recommendations",
                value=message,
                key=job_id,
                partition=None  # Let Kafka decide based on key
            )
        
        # Store job metadata in Redis
        job_data = {
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "query": query.query,
            "top_k": query.top_k,
            "user_id": query.user_id,
            "created_at": timestamp.isoformat(),
            "updated_at": timestamp.isoformat()
        }
        
        with redis_operation_duration.time():
            # Store job data
            await redis_client.setex(
                f"job:{job_id}",
                3600,  # 1 hour TTL
                json.dumps(job_data)
            )
            
            # Add to pending queue for queue position
            await redis_client.lpush("queue:pending", job_id)
            queue_position = await redis_client.llen("queue:pending")
            
            # Update metrics
            await redis_client.incr(f"metrics:jobs:{JobStatus.PENDING.value}")
        
        job_submitted.inc()
        
        # Estimate processing time based on queue
        estimated_time = 2.0 + (queue_position * 0.5)
        
        # Background task to clean up old jobs
        background_tasks.add_task(cleanup_old_jobs)
        
        return JobSubmission(
            job_id=job_id,
            status=JobStatus.PENDING,
            estimated_time_seconds=estimated_time,
            message=f"Job submitted successfully",
            queue_position=queue_position
        )
        
    except KafkaError as e:
        logger.error(f"Kafka error for job {job_id}: {e}")
        raise HTTPException(status_code=503, detail="Message queue unavailable")
    except Exception as e:
        logger.error(f"Unexpected error for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/job/{job_id}", response_model=JobResult)
async def get_job_result(job_id: str):
    """Get job result or status"""
    try:
        # Try to get result first
        result_data = await redis_client.get(f"result:{job_id}")
        if result_data:
            return JobResult(**json.loads(result_data))
        
        # Get job status
        job_data = await redis_client.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_info = json.loads(job_data)
        
        # Check queue position if pending
        queue_position = None
        if job_info.get("status") == JobStatus.PENDING.value:
            position = await redis_client.lpos("queue:pending", job_id)
            if position is not None:
                queue_position = position + 1
        
        return JobResult(
            job_id=job_id,
            status=JobStatus(job_info["status"]),
            query=job_info["query"],
            created_at=datetime.fromisoformat(job_info["created_at"]),
            error=job_info.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving job")

@app.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending job"""
    try:
        # Check if job exists
        job_data = await redis_client.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_info = json.loads(job_data)
        
        if job_info["status"] != JobStatus.PENDING.value:
            return {"message": f"Job {job_id} is already {job_info['status']}"}
        
        # Remove from pending queue
        removed = await redis_client.lrem("queue:pending", 0, job_id)
        
        # Update job status
        job_info["status"] = JobStatus.CANCELLED.value
        job_info["updated_at"] = datetime.utcnow().isoformat()
        
        await redis_client.setex(
            f"job:{job_id}",
            300,  # Keep cancelled jobs for 5 minutes
            json.dumps(job_info)
        )
        
        # Update metrics
        await redis_client.decr(f"metrics:jobs:{JobStatus.PENDING.value}")
        await redis_client.incr(f"metrics:jobs:{JobStatus.CANCELLED.value}")
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Error cancelling job")

@app.get("/queue/status")
async def get_queue_status():
    """Get current queue status"""
    try:
        pending_count = await redis_client.llen("queue:pending")
        
        # Get counts by status
        status_counts = {}
        for status in JobStatus:
            count = await redis_client.get(f"metrics:jobs:{status.value}") or 0
            status_counts[status.value] = int(count)
        
        # Get processing rate (last minute)
        completed_last_minute = await redis_client.get("metrics:completed_last_minute") or 0
        
        return {
            "queue_length": pending_count,
            "status_counts": status_counts,
            "processing_rate_per_minute": int(completed_last_minute),
            "estimated_wait_time": pending_count * 2.0  # Rough estimate
        }
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving queue status")

async def cleanup_old_jobs():
    """Background task to clean up old completed jobs"""
    try:
        # Clean up jobs older than 1 hour
        cutoff_time = datetime.utcnow().timestamp() - 3600
        
        # This is simplified - in production use Redis SCAN
        keys = await redis_client.keys("job:*")
        for key in keys[:100]:  # Limit to avoid blocking
            job_data = await redis_client.get(key)
            if job_data:
                job_info = json.loads(job_data)
                created_at = datetime.fromisoformat(job_info["created_at"]).timestamp()
                if created_at < cutoff_time and job_info["status"] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                    await redis_client.delete(key)
    except Exception as e:
        logger.error(f"Error in cleanup task: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)