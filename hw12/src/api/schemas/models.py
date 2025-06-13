"""
Pydantic models for async API
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BookQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    user_id: Optional[str] = None

class JobSubmission(BaseModel):
    job_id: str
    status: JobStatus
    estimated_time_seconds: float
    message: str
    queue_position: Optional[int] = None

class BookRecommendation(BaseModel):
    rank: int
    title: str
    description: str
    genre: Optional[str]
    rating: Optional[float]
    score: float

class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    query: str
    recommendations: Optional[List[BookRecommendation]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    worker_id: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class HealthStatus(BaseModel):
    status: str
    kafka: str
    redis: str
    timestamp: datetime