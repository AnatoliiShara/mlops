FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-async.txt .
RUN pip install --no-cache-dir -r requirements-async.txt

# Copy application code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the producer API
CMD ["uvicorn", "src.api.producer_api:app", "--host", "0.0.0.0", "--port", "8000"]