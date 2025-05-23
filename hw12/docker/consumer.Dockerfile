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

# Create directories
RUN mkdir -p /app/data /app/models

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose metrics port
EXPOSE 9090

# Run the consumer worker
CMD ["python", "-m", "src.api.consumer_worker"]