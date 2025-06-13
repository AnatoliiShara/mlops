FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy data directory for local development
COPY data/ ./data/

# Set environment variables for OpenTelemetry
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
ENV OTEL_PYTHON_LOG_CORRELATION=true
ENV OTEL_PYTHON_LOG_LEVEL=info

# Expose ports
EXPOSE 7860 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860 || exit 1

# Run the application
CMD ["python", "app.py"]