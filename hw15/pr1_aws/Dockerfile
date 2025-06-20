FROM python:3.9-slim

WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY ../shared ./shared

# Copy AWS specific files
COPY models/multi_model.py ./inference.py

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV SAGEMAKER_PROGRAM=inference.py

# Create necessary directories
RUN mkdir -p /opt/ml/model /opt/ml/input /opt/ml/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import inference; print('healthy')" || exit 1

EXPOSE 8080

# SageMaker model server
CMD ["serve"]