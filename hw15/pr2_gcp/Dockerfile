FROM python:3.9-slim

WORKDIR /app

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

# Copy GCP specific files
COPY models ./models

# Create main app file
RUN cat > main.py << 'EOF'
import os
import sys
import json
from flask import Flask, request, jsonify

sys.path.append('/app/shared')
from book_models import create_multi_model_predictor

app = Flask(__name__)
predictor = None

@app.before_first_request
def load_model():
    global predictor
    predictor = create_multi_model_predictor()

@app.route('/health')
def health():
    if predictor:
        return jsonify({"status": "healthy"})
    return jsonify({"status": "unhealthy"}), 503

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if 'instances' in data:
        results = []
        for instance in data['instances']:
            model_name = instance.get('model_name', 'ensemble')
            input_data = instance.get('input_data', {})
            result = predictor.predict(model_name, input_data)
            results.append(result)
        return jsonify({"predictions": results})
    else:
        model_name = data.get('model_name', 'ensemble')
        input_data = data.get('input_data', {})
        result = predictor.predict(model_name, input_data)
        return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
EOF

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=TRUE

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["python", "main.py"]