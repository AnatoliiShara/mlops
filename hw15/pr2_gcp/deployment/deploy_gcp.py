# pr2_gcp/deployment/deploy_gcp.py
# Простий deployment на GCP Vertex AI

import os
import json
import time
from datetime import datetime
from typing import Dict, Any
from google.cloud import aiplatform
from google.cloud import storage
import zipfile
import tempfile

class SimpleGCPDeployment:
    """Простий класс для деплою мульти-моделі на GCP Vertex AI"""
    
    def __init__(self, project_id: str = None, region: str = "us-central1"):
        """Ініціалізація GCP клієнтів"""
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.region = region
        
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID не встановлено")
        
        # Ініціалізуємо Vertex AI
        aiplatform.init(project=self.project_id, location=self.region)
        
        # Storage клієнт
        self.storage_client = storage.Client(project=self.project_id)
        self.bucket_name = f"{self.project_id}-bookstore-models"
        
        print(f"🔧 GCP Project: {self.project_id}")
        print(f"🌍 Region: {self.region}")
        print(f"📦 Bucket: {self.bucket_name}")
        
        # Створюємо bucket якщо не існує
        self._create_bucket_if_not_exists()
    
    def _create_bucket_if_not_exists(self):
        """Створення GCS bucket якщо не існує"""
        try:
            bucket = self.storage_client.get_bucket(self.bucket_name)
            print(f"✅ Bucket {self.bucket_name} exists")
        except:
            print(f"🔨 Creating bucket {self.bucket_name}...")
            bucket = self.storage_client.create_bucket(self.bucket_name, location="US")
            print(f"✅ Bucket {self.bucket_name} created")
    
    def create_prediction_container(self):
        """Створення Docker контейнера для prediction"""
        # Створюємо main.py для Vertex AI custom container
        main_py_content = '''
import os
import json
import sys
from flask import Flask, request, jsonify
import logging

# Додаємо шлях до shared модулів
sys.path.append('/app/shared')

from book_models import create_multi_model_predictor

# Створюємо Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Глобальна змінна для предиктора
predictor = None

@app.before_first_request
def load_model():
    """Завантаження моделі при старті"""
    global predictor
    predictor = create_multi_model_predictor()
    app.logger.info("✅ Multi-model predictor loaded")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if predictor is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    
    return jsonify(predictor.health_check())

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint для Vertex AI"""
    global predictor
    
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Vertex AI передає дані в форматі {"instances": [...]}
        data = request.get_json()
        
        if "instances" in data:
            # Стандартний Vertex AI формат
            instances = data["instances"]
            predictions = []
            
            for instance in instances:
                model_name = instance.get("model_name", "ensemble")
                input_data = instance.get("input_data", {})
                
                result = predictor.predict(model_name, input_data)
                predictions.append(result)
            
            return jsonify({"predictions": predictions})
        
        else:
            # Прямий формат
            model_name = data.get("model_name", "ensemble")
            input_data = data.get("input_data", {})
            
            result = predictor.predict(model_name, input_data)
            return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "service": "Multi-Model Book Recommendation",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict"]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
'''
        
        # Створюємо Dockerfile
        dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

# Встановлюємо системні залежності
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Копіюємо requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код
COPY main.py .
COPY shared/ ./shared/

# Встановлюємо PORT для Vertex AI
ENV PORT=8080

EXPOSE 8080

CMD ["python", "main.py"]
'''

        # Створюємо requirements.txt
        requirements_content = '''
flask>=2.3.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
numpy>=1.24.0
google-cloud-storage>=2.7.0
'''

        # Створюємо директорію для контейнера
        container_dir = 'container_build'
        os.makedirs(container_dir, exist_ok=True)
        
        # Записуємо файли
        with open(f'{container_dir}/main.py', 'w', encoding='utf-8') as f:
            f.write(main_py_content)
        
        with open(f'{container_dir}/Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        with open(f'{container_dir}/requirements.txt', 'w') as f:
            f.write(requirements_content)
        
        # Копіюємо shared модулі
        import shutil
        shutil.copytree('../shared', f'{container_dir}/shared', dirs_exist_ok=True)
        
        print("✅ Container files created")
        return container_dir
    
    def build_and_push_container(self, container_dir: str) -> str:
        """Білд та push Docker контейнера в GCR"""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        image_name = f"gcr.io/{self.project_id}/bookstore-multi-model"
        image_tag = f"{image_name}:{timestamp}"
        
        print(f"🔨 Building container: {image_tag}")
        
        # Docker build команда
        import subprocess
        
        build_cmd = [
            "docker", "build", 
            "-t", image_tag,
            container_dir
        ]
        
        try:
            subprocess.run(build_cmd, check=True, cwd=".")
            print(f"✅ Container built: {image_tag}")
            
            # Push до GCR
            print(f"📤 Pushing to GCR...")
            push_cmd = ["docker", "push", image_tag]
            subprocess.run(push_cmd, check=True)
            print(f"✅ Container pushed: {image_tag}")
            
            return image_tag
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Container build/push failed: {e}")
            raise
    
    def deploy_model_to_vertex(self, image_uri: str) -> str:
        """Deploy моделі на Vertex AI"""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_name = f"bookstore-multi-model-{timestamp}"
        endpoint_name = f"bookstore-endpoint-{timestamp}"
        
        print(f"🚀 Deploying model: {model_name}")
        
        # Створюємо модель
        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=f"gs://{self.bucket_name}/artifacts",  # Placeholder
            serving_container_image_uri=image_uri,
            serving_container_ports=[8080],
            serving_container_predict_route="/predict",
            serving_container_health_route="/health"
        )
        
        print(f"✅ Model created: {model.display_name}")
        
        # Створюємо endpoint
        print(f"🔗 Creating endpoint: {endpoint_name}")
        endpoint = model.deploy(
            endpoint=aiplatform.Endpoint.create(display_name=endpoint_name),
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=3,
            accelerator_type=None,  # CPU only для простоти
            accelerator_count=0
        )
        
        print(f"✅ Model deployed to endpoint: {endpoint.display_name}")
        return endpoint, model
    
    def test_vertex_endpoint(self, endpoint) -> bool:
        """Тестування Vertex AI endpoint"""
        test_instances = [{
            "model_name": "ensemble",
            "input_data": {
                "query": "фентезі книги про магію",
                "top_k": 2
            }
        }]
        
        print(f"🧪 Testing endpoint...")
        try:
            predictions = endpoint.predict(instances=test_instances)
            print("✅ Test successful!")
            print(f"📋 Result: {json.dumps(predictions.predictions, ensure_ascii=False, indent=2)}")
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def deploy_complete_pipeline(self):
        """Повний pipeline деплою на GCP"""
        try:
            # 1. Створення контейнера
            print("🔨 Creating container files...")
            container_dir = self.create_prediction_container()
            
            # 2. Білд та push контейнера
            print("📦 Building and pushing container...")
            image_uri = self.build_and_push_container(container_dir)
            
            # 3. Deploy на Vertex AI
            print("🚀 Deploying to Vertex AI...")
            endpoint, model = self.deploy_model_to_vertex(image_uri)
            
            # 4. Тестування
            print("🧪 Testing deployment...")
            test_success = self.test_vertex_endpoint(endpoint)
            
            if test_success:
                print(f"🎉 GCP Deployment successful!")
                print(f"📍 Endpoint: {endpoint.display_name}")
                print(f"🔗 Resource name: {endpoint.resource_name}")
                
                return {
                    "endpoint": endpoint,
                    "model": model,
                    "image_uri": image_uri,
                    "status": "success"
                }
            else:
                print("❌ Deployment failed - endpoint test unsuccessful")
                return {"status": "failed", "reason": "endpoint test failed"}
                
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return {"status": "failed", "reason": str(e)}

def main():
    """Основна функція для деплою"""
    # Перевіряємо GCP credentials
    try:
        project_id = os.getenv('GCP_PROJECT_ID')
        if not project_id:
            print("❌ GCP_PROJECT_ID не встановлено")
            return
        
        print("✅ GCP credentials found")
    except Exception as e:
        print(f"❌ GCP credentials error: {e}")
        return
    
    # Створюємо deployment
    deployment = SimpleGCPDeployment()
    
    # Запускаємо повний pipeline
    result = deployment.deploy_complete_pipeline()
    
    if result["status"] == "success":
        print("\n🎯 GCP Deployment completed successfully!")
        print(f"Use endpoint: {result['endpoint'].display_name}")
    else:
        print(f"\n❌ Deployment failed: {result.get('reason', 'Unknown error')}")

if __name__ == "__main__":
    main()