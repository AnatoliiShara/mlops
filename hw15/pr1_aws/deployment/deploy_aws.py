# pr1_aws/deployment/deploy_aws.py
# Простий deployment на AWS SageMaker

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.multi_model import MultiDataModel
import time
import json
import os
from datetime import datetime

class SimpleAWSDeployment:
    """Простий класс для деплою мульти-моделі на AWS SageMaker"""
    
    def __init__(self, role_arn: str = None):
        """Ініціалізація AWS клієнтів"""
        self.session = sagemaker.Session()
        self.region = boto3.Session().region_name or 'us-east-1'
        
        # IAM роль для SageMaker (потрібно створити заздалегідь)
        self.role = role_arn or os.getenv('SAGEMAKER_ROLE_ARN')
        if not self.role:
            # Fallback - спробуємо знайти існуючу роль
            iam = boto3.client('iam')
            try:
                response = iam.get_role(RoleName='SageMakerExecutionRole')
                self.role = response['Role']['Arn']
            except:
                print("❌ SageMaker роль не знайдена. Створіть IAM роль 'SageMakerExecutionRole'")
                self.role = None
        
        self.bucket = self.session.default_bucket()
        self.prefix = 'multi-model-bookstore'
        
        print(f"🔧 AWS Region: {self.region}")
        print(f"📦 S3 Bucket: {self.bucket}")
        print(f"🔑 IAM Role: {self.role}")
    
    def create_model_artifacts(self):
        """Створення model artifacts для SageMaker"""
        # Створюємо inference.py для SageMaker
        inference_code = '''
import json
import sys
import os

# Додаємо шлях до shared модулів
sys.path.append('/opt/ml/code/shared')

from book_models import create_multi_model_predictor

# Глобальна змінна для предиктора
predictor = None

def model_fn(model_dir):
    """Завантаження моделі"""
    global predictor
    predictor = create_multi_model_predictor()
    print("✅ Multi-model predictor loaded")
    return predictor

def input_fn(request_body, request_content_type):
    """Обробка вхідних даних"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Інференс"""
    model_name = input_data.get('model_name', 'ensemble')
    input_payload = input_data.get('input_data', {})
    
    result = model.predict(model_name, input_payload)
    return result

def output_fn(prediction, content_type):
    """Форматування виводу"""
    if content_type == 'application/json':
        return json.dumps(prediction, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
        
        # Збереження inference.py
        os.makedirs('model_artifacts', exist_ok=True)
        with open('model_artifacts/inference.py', 'w', encoding='utf-8') as f:
            f.write(inference_code)
        
        # Копіюємо shared модулі
        import shutil
        shutil.copytree('../shared', 'model_artifacts/shared', dirs_exist_ok=True)
        
        # Створюємо requirements.txt для контейнера
        requirements = '''
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
numpy>=1.24.0
'''
        with open('model_artifacts/requirements.txt', 'w') as f:
            f.write(requirements)
        
        print("✅ Model artifacts created")
        return 'model_artifacts/'
    
    def upload_model_to_s3(self, local_path: str) -> str:
        """Завантаження моделі на S3"""
        import tarfile
        
        # Створюємо tar.gz архів
        model_archive = 'model.tar.gz'
        with tarfile.open(model_archive, 'w:gz') as tar:
            tar.add(local_path, arcname='.')
        
        # Завантажуємо на S3
        s3_path = f"s3://{self.bucket}/{self.prefix}/model.tar.gz"
        self.session.upload_data(
            path=model_archive,
            bucket=self.bucket,
            key_prefix=self.prefix
        )
        
        print(f"✅ Model uploaded to {s3_path}")
        return s3_path
    
    def create_multi_model_endpoint(self, model_s3_path: str) -> str:
        """Створення Multi-Model Endpoint"""
        if not self.role:
            raise ValueError("IAM роль не налаштована")
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        endpoint_name = f"bookstore-multi-model-{timestamp}"
        
        # Створюємо Multi-Model
        multi_model = MultiDataModel(
            name=f"bookstore-multi-{timestamp}",
            model_data_prefix=f"s3://{self.bucket}/{self.prefix}/models/",
            image_uri=sagemaker.image_uris.retrieve(
                framework="pytorch",
                region=self.region,
                version="1.13",
                py_version="py39",
                instance_type="ml.m5.large",
                image_scope="inference"
            ),
            role=self.role,
            sagemaker_session=self.session
        )
        
        # Додаємо нашу модель
        multi_model.add_model(model_s3_path, "bookstore-model")
        
        # Deploy endpoint
        print(f"🚀 Deploying endpoint: {endpoint_name}")
        predictor = multi_model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.large",
            endpoint_name=endpoint_name
        )
        
        print(f"✅ Endpoint deployed: {endpoint_name}")
        return endpoint_name, predictor
    
    def test_endpoint(self, predictor, endpoint_name: str):
        """Тестування endpoint"""
        test_data = {
            "model_name": "ensemble",
            "input_data": {
                "query": "фентезі книги про магію",
                "top_k": 2
            }
        }
        
        print(f"🧪 Testing endpoint {endpoint_name}...")
        try:
            result = predictor.predict(
                test_data,
                target_model="bookstore-model"
            )
            print("✅ Test successful!")
            print(f"📋 Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def deploy_complete_pipeline(self):
        """Повний pipeline деплою"""
        try:
            # 1. Створення artifacts
            print("🔨 Creating model artifacts...")
            artifacts_path = self.create_model_artifacts()
            
            # 2. Завантаження на S3
            print("📤 Uploading to S3...")
            s3_path = self.upload_model_to_s3(artifacts_path)
            
            # 3. Створення endpoint
            print("🚀 Creating multi-model endpoint...")
            endpoint_name, predictor = self.create_multi_model_endpoint(s3_path)
            
            # 4. Тестування
            print("🧪 Testing endpoint...")
            test_success = self.test_endpoint(predictor, endpoint_name)
            
            if test_success:
                print(f"🎉 Deployment successful!")
                print(f"📍 Endpoint: {endpoint_name}")
                print(f"🔗 Invoke URL: https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{endpoint_name}/invocations")
                
                return {
                    "endpoint_name": endpoint_name,
                    "predictor": predictor,
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
    # Перевіряємо AWS credentials
    try:
        boto3.Session().get_credentials()
        print("✅ AWS credentials found")
    except:
        print("❌ AWS credentials not found. Configure with 'aws configure'")
        return
    
    # Створюємо deployment
    deployment = SimpleAWSDeployment()
    
    # Запускаємо повний pipeline
    result = deployment.deploy_complete_pipeline()
    
    if result["status"] == "success":
        print("\n🎯 Deployment completed successfully!")
        print(f"Use endpoint: {result['endpoint_name']}")
    else:
        print(f"\n❌ Deployment failed: {result.get('reason', 'Unknown error')}")

if __name__ == "__main__":
    main()