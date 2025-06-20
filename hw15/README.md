# HW15: Multi Model Deployment on AWS & GCP

## Завдання

**PR1**: Multi Model Deployment на AWS SageMaker  
**PR2**: Multi Model Deployment на GCP Vertex AI

## Архітектура

Система розгортає **4 моделі** одночасно:
- **Embedding Model**: Sentence Transformers для векторизації
- **Recommendation Model**: Логіка рекомендацій книг
- **Quality Model**: Оцінка якості рекомендацій  
- **Ensemble Model**: Комбінований pipeline

## Структура проекту

```
hw15/
├── pr1_aws/                   # PR1: AWS SageMaker
│   ├── models/
│   │   └── multi_model.py     # AWS-specific модель
│   ├── deployment/
│   │   └── deploy_aws.py      # AWS deployment script
│   ├── Dockerfile             # AWS контейнер
│   └── README.md
│
├── pr2_gcp/                   # PR2: GCP Vertex AI  
│   ├── models/
│   │   └── multi_model.py     # GCP-specific модель
│   ├── deployment/
│   │   └── deploy_gcp.py      # GCP deployment script
│   ├── Dockerfile             # GCP контейнер
│   └── README.md
│
├── shared/                    # Спільні компоненти
│   ├── book_models.py         # Базові ML моделі
│   └── api_handler.py         # FastAPI handler
│
├── requirements.txt           # Python залежності
└── README.md                  # Цей файл
```

## Швидкий старт

### 1. Встановлення залежностей
```bash
cd hw15
pip install -r requirements.txt
```

### 2. Локальне тестування
```bash
cd shared
python api_handler.py
# Відкрийте http://localhost:8000/docs
```

### 3. AWS Deployment (PR1)
```bash
cd pr1_aws/deployment
python deploy_aws.py
```

### 4. GCP Deployment (PR2)  
```bash
cd pr2_gcp/deployment
python deploy_gcp.py
```

## Налаштування

### AWS (PR1)
```bash
# AWS credentials
aws configure

# Створіть IAM роль SageMakerExecutionRole
# Або встановіть змінну:
export SAGEMAKER_ROLE_ARN="arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole"
```

### GCP (PR2)
```bash
# GCP credentials  
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Або встановіть змінну:
export GCP_PROJECT_ID="your-project-id"
```

## Функціональність

### Multi-Model API Endpoints

**Доступні моделі:**
- `embedding`: Генерація embeddings
- `recommendation`: Рекомендації книг
- `quality`: Оцінка якості
- `ensemble`: Повний pipeline

**Приклади запитів:**

```python
# Ensemble (рекомендації + якість)
{
  "model_name": "ensemble",
  "input_data": {
    "query": "пригодницькі книги про магію",
    "top_k": 3
  }
}

# Тільки рекомендації
{
  "model_name": "recommendation", 
  "input_data": {
    "query": "романтичні романи",
    "top_k": 2
  }
}

# Тільки embeddings
{
  "model_name": "embedding",
  "input_data": {
    "texts": ["Гаррі Поттер", "Володар Перснів"]
  }
}
```

## Порівняння AWS vs GCP

| Характеристика | AWS SageMaker | GCP Vertex AI |
|---------------|---------------|---------------|
| **Multi-Model Support** | ✅ Native | ✅ Custom Container |
| **Auto-scaling** | ✅ Built-in | ✅ Built-in |
| **Container Registry** | ECR | GCR |
| **Model Registry** | SageMaker Model Registry | Vertex Model Registry |
| **Monitoring** | CloudWatch | Cloud Monitoring |
| **Cost** | Pay per inference | Pay per inference |

## Тестування

### Локальне тестування
```bash
cd shared
python -c "
from book_models import create_multi_model_predictor
predictor = create_multi_model_predictor()
result = predictor.predict('ensemble', {'query': 'фентезі', 'top_k': 2})
print(result)
"
```

### AWS Endpoint тест
```python
import boto3

client = boto3.client('sagemaker-runtime')
response = client.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps({
        "model_name": "ensemble",
        "input_data": {"query": "фентезі книги", "top_k": 2}
    })
)
```

### GCP Endpoint тест
```python
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint('projects/PROJECT/locations/REGION/endpoints/ENDPOINT_ID')
predictions = endpoint.predict(instances=[{
    "model_name": "ensemble", 
    "input_data": {"query": "фентезі книги", "top_k": 2}
}])
```

## 🎓 Ключові особливості

### AWS SageMaker Features:
- **Multi-Model Endpoints**: Один endpoint для кількох моделей
- **Dynamic Model Loading**: Завантаження моделей on-demand
- **Built-in Monitoring**: CloudWatch integration
- **Auto-scaling**: Automatic instance scaling

### GCP Vertex AI Features:
- **Custom Containers**: Повний контроль над середовищем
- **Traffic Splitting**: A/B testing between models
- **Managed Infrastructure**: Serverless deployment
- **ML Monitoring**: Built-in model monitoring

## 📈 Production Ready Features

- **Health Checks**: `/health` endpoints
- **Error Handling**: Graceful error responses  
- **Logging**: Structured logging
- **Monitoring**: Cloud-native monitoring
- **Auto-scaling**: Dynamic scaling based on load
- **Model Versioning**: Support for multiple model versions

## 🔍 Troubleshooting

### Загальні проблеми:

**AWS Issues:**
```bash
# IAM роль проблеми
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json

# S3 bucket permissions
aws s3 ls s3://your-bucket-name
```

**GCP Issues:**
```bash
# Автентифікація
gcloud auth application-default login

# Docker permissions  
gcloud auth configure-docker
```

## Автори

