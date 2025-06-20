# pr1_aws/models/multi_model.py
# AWS SageMaker специфічний адаптер для наших моделей

import json
import sys
import os
import logging

# Додаємо шлях до shared модулів (для AWS SageMaker)
sys.path.append('/opt/ml/code')
sys.path.append('/opt/ml/code/shared')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared'))

try:
    from book_models import create_multi_model_predictor
except ImportError:
    # Fallback для локальної розробки
    import sys
    sys.path.append('../../shared')
    from book_models import create_multi_model_predictor

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальна змінна для предиктора (AWS SageMaker pattern)
predictor = None

def model_fn(model_dir):
    """
    AWS SageMaker функція для завантаження моделі
    Викликається один раз при старті контейнера
    """
    global predictor
    try:
        logger.info("🔄 Loading multi-model predictor for AWS SageMaker...")
        predictor = create_multi_model_predictor()
        logger.info("✅ Multi-model predictor loaded successfully")
        return predictor
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def input_fn(request_body, request_content_type):
    """
    AWS SageMaker функція для обробки вхідних даних
    """
    logger.info(f"📥 Processing input with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        try:
            input_data = json.loads(request_body)
            logger.info(f"📋 Input data: {input_data}")
            return input_data
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            raise ValueError(f"Invalid JSON: {e}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    AWS SageMaker функція для інференсу
    """
    global predictor
    
    try:
        logger.info("🔮 Starting prediction...")
        
        # Отримуємо параметри з input_data
        model_name = input_data.get('model_name', 'ensemble')
        input_payload = input_data.get('input_data', {})
        
        logger.info(f"🎯 Model: {model_name}, Input: {input_payload}")
        
        # Викликаємо предикцію
        result = predictor.predict(model_name, input_payload)
        
        # Додаємо AWS-specific metadata
        result['aws_metadata'] = {
            'sagemaker_endpoint': True,
            'instance_type': os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'unknown'),
            'region': os.environ.get('AWS_DEFAULT_REGION', 'unknown')
        }
        
        logger.info("✅ Prediction completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return {
            'error': str(e),
            'model_name': input_data.get('model_name', 'unknown'),
            'aws_metadata': {
                'error_location': 'predict_fn',
                'sagemaker_endpoint': True
            }
        }

def output_fn(prediction, content_type):
    """
    AWS SageMaker функція для форматування виводу
    """
    logger.info(f"📤 Formatting output with content type: {content_type}")
    
    if content_type == 'application/json':
        try:
            output = json.dumps(prediction, ensure_ascii=False, indent=2)
            logger.info("✅ Output formatted successfully")
            return output
        except Exception as e:
            logger.error(f"❌ Output formatting failed: {e}")
            return json.dumps({
                'error': f'Output formatting failed: {e}',
                'original_prediction': str(prediction)
            })
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# AWS SageMaker Multi-Model специфічні функції
def multi_model_fn(model_dir, model_name):
    """
    Функція для Multi-Model Endpoints в AWS SageMaker
    Дозволяє завантажувати різні версії моделі
    """
    logger.info(f"🔄 Loading model '{model_name}' from {model_dir}")
    
    # В нашому випадку всі моделі об'єднані в один predictor
    # Але можна розширити для завантаження різних версій
    if model_name in ['bookstore-model', 'ensemble-model', 'v1', 'latest']:
        return model_fn(model_dir)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Health check для AWS SageMaker
def health_check():
    """
    Health check функція для AWS SageMaker
    """
    global predictor
    
    if predictor is None:
        return {
            'status': 'unhealthy',
            'reason': 'Model not loaded',
            'aws_sagemaker': True
        }
    
    try:
        # Тестуємо базову функціональність
        test_result = predictor.predict('recommendation', {
            'query': 'test query',
            'top_k': 1
        })
        
        if 'error' in test_result:
            return {
                'status': 'unhealthy',
                'reason': f'Model test failed: {test_result["error"]}',
                'aws_sagemaker': True
            }
        
        return {
            'status': 'healthy',
            'models_available': ['embedding', 'recommendation', 'quality', 'ensemble'],
            'aws_sagemaker': True,
            'version': '1.0.0'
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy', 
            'reason': str(e),
            'aws_sagemaker': True
        }

# Для локального тестування AWS адаптера
if __name__ == "__main__":
    print("🧪 Testing AWS SageMaker adapter locally...")
    
    # Симулюємо AWS SageMaker workflow
    try:
        # 1. Model loading
        model = model_fn("/tmp/model")
        print("✅ Model loaded")
        
        # 2. Input processing
        test_input = json.dumps({
            "model_name": "ensemble",
            "input_data": {
                "query": "фентезі книги про магію",
                "top_k": 2
            }
        })
        
        processed_input = input_fn(test_input, "application/json")
        print("✅ Input processed")
        
        # 3. Prediction
        result = predict_fn(processed_input, model)
        print("✅ Prediction completed")
        
        # 4. Output formatting
        output = output_fn(result, "application/json")
        print("✅ Output formatted")
        
        print("\n📋 Final result:")
        print(output)
        
        # 5. Health check
        health = health_check()
        print(f"\n🏥 Health check: {health['status']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()