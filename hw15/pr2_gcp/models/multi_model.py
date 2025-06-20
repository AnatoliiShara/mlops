# pr2_gcp/models/multi_model.py
# GCP Vertex AI специфічний адаптер для наших моделей

import json
import sys
import os
import logging
from typing import Dict, List, Any
import time

# Додаємо шлях до shared модулів (для GCP Vertex AI)
sys.path.append('/app')
sys.path.append('/app/shared')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared'))

try:
    from book_models import create_multi_model_predictor
except ImportError:
    # Fallback для локальної розробки
    import sys
    sys.path.append('../../shared')
    from book_models import create_multi_model_predictor

# Налаштування логування для GCP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VertexAIPredictor:
    """
    GCP Vertex AI специфічний клас для multi-model prediction
    """
    
    def __init__(self):
        """Ініціалізація предиктора"""
        self.predictor = None
        self.model_loaded = False
        self.load_time = None
        self.prediction_count = 0
        
    def load_model(self):
        """Завантаження моделі для Vertex AI"""
        try:
            logger.info("🔄 Loading multi-model predictor for GCP Vertex AI...")
            start_time = time.time()
            
            self.predictor = create_multi_model_predictor()
            self.load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.info(f"✅ Multi-model predictor loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model_loaded = False
            raise
    
    def predict(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        GCP Vertex AI prediction функція
        
        Args:
            instances: List of prediction instances in Vertex AI format
            
        Returns:
            List of predictions
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        predictions = []
        
        for instance in instances:
            try:
                start_time = time.time()
                
                # Отримуємо параметри з instance
                model_name = instance.get('model_name', 'ensemble')
                input_data = instance.get('input_data', {})
                
                logger.info(f"🎯 Processing instance: model={model_name}")
                
                # Викликаємо предикцію
                result = self.predictor.predict(model_name, input_data)
                
                # Додаємо GCP-specific metadata
                prediction_time = time.time() - start_time
                self.prediction_count += 1
                
                result['gcp_metadata'] = {
                    'vertex_ai': True,
                    'prediction_time_ms': round(prediction_time * 1000, 2),
                    'prediction_id': f"pred_{self.prediction_count}",
                    'instance_id': os.environ.get('HOSTNAME', 'unknown'),
                    'project_id': os.environ.get('GOOGLE_CLOUD_PROJECT', 'unknown'),
                    'region': os.environ.get('GOOGLE_CLOUD_REGION', 'unknown')
                }
                
                predictions.append(result)
                logger.info(f"✅ Prediction completed in {prediction_time:.3f}s")
                
            except Exception as e:
                logger.error(f"❌ Prediction failed for instance: {e}")
                
                error_result = {
                    'error': str(e),
                    'model_name': instance.get('model_name', 'unknown'),
                    'gcp_metadata': {
                        'vertex_ai': True,
                        'error_location': 'predict',
                        'prediction_id': f"error_{self.prediction_count}"
                    }
                }
                predictions.append(error_result)
        
        return predictions
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health check для GCP Vertex AI
        """
        if not self.model_loaded:
            return {
                'status': 'unhealthy',
                'reason': 'Model not loaded',
                'vertex_ai': True
            }
        
        try:
            # Тестуємо базову функціональність
            test_result = self.predictor.predict('recommendation', {
                'query': 'test query',
                'top_k': 1
            })
            
            if 'error' in test_result:
                return {
                    'status': 'unhealthy',
                    'reason': f'Model test failed: {test_result["error"]}',
                    'vertex_ai': True
                }
            
            return {
                'status': 'healthy',
                'models_available': ['embedding', 'recommendation', 'quality', 'ensemble'],
                'vertex_ai': True,
                'version': '1.0.0',
                'load_time_seconds': self.load_time,
                'predictions_served': self.prediction_count,
                'uptime_info': {
                    'instance_id': os.environ.get('HOSTNAME', 'unknown'),
                    'project_id': os.environ.get('GOOGLE_CLOUD_PROJECT', 'unknown')
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'reason': str(e),
                'vertex_ai': True
            }

# Глобальний інстанс предиктора
vertex_predictor = VertexAIPredictor()

# GCP Vertex AI custom prediction functions
def load_model():
    """Завантаження моделі при старті Vertex AI контейнера"""
    return vertex_predictor.load_model()

def predict(instances: List[Dict]) -> List[Dict]:
    """
    Головна prediction функція для Vertex AI
    
    Vertex AI очікує функцію з такою сигнатурою:
    predict(instances: List[Dict]) -> List[Dict]
    """
    return vertex_predictor.predict(instances)

def health():
    """Health check endpoint для Vertex AI"""
    return vertex_predictor.health_check()

# Функції для custom container (якщо використовуємо Flask/FastAPI)
class VertexAIFlaskHandler:
    """
    Handler для Flask/FastAPI в custom Vertex AI container
    """
    
    @staticmethod
    def handle_predict_request(request_data: Dict) -> Dict:
        """
        Обробка Vertex AI prediction request
        """
        try:
            # Vertex AI може передавати дані в різних форматах
            if 'instances' in request_data:
                # Стандартний Vertex AI формат
                instances = request_data['instances']
                predictions = predict(instances)
                return {'predictions': predictions}
            
            elif 'model_name' in request_data and 'input_data' in request_data:
                # Прямий формат
                instances = [request_data]
                predictions = predict(instances)
                return predictions[0]  # Повертаємо одну предикцію
            
            else:
                # Fallback - припускаємо ensemble
                instances = [{
                    'model_name': 'ensemble',
                    'input_data': request_data
                }]
                predictions = predict(instances)
                return predictions[0]
                
        except Exception as e:
            logger.error(f"❌ Request handling failed: {e}")
            return {
                'error': str(e),
                'vertex_ai': True,
                'error_location': 'handle_predict_request'
            }
    
    @staticmethod
    def handle_health_request() -> Dict:
        """Обробка health check request"""
        return health()

# Vertex AI Model Server специфічні функції
def preprocess(instances: List[Dict]) -> List[Dict]:
    """
    Preprocessing для Vertex AI Model Server
    """
    logger.info(f"🔄 Preprocessing {len(instances)} instances")
    
    # В нашому випадку preprocessing не потрібен
    # Але можна додати валідацію, нормалізацію тощо
    processed = []
    
    for instance in instances:
        # Валідація обов'язкових полів
        if 'model_name' not in instance:
            instance['model_name'] = 'ensemble'  # default
        
        if 'input_data' not in instance:
            instance['input_data'] = {}
        
        processed.append(instance)
    
    logger.info("✅ Preprocessing completed")
    return processed

def postprocess(predictions: List[Dict]) -> List[Dict]:
    """
    Postprocessing для Vertex AI Model Server
    """
    logger.info(f"🔄 Postprocessing {len(predictions)} predictions")
    
    # Можна додати додаткову обробку результатів
    # Наприклад, фільтрацію, форматування тощо
    
    processed = []
    for prediction in predictions:
        # Додаємо timestamp
        prediction['timestamp'] = time.time()
        
        # Фільтруємо великі embedding'и для економії bandwidth
        if 'query_embedding' in prediction and len(prediction['query_embedding']) > 100:
            prediction['query_embedding_size'] = len(prediction['query_embedding'])
            del prediction['query_embedding']  # Видаляємо для економії
        
        processed.append(prediction)
    
    logger.info("✅ Postprocessing completed")
    return processed

# Для локального тестування GCP адаптера
if __name__ == "__main__":
    print("🧪 Testing GCP Vertex AI adapter locally...")
    
    try:
        # 1. Model loading
        print("🔄 Loading model...")
        success = load_model()
        if not success:
            raise Exception("Model loading failed")
        print("✅ Model loaded")
        
        # 2. Test prediction with Vertex AI format
        test_instances = [
            {
                "model_name": "ensemble",
                "input_data": {
                    "query": "фентезі книги про магію",
                    "top_k": 2
                }
            },
            {
                "model_name": "recommendation",
                "input_data": {
                    "query": "романтичні романи",
                    "top_k": 1
                }
            }
        ]
        
        print("🔄 Running predictions...")
        predictions = predict(test_instances)
        print("✅ Predictions completed")
        
        print("\n📋 Results:")
        for i, pred in enumerate(predictions):
            print(f"Prediction {i+1}: {json.dumps(pred, ensure_ascii=False, indent=2)}")
        
        # 3. Health check
        health_status = health()
        print(f"\n🏥 Health check: {health_status['status']}")
        print(f"📊 Predictions served: {health_status.get('predictions_served', 0)}")
        
        # 4. Test Flask handler
        print("\n🧪 Testing Flask handler...")
        handler = VertexAIFlaskHandler()
        
        flask_request = {
            "instances": test_instances
        }
        
        flask_result = handler.handle_predict_request(flask_request)
        print(f"✅ Flask handler test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()