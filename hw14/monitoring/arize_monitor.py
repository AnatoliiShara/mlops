# monitoring/arize_monitor.py

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
import uuid

@dataclass
class BookRecommendationPrediction:
    """Структура для логування предикшенів"""
    prediction_id: str
    query: str
    recommended_books: List[Dict]
    similarity_scores: List[float]
    response_quality_score: float
    embedding_drift_score: float
    timestamp: datetime
    user_feedback: Optional[int] = None  # 1-5 рейтинг від користувача
    click_through_rate: Optional[float] = None
    conversion_rate: Optional[float] = None

class ArizeBookstoreMonitor:
    """Arize моніторинг для RAG системи книжкового магазину"""
    
    def __init__(self):
        """Ініціалізація Arize клієнта"""
        self.api_key = os.getenv('ARIZE_API_KEY')
        self.space_key = os.getenv('ARIZE_SPACE_KEY')
        
        if not self.api_key or not self.space_key:
            raise ValueError("Встановіть ARIZE_API_KEY та ARIZE_SPACE_KEY в .env")
        
        self.arize_client = Client(space_key=self.space_key, api_key=self.api_key)
        self.model_id = "bookstore-rag-v1"
        self.model_version = "1.0.0"
        
        # Базові метрики для відстеження
        self.baseline_embedding_stats = None
        self.query_categories = [
            "жанр_пошук", "автор_пошук", "рейтинг_пошук", 
            "настрій_пошук", "тематичний_пошук", "загальний_пошук"
        ]
        
        print(f"✅ Arize Monitor ініціалізовано для моделі {self.model_id}")
    
    def categorize_query(self, query: str) -> str:
        """Категоризація запиту для моніторингу"""
        query_lower = query.lower()
        
        if any(genre in query_lower for genre in ["фантастика", "детектив", "роман", "історія"]):
            return "жанр_пошук"
        elif any(word in query_lower for word in ["автор", "письменник"]):
            return "автор_пошук" 
        elif any(word in query_lower for word in ["рейтинг", "кращі", "топ"]):
            return "рейтинг_пошук"
        elif any(word in query_lower for word in ["сумний", "веселий", "романтичний"]):
            return "настрій_пошук"
        elif any(word in query_lower for word in ["війна", "кохання", "пригоди"]):
            return "тематичний_пошук"
        else:
            return "загальний_пошук"
    
    def calculate_embedding_drift(self, current_embedding: np.ndarray) -> float:
        """Розрахунок дрифту embeddings відносно baseline"""
        if self.baseline_embedding_stats is None:
            return 0.0
        
        # Cosine similarity з baseline
        from sklearn.metrics.pairwise import cosine_similarity
        baseline_mean = self.baseline_embedding_stats['mean']
        similarity = cosine_similarity([current_embedding], [baseline_mean])[0][0]
        drift_score = 1 - similarity  # Чим більше дрифт, тим менше схожість
        
        return float(drift_score)
    
    def calculate_response_quality(self, query: str, response: str, books: List[Dict]) -> float:
        """Евристична оцінка якості відповіді"""
        quality_score = 0.5  # базова оцінка
        
        # Перевіряємо релевантність
        if len(books) > 0:
            quality_score += 0.2
        
        # Перевіряємо довжину відповіді
        if 50 <= len(response) <= 500:
            quality_score += 0.1
        
        # Перевіряємо наявність українських слів
        ukrainian_words = ["книга", "читати", "автор", "рекомендую"]
        if any(word in response.lower() for word in ukrainian_words):
            quality_score += 0.1
        
        # Перевіряємо структурованість
        if "📚" in response or "⭐" in response:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def set_baseline_embedding_stats(self, embeddings: List[np.ndarray]):
        """Встановлення baseline статистик для embeddings"""
        embeddings_array = np.array(embeddings)
        self.baseline_embedding_stats = {
            'mean': np.mean(embeddings_array, axis=0),
            'std': np.std(embeddings_array, axis=0),
            'timestamp': datetime.now(timezone.utc)
        }
        print(f"✅ Baseline embedding stats встановлено ({len(embeddings)} зразків)")
    
    def log_prediction(self, prediction: BookRecommendationPrediction):
        """Логування одного предикшену в Arize"""
        try:
            # Підготовка features
            features = {
                'query': prediction.query,
                'query_length': len(prediction.query),
                'query_category': self.categorize_query(prediction.query),
                'num_books_found': len(prediction.recommended_books),
                'avg_similarity_score': np.mean(prediction.similarity_scores) if prediction.similarity_scores else 0,
                'max_similarity_score': max(prediction.similarity_scores) if prediction.similarity_scores else 0,
                'embedding_drift_score': prediction.embedding_drift_score,
                'timestamp': prediction.timestamp
            }
            
            # Додаємо інформацію про знайдені книги
            if prediction.recommended_books:
                top_book = prediction.recommended_books[0]
                features.update({
                    'top_book_genre': top_book.get('genre', 'Unknown'),
                    'top_book_rating': top_book.get('rating', 0),
                    'top_book_title_length': len(top_book.get('title', ''))
                })
            
            # Підготовка predictions та actuals
            predictions_data = {
                'response_quality_predicted': prediction.response_quality_score,
                'relevance_score': np.mean(prediction.similarity_scores) if prediction.similarity_scores else 0
            }
            
            actuals_data = {}
            if prediction.user_feedback is not None:
                actuals_data['user_satisfaction'] = prediction.user_feedback
            if prediction.click_through_rate is not None:
                actuals_data['click_through_rate'] = prediction.click_through_rate
            if prediction.conversion_rate is not None:
                actuals_data['conversion_rate'] = prediction.conversion_rate
            
            # Створюємо DataFrame для Arize
            log_data = {
                'prediction_id': [prediction.prediction_id],
                **{f'feature_{k}': [v] for k, v in features.items()},
                **{f'prediction_{k}': [v] for k, v in predictions_data.items()},
                **{f'actual_{k}': [v] for k, v in actuals_data.items()}
            }
            
            df = pd.DataFrame(log_data)
            
            # Схема для Arize
            schema = Schema(
                prediction_id_column_name='prediction_id',
                timestamp_column_name='feature_timestamp',
                feature_column_names=[col for col in df.columns if col.startswith('feature_')],
                prediction_label_column_name='prediction_response_quality_predicted',
                actual_label_column_name='actual_user_satisfaction' if 'actual_user_satisfaction' in df.columns else None
            )
            
            # Відправляємо в Arize
            response = self.arize_client.log(
                dataframe=df,
                model_id=self.model_id,
                model_version=self.model_version,
                model_type=ModelTypes.SCORE_CATEGORICAL,
                environment=Environments.PRODUCTION,
                schema=schema
            )
            
            if response.status_code == 200:
                print(f"✅ Prediction {prediction.prediction_id} logged to Arize")
            else:
                print(f"❌ Failed to log prediction: {response.text}")
                
        except Exception as e:
            print(f"❌ Error logging to Arize: {e}")
    
    def log_batch_predictions(self, predictions: List[BookRecommendationPrediction]):
        """Батчеве логування предикшенів"""
        print(f"📊 Логуємо {len(predictions)} предикшенів в Arize...")
        
        for prediction in predictions:
            self.log_prediction(prediction)
    
    def create_drift_alert(self, threshold: float = 0.1):
        """Створення алерту для дрифту (через Arize UI API)"""
        # Це зазвичай налаштовується через Arize UI, але можна автоматизувати
        alert_config = {
            "name": "Embedding Drift Alert",
            "metric": "embedding_drift_score",
            "threshold": threshold,
            "comparison": "greater_than",
            "evaluation_window": "24h"
        }
        print(f"📢 Alert config for embedding drift: {alert_config}")
        return alert_config
    
    def get_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Отримання метрик перформансу з Arize (через API)"""
        # Це симуляція - реальний Arize API має інші ендпоінти
        metrics = {
            "total_predictions": 1000,
            "avg_response_quality": 0.75,
            "avg_user_satisfaction": 4.2,
            "drift_incidents": 3,
            "query_categories_distribution": {
                "жанр_пошук": 0.35,
                "загальний_пошук": 0.25,
                "рейтинг_пошук": 0.20,
                "тематичний_пошук": 0.15,
                "настрій_пошук": 0.05
            }
        }
        return metrics

# Утилітарні функції для інтеграції з RAG системою
class RAGArizeIntegration:
    """Інтеграція Arize моніторингу з RAG системою"""
    
    def __init__(self, rag_engine, arize_monitor: ArizeBookstoreMonitor):
        self.rag_engine = rag_engine
        self.arize_monitor = arize_monitor
        
    async def monitored_search_and_respond(self, query: str, top_k: int = 5) -> Dict:
        """Обгортка для RAG пошуку з моніторингом"""
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Виконуємо стандартний пошук
        result = await self.rag_engine.search_and_respond(query, top_k)
        
        if result['status'] == 'success':
            # Створюємо embedding для запиту для drift detection
            query_embedding = self.rag_engine.create_query_embedding(query)
            drift_score = self.arize_monitor.calculate_embedding_drift(query_embedding)
            
            # Розраховуємо якість відповіді
            quality_score = self.arize_monitor.calculate_response_quality(
                query, result['response'], result['found_books']
            )
            
            # Створюємо предикшен для логування
            prediction = BookRecommendationPrediction(
                prediction_id=prediction_id,
                query=query,
                recommended_books=result['found_books'],
                similarity_scores=[book['similarity_score'] for book in result['found_books']],
                response_quality_score=quality_score,
                embedding_drift_score=drift_score,
                timestamp=timestamp
            )
            
            # Логуємо в Arize
            self.arize_monitor.log_prediction(prediction)
            
            # Додаємо monitoring metadata до результату
            result['monitoring'] = {
                'prediction_id': prediction_id,
                'quality_score': quality_score,
                'drift_score': drift_score,
                'timestamp': timestamp.isoformat()
            }
        
        return result
    
    def log_user_feedback(self, prediction_id: str, user_rating: int, clicked_books: List[str] = None):
        """Логування зворотного зв'язку від користувача"""
        # Тут ми б оновили запис в Arize з actual значеннями
        feedback_data = {
            'prediction_id': prediction_id,
            'user_satisfaction': user_rating,
            'click_through_rate': len(clicked_books) / 5 if clicked_books else 0  # Припускаємо 5 рекомендацій
        }
        
        print(f"📝 User feedback logged for {prediction_id}: {feedback_data}")
        # В реальному проекті тут був би виклик Arize API для оновлення actual значень

# Приклад використання
def example_usage():
    """Приклад використання Arize моніторингу"""
    from rag_engine import RAGEngine
    import asyncio
    
    async def main():
        # Ініціалізація
        rag_engine = RAGEngine()
        arize_monitor = ArizeBookstoreMonitor()
        integration = RAGArizeIntegration(rag_engine, arize_monitor)
        
        # Встановлення baseline (зазвичай робиться один раз)
        # baseline_embeddings = [rag_engine.create_query_embedding(q) for q in sample_queries]
        # arize_monitor.set_baseline_embedding_stats(baseline_embeddings)
        
        # Тестовий запит з моніторингом
        result = await integration.monitored_search_and_respond("пригодницькі книги для дітей")
        print(f"📊 Monitored result: {result['monitoring']}")
        
        # Симуляція user feedback
        if 'monitoring' in result:
            integration.log_user_feedback(
                result['monitoring']['prediction_id'], 
                user_rating=4,
                clicked_books=['book1', 'book2']
            )
    
    asyncio.run(main())

if __name__ == "__main__":
    example_usage()