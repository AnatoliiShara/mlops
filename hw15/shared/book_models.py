# shared/book_models.py
# Прості моделі для book recommendation

import json
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import random

class SimpleEmbeddingModel:
    """Проста модель для створення embeddings"""
    
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.name = "embedding_model"
        self.version = "1.0"
    
    def predict(self, texts: List[str]) -> List[List[float]]:
        """Створення embeddings для текстів"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

class SimpleRecommendationModel:
    """Проста модель рекомендацій"""
    
    def __init__(self):
        self.name = "recommendation_model" 
        self.version = "1.0"
        # Симуляція бази книг
        self.books = [
            {"id": 1, "title": "Гаррі Поттер", "genre": "Фентезі", "rating": 4.8},
            {"id": 2, "title": "Володар Перснів", "genre": "Фентезі", "rating": 4.9},
            {"id": 3, "title": "1984", "genre": "Антиутопія", "rating": 4.7},
            {"id": 4, "title": "Гра престолів", "genre": "Фентезі", "rating": 4.6},
            {"id": 5, "title": "Майстер і Маргарита", "genre": "Класика", "rating": 4.8}
        ]
    
    def predict(self, query: str, top_k: int = 3) -> List[Dict]:
        """Рекомендація книг за запитом"""
        # Проста логіка рекомендацій
        if "фентезі" in query.lower() or "магія" in query.lower():
            filtered = [b for b in self.books if b["genre"] == "Фентезі"]
        elif "класика" in query.lower():
            filtered = [b for b in self.books if b["genre"] == "Класика"]
        else:
            filtered = self.books.copy()
        
        # Сортуємо за рейтингом та повертаємо top_k
        filtered.sort(key=lambda x: x["rating"], reverse=True)
        return filtered[:top_k]

class SimpleQualityModel:
    """Проста модель оцінки якості"""
    
    def __init__(self):
        self.name = "quality_model"
        self.version = "1.0"
    
    def predict(self, query: str, recommendations: List[Dict]) -> float:
        """Оцінка якості рекомендацій"""
        # Проста евристика
        quality_score = 0.5  # base score
        
        if len(recommendations) > 0:
            quality_score += 0.2
        
        avg_rating = sum(book["rating"] for book in recommendations) / len(recommendations) if recommendations else 0
        if avg_rating > 4.5:
            quality_score += 0.2
        
        if len(query) > 10:  # детальний запит
            quality_score += 0.1
        
        return min(quality_score, 1.0)

class MultiModelPredictor:
    """Мульти-модель предиктор який комбінує всі моделі"""
    
    def __init__(self):
        self.embedding_model = SimpleEmbeddingModel()
        self.recommendation_model = SimpleRecommendationModel() 
        self.quality_model = SimpleQualityModel()
        self.models = {
            "embedding": self.embedding_model,
            "recommendation": self.recommendation_model,
            "quality": self.quality_model
        }
    
    def predict(self, model_name: str, input_data: Dict) -> Dict:
        """Роутинг до потрібної моделі"""
        try:
            if model_name == "embedding":
                texts = input_data.get("texts", [])
                embeddings = self.embedding_model.predict(texts)
                return {
                    "embeddings": embeddings,
                    "model": "embedding_model",
                    "version": "1.0"
                }
            
            elif model_name == "recommendation":
                query = input_data.get("query", "")
                top_k = input_data.get("top_k", 3)
                recommendations = self.recommendation_model.predict(query, top_k)
                return {
                    "recommendations": recommendations,
                    "model": "recommendation_model", 
                    "version": "1.0"
                }
            
            elif model_name == "quality":
                query = input_data.get("query", "")
                recommendations = input_data.get("recommendations", [])
                quality_score = self.quality_model.predict(query, recommendations)
                return {
                    "quality_score": quality_score,
                    "model": "quality_model",
                    "version": "1.0"
                }
            
            elif model_name == "ensemble":
                # Комбінований пайплайн
                query = input_data.get("query", "")
                top_k = input_data.get("top_k", 3)
                
                # 1. Отримуємо рекомендації
                recommendations = self.recommendation_model.predict(query, top_k)
                
                # 2. Оцінюємо якість
                quality_score = self.quality_model.predict(query, recommendations)
                
                # 3. Створюємо embedding запиту
                query_embedding = self.embedding_model.predict([query])[0]
                
                return {
                    "query": query,
                    "recommendations": recommendations,
                    "quality_score": quality_score,
                    "query_embedding": query_embedding,
                    "model": "ensemble_model",
                    "version": "1.0"
                }
            
            else:
                return {"error": f"Unknown model: {model_name}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def health_check(self) -> Dict:
        """Health check для всіх моделей"""
        return {
            "status": "healthy",
            "models": {
                name: {"status": "ready", "version": model.version}
                for name, model in self.models.items()
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }

# Factory для створення моделей
def create_multi_model_predictor() -> MultiModelPredictor:
    """Фабрика для створення мульти-модель предиктора"""
    return MultiModelPredictor()

# Для тестування
if __name__ == "__main__":
    predictor = create_multi_model_predictor()
    
    # Тест ensemble
    result = predictor.predict("ensemble", {
        "query": "пригодницькі книги про магію",
        "top_k": 2
    })
    
    print("🧪 Test result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))