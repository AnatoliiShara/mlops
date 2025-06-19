# src/rag_engine.py - ОНОВЛЕНА ВЕРСІЯ З МОНІТОРИНГОМ

import os
import json
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Імпорт Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Generative AI не встановлено. Запустіть: pip install google-generativeai")

# Імпорт моніторингу
MONITORING_ENABLED = os.getenv('MONITORING_ENABLED', 'false').lower() == 'true'

if MONITORING_ENABLED:
    try:
        from monitoring.arize_monitor import ArizeBookstoreMonitor, RAGArizeIntegration
        from monitoring.langsmith_monitor import LangSmithBookstoreMonitor, RAGLangSmithIntegration
        MONITORING_AVAILABLE = True
        print("✅ Monitoring modules imported")
    except ImportError as e:
        MONITORING_AVAILABLE = False
        print(f"⚠️ Monitoring не доступний: {e}")
else:
    MONITORING_AVAILABLE = False
    print("📊 Monitoring вимкнено")

load_dotenv()

class RAGEngine:
    def __init__(self):
        """Ініціалізація RAG движка з опціональним моніторингом"""
        # Завантажуємо модель для embeddings
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Налаштування для Google Gemini API
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        if self.google_api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.google_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.use_gemini = True
            print("🤖 Використовуємо Google Gemini API")
        else:
            self.use_gemini = False
            print("⚠️ Gemini API не налаштовано, використовуємо fallback")
        
        # Ініціалізація моніторингу
        self.monitoring_enabled = MONITORING_AVAILABLE and MONITORING_ENABLED
        if self.monitoring_enabled:
            try:
                self.arize_monitor = ArizeBookstoreMonitor()
                self.langsmith_monitor = LangSmithBookstoreMonitor()
                self.arize_integration = RAGArizeIntegration(self, self.arize_monitor)
                self.langsmith_integration = RAGLangSmithIntegration(self, self.langsmith_monitor)
                print("📊 Monitoring ініціалізовано")
            except Exception as e:
                print(f"⚠️ Помилка ініціалізації моніторингу: {e}")
                self.monitoring_enabled = False
        
        # Завантажуємо оброблені книги
        self.books_data = []
        self.embeddings_matrix = None
        self.load_processed_books()
        
        print("✅ RAG Engine ініціалізовано!")
    
    def load_processed_books(self, data_path: str = "data/test_processed_books.json"):
        """Завантаження оброблених книг з JSON файлу"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.books_data = json.load(f)
            
            # Створюємо матрицю embeddings для швидкого пошуку
            embeddings_list = [book['embedding'] for book in self.books_data]
            self.embeddings_matrix = np.array(embeddings_list)
            
            print(f"✅ Завантажено {len(self.books_data)} книг")
            print(f"📊 Розмір embeddings матриці: {self.embeddings_matrix.shape}")
            
        except FileNotFoundError:
            print(f"❌ Файл {data_path} не знайдено. Запустіть спочатку data_processor.py")
            self.books_data = []
            self.embeddings_matrix = None
        except Exception as e:
            print(f"❌ Помилка при завантаженні даних: {e}")
            self.books_data = []
            self.embeddings_matrix = None
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Створення embedding для пошукового запиту"""
        return self.model.encode([query])[0]
    
    def search_similar_books(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Пошук схожих книг за запитом"""
        if self.embeddings_matrix is None or len(self.books_data) == 0:
            return []
        
        # Створюємо embedding для запиту
        query_embedding = self.create_query_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Обчислюємо cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        # Знаходимо топ-k найсхожіших книг
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            book = self.books_data[idx].copy()
            similarity_score = float(similarities[idx])
            # Видаляємо embedding з результату
            book.pop('embedding', None)
            results.append((book, similarity_score))
        
        return results
    
    def format_context_for_llm(self, search_results: List[Tuple[Dict, float]]) -> str:
        """Форматування результатів пошуку для LLM"""
        if not search_results:
            return "Книги не знайдено."
        
        context_parts = []
        for i, (book, score) in enumerate(search_results, 1):
            context_part = f"""
Книга {i}:
📚 Назва: {book['title']}
🎭 Жанр: {book['genre']}
⭐ Рейтинг: {book['rating']}/5
📖 Опис: {book['description'][:300]}...
🔍 Релевантність: {score:.3f}
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)
    
    def generate_simple_response(self, query: str, search_results: List[Tuple[Dict, float]]) -> str:
        """Генерація простої відповіді без LLM"""
        if not search_results:
            return "Вибачте, не знайшов підходящих книг за вашим запитом."
        
        response_parts = [f"🔍 За запитом '{query}' знайдено {len(search_results)} підходящих книг:\n"]
        
        for i, (book, score) in enumerate(search_results, 1):
            response_part = f"""
📚 {i}. "{book['title']}"
   🎭 Жанр: {book['genre']}
   ⭐ Рейтинг: {book['rating']}/5
   📖 {book['description'][:150]}...
   🎯 Відповідність: {score:.0%}
"""
            response_parts.append(response_part.strip())
        
        # Додаємо рекомендацію
        best_book = search_results[0][0]
        response_parts.append(f"\n💡 Особливо рекомендую '{best_book['title']}' - найкраща відповідність вашому запиту!")
        
        return "\n".join(response_parts)

    async def generate_response_with_gemini(self, query: str, context: str) -> str:
        """Генерація відповіді через Google Gemini API"""
        if not self.use_gemini:
            return None
            
        prompt = f"""Ти - експерт-консультант у книжковому магазині в Україні. На основі наступної інформації про книги, дай персоналізовану рекомендацію клієнту.

ЗАПИТ КЛІЄНТА: {query}

ЗНАЙДЕНІ КНИГИ:
{context}

ІНСТРУКЦІЇ:
1. Дай детальну відповідь українською мовою
2. Рекомендуй найбільш підходящі книги з списку
3. Поясни чому саме ці книги підходять клієнту
4. Додай цікаві деталі про книги та їх особливості
5. Будь дружнім, ентузіастичним та допомагаючим
6. Використовуй емодзі для кращого сприйняття
7. Завершуй рекомендацією найкращого варіанту

ВІДПОВІДЬ:"""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.7,
                )
            )
            
            if response.text:
                return response.text
            else:
                print("⚠️ Gemini не повернув відповідь, використовуємо fallback")
                return None
                
        except Exception as e:
            print(f"⚠️ Gemini API помилка ({e}), використовуємо fallback")
            return None
    
    async def search_and_respond(self, query: str, top_k: int = 5) -> Dict:
        """Основний метод RAG: пошук + генерація відповіді з моніторингом"""
        print(f"🔍 Обробляємо запит: {query}")
        
        # Якщо моніторинг увімкнено, використовуємо інтеграцію
        if self.monitoring_enabled and hasattr(self, 'arize_integration'):
            return await self.arize_integration.monitored_search_and_respond(query, top_k)
        
        # Стандартний потік без моніторингу
        search_results = self.search_similar_books(query, top_k)
        
        if not search_results:
            return {
                "query": query,
                "found_books": [],
                "response": "Вибачте, не знайшов підходящих книг за вашим запитом. Спробуйте уточнити пошук.",
                "status": "no_results"
            }
        
        context = self.format_context_for_llm(search_results)
        llm_response = await self.generate_response_with_gemini(query, context)
        
        if llm_response is None:
            llm_response = self.generate_simple_response(query, search_results)
        
        result = {
            "query": query,
            "found_books": [
                {
                    "title": book['title'],
                    "genre": book['genre'],
                    "rating": book['rating'],
                    "description": book['description'][:200] + "...",
                    "similarity_score": score
                }
                for book, score in search_results
            ],
            "response": llm_response,
            "status": "success"
        }
        
        print(f"✅ Знайдено {len(search_results)} книг, відповідь згенеровано")
        return result
    
    def get_stats(self) -> Dict:
        """Статистика RAG системи"""
        if not self.books_data:
            return {"total_books": 0, "status": "no_data"}
        
        genres = {}
        ratings = []
        
        for book in self.books_data:
            genre = book.get('genre', 'Невідомо')
            genres[genre] = genres.get(genre, 0) + 1
            
            rating = book.get('rating', 0)
            if rating:
                ratings.append(rating)
        
        stats = {
            "total_books": len(self.books_data),
            "unique_genres": len(genres),
            "genres": genres,
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "embedding_size": self.embeddings_matrix.shape[1] if self.embeddings_matrix is not None else 0,
            "status": "ready",
            "monitoring_enabled": self.monitoring_enabled
        }
        
        return stats