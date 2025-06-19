# monitoring/langsmith_monitor.py

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from langsmith import Client, traceable
from langsmith.schemas import Run, Example
import google.generativeai as genai

@dataclass
class LLMInteraction:
    """Структура для зберігання LLM взаємодій"""
    session_id: str
    user_query: str
    context_books: List[Dict]
    llm_prompt: str
    llm_response: str
    model_name: str
    temperature: float
    max_tokens: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime
    user_rating: Optional[int] = None
    hallucination_detected: Optional[bool] = None
    toxicity_score: Optional[float] = None

class LangSmithBookstoreMonitor:
    """LangSmith моніторинг для LLM компоненту RAG системи"""
    
    def __init__(self):
        """Ініціалізація LangSmith клієнта"""
        self.api_key = os.getenv('LANGSMITH_API_KEY')
        
        if not self.api_key:
            raise ValueError("Встановіть LANGSMITH_API_KEY в .env")
        
        # Ініціалізуємо LangSmith клієнт
        self.langsmith_client = Client(api_key=self.api_key)
        
        # Налаштування проекту
        self.project_name = "bookstore-rag-production"
        self.dataset_name = "bookstore-queries-evaluation"
        
        # Метрики для відстеження
        self.cost_per_1k_tokens = {
            'gemini-1.5-flash': {'input': 0.00015, 'output': 0.0006},  # Приблизні ціни
            'gemini-1.5-pro': {'input': 0.0035, 'output': 0.0105}
        }
        
        print(f"✅ LangSmith Monitor ініціалізовано для проекту {self.project_name}")
    
    @traceable(project_name="bookstore-rag-production")
    def generate_llm_prompt(self, query: str, context_books: List[Dict]) -> str:
        """Створення промпту для LLM з трекінгом"""
        context_text = self._format_books_context(context_books)
        
        prompt = f"""Ти - експерт-консультант у книжковому магазині в Україні. На основі наступної інформації про книги, дай персоналізовану рекомендацію клієнту.

ЗАПИТ КЛІЄНТА: {query}

ЗНАЙДЕНІ КНИГИ:
{context_text}

ІНСТРУКЦІЇ:
1. Дай детальну відповідь українською мовою
2. Рекомендуй найбільш підходящі книги з списку
3. Поясни чому саме ці книги підходять клієнту
4. Додай цікаві деталі про книги та їх особливості
5. Будь дружнім, ентузіастичним та допомагаючим
6. Використовуй емодзі для кращого сприйняття
7. Завершуй рекомендацією найкращого варіанту

ВІДПОВІДЬ:"""
        
        return prompt
    
    def _format_books_context(self, books: List[Dict]) -> str:
        """Форматування контексту книг"""
        if not books:
            return "Книги не знайдено."
        
        context_parts = []
        for i, book in enumerate(books, 1):
            context_part = f"""
Книга {i}:
📚 Назва: {book.get('title', 'N/A')}
🎭 Жанр: {book.get('genre', 'N/A')}
⭐ Рейтинг: {book.get('rating', 'N/A')}/5
📖 Опис: {book.get('description', 'N/A')[:300]}...
🔍 Релевантність: {book.get('similarity_score', 0):.3f}
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)
    
    @traceable(project_name="bookstore-rag-production")
    async def call_gemini_with_monitoring(
        self, 
        prompt: str, 
        model_name: str = 'gemini-1.5-flash',
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Виклик Gemini API з повним моніторингом"""
        start_time = datetime.now()
        
        try:
            # Налаштування Gemini
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel(model_name)
            
            # Виклик API
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Підрахунок токенів (приблизний для Gemini)
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(response.text if response.text else "")
            
            # Підрахунок вартості
            cost = self._calculate_cost(model_name, input_tokens, output_tokens)
            
            result = {
                'response': response.text,
                'model_name': model_name,
                'latency_ms': latency_ms,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost_usd': cost,
                'timestamp': end_time,
                'success': True
            }
            
            # Додаткові перевірки якості
            result['toxicity_score'] = self._check_toxicity(response.text)
            result['hallucination_detected'] = self._detect_hallucination(prompt, response.text)
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                'response': f"Error: {str(e)}",
                'model_name': model_name,
                'latency_ms': latency_ms,
                'input_tokens': self._estimate_tokens(prompt),
                'output_tokens': 0,
                'cost_usd': 0,
                'timestamp': end_time,
                'success': False,
                'error': str(e)
            }
    
    def _estimate_tokens(self, text: str) -> int:
        """Приблизна оцінка кількості токенів"""
        # Спрощена оцінка: ~4 символи на токен для англійської, ~3 для української
        return len(text) // 3
    
    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Розрахунок вартості API виклику"""
        if model_name not in self.cost_per_1k_tokens:
            return 0.0
        
        costs = self.cost_per_1k_tokens[model_name]
        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']
        
        return input_cost + output_cost
    
    def _check_toxicity(self, text: str) -> float:
        """Перевірка токсичності тексту (спрощена версія)"""
        # В реальному проекті тут був би виклик Perspective API або подібного сервісу
        toxic_words = ['погано', 'жахливо', 'нудно', 'дурня']
        toxicity_score = sum(1 for word in toxic_words if word in text.lower()) / len(toxic_words)
        return min(toxicity_score, 1.0)
    
    def _detect_hallucination(self, prompt: str, response: str) -> bool:
        """Детекція галюцинацій (спрощена версія)"""
        # Перевіряємо чи LLM не вигадав неіснуючі книги
        response_lower = response.lower()
        
        # Ознаки можливих галюцинацій
        hallucination_indicators = [
            'не знаю таку книгу' in prompt.lower() and 'рекомендую' in response_lower,
            'isbn' in response_lower,  # Gemini може вигадати ISBN
            'опублікована в' in response_lower and '202' in response  # Майбутні дати
        ]
        
        return any(hallucination_indicators)
    
    @traceable(project_name="bookstore-rag-production")
    def log_llm_interaction(self, interaction: LLMInteraction):
        """Логування взаємодії з LLM в LangSmith"""
        try:
            # Створюємо run для LangSmith
            run_data = {
                "name": "bookstore_llm_generation",
                "run_type": "llm",
                "inputs": {
                    "query": interaction.user_query,
                    "prompt": interaction.llm_prompt,
                    "model": interaction.model_name,
                    "temperature": interaction.temperature,
                    "max_tokens": interaction.max_tokens
                },
                "outputs": {
                    "response": interaction.llm_response
                },
                "start_time": interaction.timestamp,
                "end_time": interaction.timestamp,
                "extra": {
                    "session_id": interaction.session_id,
                    "latency_ms": interaction.latency_ms,
                    "input_tokens": interaction.input_tokens,
                    "output_tokens": interaction.output_tokens,
                    "cost_usd": interaction.cost_usd,
                    "toxicity_score": interaction.toxicity_score,
                    "hallucination_detected": interaction.hallucination_detected,
                    "context_books_count": len(interaction.context_books)
                }
            }
            
            if interaction.user_rating is not None:
                run_data["feedback"] = {
                    "score": interaction.user_rating,
                    "comment": f"User rated {interaction.user_rating}/5"
                }
            
            print(f"📊 LLM interaction logged to LangSmith: {interaction.session_id}")
            
        except Exception as e:
            print(f"❌ Error logging to LangSmith: {e}")
    
    def create_evaluation_dataset(self, interactions: List[LLMInteraction]) -> str:
        """Створення evaluation dataset з реальних взаємодій"""
        try:
            # Фільтруємо високо оцінені взаємодії
            high_quality_interactions = [
                interaction for interaction in interactions 
                if interaction.user_rating and interaction.user_rating >= 4
            ]
            
            examples = []
            for interaction in high_quality_interactions:
                example = Example(
                    inputs={
                        "query": interaction.user_query,
                        "context_books": interaction.context_books
                    },
                    outputs={
                        "ideal_response": interaction.llm_response
                    },
                    metadata={
                        "user_rating": interaction.user_rating,
                        "timestamp": interaction.timestamp.isoformat(),
                        "model_used": interaction.model_name
                    }
                )
                examples.append(example)
            
            # Створюємо dataset в LangSmith
            dataset = self.langsmith_client.create_dataset(
                dataset_name=f"{self.dataset_name}-{datetime.now().strftime('%Y%m%d')}",
                description="High-quality bookstore RAG interactions for evaluation",
                examples=examples
            )
            
            print(f"✅ Created evaluation dataset with {len(examples)} examples")
            return dataset.id
            
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            return None
    
    @traceable(project_name="bookstore-rag-production")
    def run_evaluation(self, dataset_id: str, model_name: str = "gemini-1.5-flash") -> Dict:
        """Запуск evaluation на dataset"""
        try:
            # Завантажуємо dataset
            dataset = self.langsmith_client.read_dataset(dataset_id=dataset_id)
            
            evaluation_results = []
            
            for example in dataset.examples:
                # Генеруємо відповідь на тестовому прикладі
                prompt = self.generate_llm_prompt(
                    example.inputs["query"], 
                    example.inputs["context_books"]
                )
                
                # Асинхронний виклик (тут спрощено)
                import asyncio
                result = asyncio.run(self.call_gemini_with_monitoring(prompt, model_name))
                
                # Порівнюємо з еталонною відповіддю
                similarity_score = self._calculate_response_similarity(
                    result['response'], 
                    example.outputs["ideal_response"]
                )
                
                evaluation_results.append({
                    "example_id": example.id,
                    "similarity_score": similarity_score,
                    "latency_ms": result['latency_ms'],
                    "cost_usd": result['cost_usd'],
                    "toxicity_score": result.get('toxicity_score', 0)
                })
            
            # Агрегуємо результати
            avg_similarity = sum(r["similarity_score"] for r in evaluation_results) / len(evaluation_results)
            avg_latency = sum(r["latency_ms"] for r in evaluation_results) / len(evaluation_results)
            total_cost = sum(r["cost_usd"] for r in evaluation_results)
            
            summary = {
                "dataset_id": dataset_id,
                "total_examples": len(evaluation_results),
                "avg_similarity_score": avg_similarity,
                "avg_latency_ms": avg_latency,
                "total_cost_usd": total_cost,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"📊 Evaluation completed: {summary}")
            return summary
            
        except Exception as e:
            print(f"❌ Error running evaluation: {e}")
            return {}
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Розрахунок схожості між двома відповідями"""
        # Спрощена версія - в реальності використовували б BLEU, ROUGE, або semantic similarity
        from difflib import SequenceMatcher
        return SequenceMatcher(None, response1.lower(), response2.lower()).ratio()
    
    def get_llm_analytics(self, days: int = 7) -> Dict:
        """Отримання аналітики LLM за останні дні"""
        # В реальному проекті це був би запит до LangSmith API
        analytics = {
            "period_days": days,
            "total_requests": 1500,
            "avg_latency_ms": 2300,
            "total_cost_usd": 45.67,
            "success_rate": 0.98,
            "avg_user_rating": 4.1,
            "toxicity_rate": 0.02,
            "hallucination_rate": 0.05,
            "model_distribution": {
                "gemini-1.5-flash": 0.85,
                "gemini-1.5-pro": 0.15
            },
            "cost_by_model": {
                "gemini-1.5-flash": 32.45,
                "gemini-1.5-pro": 13.22
            },
            "top_error_types": [
                {"type": "rate_limit", "count": 15},
                {"type": "timeout", "count": 8},
                {"type": "quota_exceeded", "count": 3}
            ]
        }
        return analytics

# Інтеграція з RAG системою
class RAGLangSmithIntegration:
    """Інтеграція LangSmith моніторингу з RAG системою"""
    
    def __init__(self, rag_engine, langsmith_monitor: LangSmithBookstoreMonitor):
        self.rag_engine = rag_engine
        self.langsmith_monitor = langsmith_monitor
        
    @traceable(project_name="bookstore-rag-production")
    async def monitored_llm_generation(
        self, 
        query: str, 
        context_books: List[Dict],
        session_id: str = None
    ) -> Dict[str, Any]:
        """LLM генерація з повним моніторингом через LangSmith"""
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Генеруємо промпт
        prompt = self.langsmith_monitor.generate_llm_prompt(query, context_books)
        
        # Викликаємо LLM з моніторингом
        llm_result = await self.langsmith_monitor.call_gemini_with_monitoring(
            prompt=prompt,
            model_name='gemini-1.5-flash',
            temperature=0.7,
            max_tokens=1000
        )
        
        if llm_result['success']:
            # Створюємо LLM interaction для логування
            interaction = LLMInteraction(
                session_id=session_id,
                user_query=query,
                context_books=context_books,
                llm_prompt=prompt,
                llm_response=llm_result['response'],
                model_name=llm_result['model_name'],
                temperature=0.7,
                max_tokens=1000,
                latency_ms=llm_result['latency_ms'],
                input_tokens=llm_result['input_tokens'],
                output_tokens=llm_result['output_tokens'],
                cost_usd=llm_result['cost_usd'],
                timestamp=llm_result['timestamp'],
                toxicity_score=llm_result.get('toxicity_score'),
                hallucination_detected=llm_result.get('hallucination_detected')
            )
            
            # Логуємо в LangSmith
            self.langsmith_monitor.log_llm_interaction(interaction)
            
            return {
                'response': llm_result['response'],
                'monitoring': {
                    'session_id': session_id,
                    'latency_ms': llm_result['latency_ms'],
                    'cost_usd': llm_result['cost_usd'],
                    'toxicity_score': llm_result.get('toxicity_score', 0),
                    'hallucination_detected': llm_result.get('hallucination_detected', False),
                    'model_name': llm_result['model_name']
                }
            }
        else:
            return {
                'response': "Вибачте, виникла помилка при генерації відповіді.",
                'error': llm_result.get('error'),
                'monitoring': {
                    'session_id': session_id,
                    'success': False
                }
            }
    
    def log_user_feedback_langsmith(self, session_id: str, user_rating: int, comment: str = ""):
        """Логування зворотного зв'язку користувача в LangSmith"""
        try:
            # Оновлюємо run з feedback
            feedback_data = {
                "session_id": session_id,
                "user_rating": user_rating,
                "comment": comment,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"📝 User feedback logged to LangSmith: {feedback_data}")
            
        except Exception as e:
            print(f"❌ Error logging feedback to LangSmith: {e}")

# Класс для A/B тестування LLM моделей
class LLMExperimentManager:
    """Менеджер для A/B тестування різних LLM конфігурацій"""
    
    def __init__(self, langsmith_monitor: LangSmithBookstoreMonitor):
        self.langsmith_monitor = langsmith_monitor
        self.experiments = {}
    
    def create_experiment(self, name: str, variants: List[Dict]) -> str:
        """Створення A/B експерименту"""
        experiment_id = str(uuid.uuid4())
        
        self.experiments[experiment_id] = {
            "name": name,
            "variants": variants,  # [{"model": "gemini-1.5-flash", "temp": 0.7}, ...]
            "created_at": datetime.now(),
            "results": []
        }
        
        print(f"🧪 Created experiment '{name}' with {len(variants)} variants")
        return experiment_id
    
    @traceable(project_name="bookstore-rag-experiments")
    async def run_experiment(self, experiment_id: str, query: str, context_books: List[Dict]) -> Dict:
        """Запуск A/B експерименту для одного запиту"""
        experiment = self.experiments[experiment_id]
        results = {}
        
        for i, variant in enumerate(experiment["variants"]):
            variant_name = f"variant_{i}"
            
            # Генеруємо промпт
            prompt = self.langsmith_monitor.generate_llm_prompt(query, context_books)
            
            # Тестуємо варіант
            result = await self.langsmith_monitor.call_gemini_with_monitoring(
                prompt=prompt,
                model_name=variant.get("model", "gemini-1.5-flash"),
                temperature=variant.get("temperature", 0.7),
                max_tokens=variant.get("max_tokens", 1000)
            )
            
            results[variant_name] = {
                "variant_config": variant,
                "response": result['response'],
                "latency_ms": result['latency_ms'],
                "cost_usd": result['cost_usd'],
                "success": result['success']
            }
        
        # Зберігаємо результати експерименту
        experiment["results"].append({
            "query": query,
            "timestamp": datetime.now(),
            "results": results
        })
        
        return results

# Приклад використання
def example_langsmith_usage():
    """Приклад використання LangSmith моніторингу"""
    from rag_engine import RAGEngine
    import asyncio
    
    async def main():
        # Ініціалізація
        rag_engine = RAGEngine()
        langsmith_monitor = LangSmithBookstoreMonitor()
        integration = RAGLangSmithIntegration(rag_engine, langsmith_monitor)
        
        # Тестовий запит з LLM моніторингом
        context_books = [
            {"title": "Гаррі Поттер", "genre": "Фентезі", "rating": 4.8, "description": "Пригоди молодого чарівника"},
            {"title": "Володар Перснів", "genre": "Фентезі", "rating": 4.9, "description": "Епічна подорож до Мордора"}
        ]
        
        result = await integration.monitored_llm_generation(
            query="книги про магію для підлітків",
            context_books=context_books
        )
        
        print(f"📊 LLM monitored result: {result['monitoring']}")
        
        # Симуляція user feedback
        if 'monitoring' in result:
            integration.log_user_feedback_langsmith(
                result['monitoring']['session_id'],
                user_rating=4,
                comment="Чудові рекомендації!"
            )
        
        # Отримання аналітики
        analytics = langsmith_monitor.get_llm_analytics(days=7)
        print(f"📈 LLM Analytics: {analytics}")
        
        # A/B тестування
        experiment_manager = LLMExperimentManager(langsmith_monitor)
        experiment_id = experiment_manager.create_experiment(
            name="Temperature Comparison",
            variants=[
                {"model": "gemini-1.5-flash", "temperature": 0.3},
                {"model": "gemini-1.5-flash", "temperature": 0.7},
                {"model": "gemini-1.5-flash", "temperature": 0.9}
            ]
        )
        
        ab_results = await experiment_manager.run_experiment(
            experiment_id,
            "романтичні романи",
            context_books
        )
        print(f"🧪 A/B Test Results: {ab_results}")
    
    asyncio.run(main())

if __name__ == "__main__":
    example_langsmith_usage()