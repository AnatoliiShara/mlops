import gradio as gr
import asyncio
import json
import sys
import os

# Додаємо поточну директорію до PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_engine import RAGEngine

# Імпорти для додаткових функцій (опціонально)
try:
    from data_processor import BookDataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    DATA_PROCESSOR_AVAILABLE = False
    print("⚠️ data_processor.py не знайдено")

try:
    from vector_store import FirestoreVectorStore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    print("⚠️ vector_store.py не знайдено")

# Глобальна змінна для RAG системи
rag_engine = None

def initialize_rag():
    """Ініціалізація RAG системи"""
    global rag_engine
    try:
        rag_engine = RAGEngine()
        return "✅ RAG система готова до роботи!"
    except Exception as e:
        return f"❌ Помилка ініціалізації: {str(e)}"

def search_books(query: str, num_results: int = 5):
    """Функція пошуку книг для Gradio"""
    if not rag_engine:
        return "❌ Система не ініціалізована. Натисніть 'Ініціалізувати систему'", ""
    
    if not query.strip():
        return "❌ Введіть запит для пошуку", ""
    
    try:
        # Запускаємо асинхронну функцію
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            rag_engine.search_and_respond(query, top_k=num_results)
        )
        loop.close()
        
        # Форматуємо результати для відображення
        if result['status'] == 'success':
            # Основна відповідь
            main_response = result['response']
            
            # Деталі знайдених книг
            books_details = "📚 ЗНАЙДЕНІ КНИГИ:\n\n"
            for i, book in enumerate(result['found_books'], 1):
                books_details += f"""
{i}. 📖 "{book['title']}"
   🎭 Жанр: {book['genre']}
   ⭐ Рейтинг: {book['rating']}/5
   📄 Опис: {book['description']}
   🎯 Схожість: {book['similarity_score']:.0%}
   
"""
            
            return main_response, books_details
        else:
            return result['response'], "Книги не знайдено."
            
    except Exception as e:
        return f"❌ Помилка при пошуку: {str(e)}", ""

def get_system_stats():
    """Отримання статистики системи"""
    if not rag_engine:
        return "Система не ініціалізована"
    
    try:
        stats = rag_engine.get_stats()
        
        stats_text = f"""
📊 СТАТИСТИКА СИСТЕМИ:

📚 Загальна кількість книг: {stats['total_books']}
🎭 Унікальних жанрів: {stats['unique_genres']}
⭐ Середній рейтинг: {stats['avg_rating']:.2f}/5
🧠 Розмір embeddings: {stats['embedding_size']}
✅ Статус: {stats['status']}

📈 РОЗПОДІЛ ЗА ЖАНРАМИ:
"""
        
        for genre, count in stats['genres'].items():
            percentage = (count / stats['total_books']) * 100
            stats_text += f"• {genre}: {count} ({percentage:.1f}%)\n"
        
        return stats_text
        
    except Exception as e:
        return f"❌ Помилка отримання статистики: {str(e)}"

def process_full_dataset():
    """Обробка повного датасету (10000 книг)"""
    if not DATA_PROCESSOR_AVAILABLE:
        return "❌ data_processor.py не доступний"
    
    try:
        processor = BookDataProcessor()
        
        # Обробляємо всі 10000 книг
        processed_books = processor.process_books("data/bookye_books_10000_with_id.csv")
        processor.save_processed_data(processed_books, "data/all_processed_books.json")
        
        # Перезавантажуємо RAG з новими даними
        global rag_engine
        if rag_engine:
            rag_engine.load_processed_books("data/all_processed_books.json")
        
        return f"✅ Успішно оброблено та завантажено {len(processed_books)} книг!"
        
    except Exception as e:
        return f"❌ Помилка обробки: {str(e)}"

def upload_to_firestore():
    """Завантаження даних у Firestore"""
    if not FIRESTORE_AVAILABLE:
        return "❌ vector_store.py не доступний"
    
    try:
        # Завантажуємо оброблені дані
        with open("data/test_processed_books.json", 'r', encoding='utf-8') as f:
            books_data = json.load(f)
        
        # Ініціалізуємо Firestore
        store = FirestoreVectorStore()
        
        # Завантажуємо книги
        added_ids = store.add_books_batch(books_data, batch_size=10)
        
        return f"✅ Завантажено {len(added_ids)} книг у Firestore!"
        
    except Exception as e:
        return f"❌ Помилка завантаження в Firestore: {str(e)}"

# Створюємо Gradio інтерфейс
def create_interface():
    """Створення Gradio інтерфейсу"""
    
    with gr.Blocks(
        title="📚 Розумний Асистент Книжкового Магазину",
        theme=gr.themes.Soft(),
        css="""
        .main-header {
            text-align: center;
            color: #2D5AA0;
            margin-bottom: 20px;
        }
        .search-box {
            font-size: 16px;
        }
        .result-box {
            font-family: 'Georgia', serif;
            line-height: 1.6;
        }
        """
    ) as iface:
        
        # Заголовок
        gr.HTML("""
        <div class="main-header">
            <h1>📚 Розумний Асистент Книжкового Магазину</h1>
            <p>Знайдіть ідеальну книгу за допомогою штучного інтелекту!</p>
        </div>
        """)
        
        # Основна секція пошуку
        with gr.Tab("🔍 Пошук Книг"):
            with gr.Row():
                with gr.Column(scale=2):
                    init_btn = gr.Button("🚀 Ініціалізувати систему", variant="primary")
                    init_output = gr.Textbox(label="Статус ініціалізації", interactive=False)
                    
                    search_input = gr.Textbox(
                        label="🔍 Що вас цікавить?",
                        placeholder="Наприклад: 'пригодницькі книги для дітей', 'романтичні романи', 'наукова фантастика'...",
                        elem_classes=["search-box"]
                    )
                    
                    with gr.Row():
                        num_results = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Кількість результатів"
                        )
                        search_btn = gr.Button("🔍 Знайти книги", variant="secondary")
                
                with gr.Column(scale=1):
                    stats_btn = gr.Button("📊 Статистика системи")
                    stats_output = gr.Textbox(
                        label="📊 Інформація про систему",
                        lines=15,
                        interactive=False
                    )
            
            # Результати пошуку
            with gr.Row():
                with gr.Column():
                    response_output = gr.Textbox(
                        label="💬 Рекомендації асистента",
                        lines=8,
                        elem_classes=["result-box"],
                        interactive=False
                    )
                
                with gr.Column():
                    books_output = gr.Textbox(
                        label="📚 Деталі знайдених книг",
                        lines=8,
                        elem_classes=["result-box"],
                        interactive=False
                    )
        
        # Секція адміністрування
        with gr.Tab("⚙️ Адміністрування"):
            gr.HTML("<h3>🔧 Управління даними</h3>")
            
            with gr.Row():
                process_btn = gr.Button(
                    "📚 Обробити всі 10000 книг", 
                    variant="primary",
                    visible=DATA_PROCESSOR_AVAILABLE
                )
                upload_btn = gr.Button(
                    "☁️ Завантажити в Firestore", 
                    variant="secondary",
                    visible=FIRESTORE_AVAILABLE
                )
            
            admin_output = gr.Textbox(
                label="📋 Результат операції",
                lines=5,
                interactive=False
            )
            
            # Інформація про доступність модулів
            if not DATA_PROCESSOR_AVAILABLE:
                gr.HTML("<p>⚠️ data_processor.py не доступний</p>")
            if not FIRESTORE_AVAILABLE:
                gr.HTML("<p>⚠️ vector_store.py не доступний</p>")
        
        # Секція довідки
        with gr.Tab("ℹ️ Довідка"):
            gr.HTML("""
            <div style="padding: 20px; font-family: Georgia, serif; line-height: 1.8;">
                <h3>🎯 Як користуватися системою:</h3>
                
                <h4>1. 🚀 Ініціалізація</h4>
                <p>Натисніть <strong>"Ініціалізувати систему"</strong> для завантаження RAG движка.</p>
                
                <h4>2. 🔍 Пошук книг</h4>
                <p>Введіть запит українською мовою. Наприклад:</p>
                <ul>
                    <li>"Пригодницькі книги для підлітків"</li>
                    <li>"Книги про кохання з високим рейтингом"</li>
                    <li>"Наукова фантастика новинки"</li>
                    <li>"Класична література"</li>
                    <li>"Українська історія"</li>
                </ul>
                
                <h4>3. 📊 Технології</h4>
                <ul>
                    <li><strong>Sentence Transformers</strong> - для створення векторних представлень</li>
                    <li><strong>Cosine Similarity</strong> - для пошуку схожих книг</li>
                    <li><strong>Google Gemini API</strong> - для генерації розумних відповідей</li>
                    <li><strong>Gradio</strong> - для веб-інтерфейсу</li>
                </ul>
                
                <h4>4. 🎭 Доступні жанри</h4>
                <p>Новинки, Дитяча література, Нехудожня література, Художня література, Серія книг, Альбоми, Художня література, Електронні книги.</p>
                
                <h4>5. 🔧 Адміністрування</h4>
                <p>У розділі адміністрування можна:</p>
                <ul>
                    <li>Обробити повний датасет з 10,000 книг</li>
                    <li>Завантажити дані в Google Firestore</li>
                    <li>Переглянути детальну статистику системи</li>
                </ul>
            </div>
            """)
        
        # Підключаємо функції до кнопок
        init_btn.click(initialize_rag, outputs=init_output)
        search_btn.click(
            search_books,
            inputs=[search_input, num_results],
            outputs=[response_output, books_output]
        )
        stats_btn.click(get_system_stats, outputs=stats_output)
        
        # Адміністративні функції (тільки якщо доступні)
        if DATA_PROCESSOR_AVAILABLE:
            process_btn.click(process_full_dataset, outputs=admin_output)
        if FIRESTORE_AVAILABLE:
            upload_btn.click(upload_to_firestore, outputs=admin_output)
        
        # Автоматична ініціалізація при завантаженні
        iface.load(initialize_rag, outputs=init_output)
    
    return iface

# Запуск програми
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=None,  # Автоматично знайде вільний порт
        share=True,
        debug=True
    )