import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class BookDataProcessor:
    def __init__(self):
        """Ініціалізація процесора даних з Sentence Transformers моделлю"""
        self.model_name = os.getenv('EMBEDDINGS_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
        print(f"Завантажуємо модель: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("✅ Модель завантажена!")
    
    def load_books_data(self, csv_path: str = "data/bookye_books_10000_with_id.csv") -> pd.DataFrame:
        """Завантаження даних з CSV файлу"""
        print(f"Завантажуємо дані з {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✅ Завантажено {len(df)} книг")
        return df
    
    def create_book_text(self, row: pd.Series) -> str:
        """Створення текстового представлення книги для embeddings"""
        # Об'єднуємо всю інформацію про книгу в один текст
        text_parts = []
        
        if pd.notna(row['title']):
            text_parts.append(f"Назва: {row['title']}")
        
        if pd.notna(row['genre']):
            text_parts.append(f"Жанр: {row['genre']}")
            
        if pd.notna(row['description']):
            text_parts.append(f"Опис: {row['description']}")
            
        if pd.notna(row['rating']):
            text_parts.append(f"Рейтинг: {row['rating']}")
        
        return " | ".join(text_parts)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Генерація embeddings для списку текстів"""
        print(f"Генеруємо embeddings для {len(texts)} текстів...")
        
        # Генеруємо embeddings батчами для економії пам'яті
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)
            print(f"Оброблено {min(i + batch_size, len(texts))}/{len(texts)} текстів")
        
        return np.vstack(embeddings)
    
    def process_books(self, csv_path: str = None) -> List[Dict]:
        """Основний метод обробки книг: завантаження + створення embeddings"""
        # Завантажуємо дані
        if csv_path is None:
            csv_path = "data/bookye_books_10000_with_id.csv"
        
        df = self.load_books_data(csv_path)
        
        # Створюємо текстові представлення
        print("Створюємо текстові представлення книг...")
        book_texts = [self.create_book_text(row) for _, row in df.iterrows()]
        
        # Генеруємо embeddings
        embeddings = self.generate_embeddings(book_texts)
        
        # Створюємо фінальну структуру даних
        processed_books = []
        for i, (_, row) in enumerate(df.iterrows()):
            book_data = {
                'id': str(row['id']),
                'title': row['title'],
                'description': row['description'] if pd.notna(row['description']) else "",
                'genre': row['genre'],
                'rating': float(row['rating']) if pd.notna(row['rating']) else 0.0,
                'text': book_texts[i],
                'embedding': embeddings[i].tolist()  # Конвертуємо в список для JSON
            }
            processed_books.append(book_data)
        
        print(f"✅ Оброблено {len(processed_books)} книг")
        return processed_books
    
    def save_processed_data(self, processed_books: List[Dict], output_path: str = "data/processed_books.json"):
        """Збереження оброблених даних в JSON файл"""
        print(f"Зберігаємо дані в {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_books, f, ensure_ascii=False, indent=2)
        print(f"✅ Збережено {len(processed_books)} книг")

def main():
    """Основна функція для тестування"""
    processor = BookDataProcessor()
    
    # Обробляємо перші 100 книг для тестування
    df = processor.load_books_data()
    test_df = df.head(100)  # Беремо перші 100 для швидкого тестування
    
    # Зберігаємо тестові дані
    test_csv_path = "data/test_books_100.csv"
    test_df.to_csv(test_csv_path, index=False)
    
    # Обробляємо тестові дані
    processed_books = processor.process_books(test_csv_path)
    processor.save_processed_data(processed_books, "data/test_processed_books.json")
    
    print("🎉 Тестова обробка завершена!")
    print(f"Розмір embedding: {len(processed_books[0]['embedding'])}")
    print(f"Приклад тексту: {processed_books[0]['text'][:200]}...")

if __name__ == "__main__":
    main()