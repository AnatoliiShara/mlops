# vector_store.py
import os
import numpy as np
from typing import List, Dict, Tuple
from google.cloud import firestore
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import json

load_dotenv()

class FirestoreVectorStore:
    def __init__(self):
        """Ініціалізація Firestore клієнта"""
        self.project_id = os.getenv('GCP_PROJECT_ID')
        
        # Ініціалізуємо Firestore клієнт
        self.db = firestore.Client(project=self.project_id)
        self.collection_name = 'books'
        print(f"✅ Підключено до Firestore проекту: {self.project_id}")
    
    def add_book(self, book_data: Dict) -> str:
        """Додавання однієї книги до Firestore"""
        try:
            # Firestore не підтримує масиви float напряму, тому зберігаємо embedding як строку
            book_data_for_firestore = book_data.copy()
            book_data_for_firestore['embedding'] = json.dumps(book_data['embedding'])
            
            doc_ref = self.db.collection(self.collection_name).document(book_data['id'])
            doc_ref.set(book_data_for_firestore)
            return book_data['id']
        except Exception as e:
            print(f"❌ Помилка при додаванні книги {book_data['id']}: {e}")
            return None
    
    def add_books_batch(self, books: List[Dict], batch_size: int = 50) -> List[str]:
        """Додавання книг батчами для швидкості"""
        added_ids = []
        
        for i in range(0, len(books), batch_size):
            batch = books[i:i + batch_size]
            batch_obj = self.db.batch()
            
            try:
                for book in batch:
                    # Підготовка даних для Firestore
                    book_data_for_firestore = book.copy()
                    book_data_for_firestore['embedding'] = json.dumps(book['embedding'])
                    
                    doc_ref = self.db.collection(self.collection_name).document(book['id'])
                    batch_obj.set(doc_ref, book_data_for_firestore)
                
                # Виконуємо батч
                batch_obj.commit()
                batch_ids = [book['id'] for book in batch]
                added_ids.extend(batch_ids)
                
                print(f"✅ Додано батч {i//batch_size + 1}: {len(batch)} книг")
                
            except Exception as e:
                print(f"❌ Помилка в батчі {i//batch_size + 1}: {e}")
        
        print(f"🎉 Всього додано {len(added_ids)} книг до Firestore")
        return added_ids
    
    def get_book_by_id(self, book_id: str) -> Dict:
        """Отримання книги за ID"""
        try:
            doc_ref = self.db.collection(self.collection_name).document(book_id)
            doc = doc_ref.get()
            
            if doc.exists:
                book_data = doc.to_dict()
                # Конвертуємо embedding назад з JSON
                book_data['embedding'] = json.loads(book_data['embedding'])
                return book_data
            else:
                return None
        except Exception as e:
            print(f"❌ Помилка при отриманні книги {book_id}: {e}")
            return None
    
    def get_all_books(self) -> List[Dict]:
        """Отримання всіх книг з Firestore"""
        try:
            docs = self.db.collection(self.collection_name).stream()
            books = []
            
            for doc in docs:
                book_data = doc.to_dict()
                # Конвертуємо embedding назад з JSON
                book_data['embedding'] = json.loads(book_data['embedding'])
                books.append(book_data)
            
            print(f"✅ Завантажено {len(books)} книг з Firestore")
            return books
        except Exception as e:
            print(f"❌ Помилка при завантаженні книг: {e}")
            return []
    
    def search_similar_books(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Пошук схожих книг за embedding використовуючи cosine similarity"""
        try:
            # Завантажуємо всі книги (для малих датасетів це OK)
            all_books = self.get_all_books()
            
            if not all_books:
                return []
            
            # Створюємо матрицю embeddings
            embeddings_matrix = np.array([book['embedding'] for book in all_books])
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            # Обчислюємо cosine similarity
            similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
            
            # Сортуємо за схожістю та беремо топ-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                book = all_books[idx].copy()
                similarity_score = float(similarities[idx])
                # Видаляємо embedding з результату для економії пам'яті
                book.pop('embedding', None)
                results.append((book, similarity_score))
            
            return results
            
        except Exception as e:
            print(f"❌ Помилка при пошуку схожих книг: {e}")
            return []
    
    def search_books_by_genre(self, genre: str, limit: int = 10) -> List[Dict]:
        """Пошук книг за жанром"""
        try:
            docs = self.db.collection(self.collection_name)\
                     .where('genre', '==', genre)\
                     .limit(limit)\
                     .stream()
            
            books = []
            for doc in docs:
                book_data = doc.to_dict()
                # Видаляємо embedding для економії
                book_data.pop('embedding', None)
                books.append(book_data)
            
            return books
        except Exception as e:
            print(f"❌ Помилка при пошуку за жанром: {e}")
            return []
    
    def search_books_by_rating(self, min_rating: float, limit: int = 10) -> List[Dict]:
        """Пошук книг за мінімальним рейтингом"""
        try:
            docs = self.db.collection(self.collection_name)\
                     .where('rating', '>=', min_rating)\
                     .order_by('rating', direction=firestore.Query.DESCENDING)\
                     .limit(limit)\
                     .stream()
            
            books = []
            for doc in docs:
                book_data = doc.to_dict()
                book_data.pop('embedding', None)
                books.append(book_data)
            
            return books
        except Exception as e:
            print(f"❌ Помилка при пошуку за рейтингом: {e}")
            return []
    
    def delete_all_books(self):
        """Видалення всіх книг (для тестування)"""
        try:
            docs = self.db.collection(self.collection_name).stream()
            for doc in docs:
                doc.reference.delete()
            print("✅ Всі книги видалено")
        except Exception as e:
            print(f"❌ Помилка при видаленні: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Статистика колекції"""
        try:
            docs = list(self.db.collection(self.collection_name).stream())
            total_books = len(docs)
            
            if total_books == 0:
                return {"total_books": 0}
            
            # Збираємо статистику
            genres = {}
            ratings = []
            
            for doc in docs:
                data = doc.to_dict()
                genre = data.get('genre', 'Невідомо')
                genres[genre] = genres.get(genre, 0) + 1
                
                rating = data.get('rating', 0)
                if rating:
                    ratings.append(rating)
            
            stats = {
                "total_books": total_books,
                "genres": genres,
                "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
                "min_rating": min(ratings) if ratings else 0,
                "max_rating": max(ratings) if ratings else 0
            }
            
            return stats
        except Exception as e:
            print(f"❌ Помилка при отриманні статистики: {e}")
            return {}

def main():
    """Функція для тестування"""
    store = FirestoreVectorStore()
    
    # Тестуємо підключення
    stats = store.get_collection_stats()
    print("📊 Статистика колекції:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()