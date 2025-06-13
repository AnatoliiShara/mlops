import pandas as pd
import sys
import ray
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
import torch
import wandb
import time

# Ініціалізація WandB
wandb.init(project="book-recommendation-system", name="embedding-recommendation-run")

# Логування параметрів
wandb.config.update({
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    "chunk_size": 100,
})

# Ініціалізація Accelerate та Ray
accelerator = Accelerator()
device = accelerator.device
ray.init(ignore_reinit_error=True)

# Завантажуємо метадані книг
books_df = pd.read_csv("data/ukr_books_dataset.csv")
required_cols = {"description", "genre"}
if not required_cols.issubset(books_df.columns):
    raise KeyError(f"У файлі метаданих немає колонок {required_cols}")

# Додаємо book_id, якщо відсутній
if "book_id" not in books_df.columns:
    books_df["book_id"] = books_df.index + 1

# Обираємо модель для ембеддінгів українською (multilingual)
model_name = "paraphrase-multilingual-MiniLM-L12-v2"

# Використовуємо Ray Actor для уникнення великих замикань
@ray.remote
class EncoderActor:
    def __init__(self, model_name, device):
        # Окремий імпорт для кращої сериалізації
        from sentence_transformers import SentenceTransformer
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts):
        import torch
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu()

# Створюємо актор для кодування
encoder = EncoderActor.remote(model_name, device)

# Розбиваємо описи на чанки
chunk_size = 100
descriptions = books_df["description"].tolist()
chunks = [descriptions[i:i + chunk_size] for i in range(0, len(descriptions), chunk_size)]

# Паралельне кодування через актора
start_time = time.time()
futures = [encoder.encode.remote(chunk) for chunk in chunks]
embeddings_chunks = ray.get(futures)
book_embeddings = torch.cat(embeddings_chunks, dim=0)
execution_time = time.time() - start_time

# Логування часу виконання
wandb.log({"execution_time": execution_time})

# Збереження ембедінгів як артефакту
torch.save(book_embeddings, "models/book_embeddings.pt")
artifact = wandb.Artifact("book_embeddings", type="embeddings")
artifact.add_file("models/book_embeddings.pt")
wandb.log_artifact(artifact)

# Головна функція рекомендації
def recommend(description: str, genre: str):
    # Фільтрація за жанром
    filtered = books_df[books_df["genre"] == genre]
    if filtered.empty:
        print(f"Жодної книги жанру '{genre}' не знайдено.")
        return

    idxs = filtered.index.tolist()
    emb_subset = book_embeddings[idxs].to(device)

    # Кодуємо введений опис
    user_emb = encoder.encode.remote([description])
    user_emb = ray.get(user_emb).to(device)

    # Обчислюємо косинусну подібність
    cos_sim = torch.nn.functional.cosine_similarity(user_emb, emb_subset)
    top_idx = torch.argmax(cos_sim).item()
    rec_idx = idxs[top_idx]

    # Виводимо рекомендацію
    rec = books_df.loc[rec_idx]
    print("\nРекомендована книга:")
    print(f"Title      : {rec['title']}")
    print(f"Description: {rec['description']}")
    print(f"Genre      : {rec['genre']}")

# Взаємодія з користувачем
if __name__ == "__main__":
    print("Введіть короткий опис книги:")
    user_description = sys.stdin.readline().strip()
    print("Введіть жанр книги (Фентезі, Нон-фікшн, Поезія, Проза, Детектив):")
    user_genre = sys.stdin.readline().strip()
    recommend(user_description, user_genre)
    # Очищаємо ресурси Ray
    ray.shutdown()
    # Завершуємо WandB
    wandb.finish()