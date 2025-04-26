import pandas as pd
import wandb
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys

# Ініціалізація W&B
wandb.init(project="book-recommendation-system-hw5", name="svd-training-run")

# 1. Завантаження даних для Collaborative Filtering
ratings_data = pd.read_csv("data/user_book_ratings.csv")
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(
    ratings_data[["user_id", "book_id", "rating"]],
    reader
)

# 2. Розділення на train/test
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# 3. Гіперпараметри
config = {
    "n_factors": 10,    # Найкраще значення з W&B Sweeps
    "n_epochs": 20,     # Залишаємо без змін
    "lr_all": 0.01,     # Найкраще значення з W&B Sweeps
    "reg_all": 0.01     # Найкраще значення з W&B Sweeps
}
wandb.config.update(config)

# 4. Тренування SVD
model = SVD(
    n_factors=config["n_factors"],
    n_epochs=config["n_epochs"],
    lr_all=config["lr_all"],
    reg_all=config["reg_all"]
)
model.fit(trainset)

# 5. Оцінка
predictions = model.test(testset)
rmse = accuracy.rmse(predictions, verbose=False)
mae  = accuracy.mae(predictions, verbose=False)

# 6. Логування метрик
wandb.log({"RMSE": rmse, "MAE": mae})

# 7. Збереження моделі
with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(model, f)
artifact = wandb.Artifact("svd-model", type="model")
artifact.add_file("models/svd_model.pkl")
wandb.log_artifact(artifact)

# ──────────────────────────────────────────────────────────────────────────── #

# 8. Завантажуємо метадані книг (файл з описами і жанрами!)
books_df = pd.read_csv("data/ukr_books_dataset.csv")

# Перевіримо, що потрібні колонки є
required_cols = {"description", "genre"}
if not required_cols.issubset(books_df.columns):
    raise KeyError(f"У файлі метаданих немає колонок {required_cols}")

# Додаємо унікальний book_id (якщо його ще немає)
if "book_id" not in books_df.columns:
    books_df["book_id"] = books_df.index + 1

# 9. TF-IDF векторизація описів
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(books_df["description"])

# 10. Отримуємо введення від користувача
print("Введіть короткий опис книги:")
user_description = input().strip()
print("Введіть жанр книги (наприклад, Фентезі, Нон-фікшн, Поезія, Проза, Детектив):")
user_genre = input().strip()

# 11. Фільтрація за жанром
filtered = books_df[books_df["genre"] == user_genre]
if filtered.empty:
    print(f"Жодної книги жанру '{user_genre}' не знайдено.")
    sys.exit(0)

# 12. Обчислення косинусної подібності
filtered_idxs = filtered.index
filtered_tfidf = tfidf_matrix[filtered_idxs]
user_tfidf = vectorizer.transform([user_description])
sims = cosine_similarity(user_tfidf, filtered_tfidf).flatten()

# 13. Обираємо найсхожішу книгу
best_idx_in_filtered = np.argmax(sims)
rec_idx = filtered_idxs[best_idx_in_filtered]
rec_book = books_df.loc[rec_idx]

# 14. Виводимо результат
print("\nРекомендована книга:")
print(f"Опис : {rec_book['description']}")
print(f"Жанр : {rec_book['genre']}")

# Завершуємо W&B
wandb.finish()
