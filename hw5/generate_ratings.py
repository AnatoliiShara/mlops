import pandas as pd
import numpy as np

# Завантаження датасету з книгами
books_df = pd.read_csv("/home/anatolii-shara/Downloads/ukr_books_dataset.csv")

# Присвоюємо унікальний book_id для кожної книги (індекс + 1)
books_df["book_id"] = books_df.index + 1

# Генерація синтетичних даних: 100 користувачів, випадкові оцінки
np.random.seed(42)
num_users = 100
num_ratings = 500  # Кількість оцінок

# Генерація user_id, book_id, rating
user_ids = np.random.randint(1, num_users + 1, num_ratings)
book_ids = np.random.choice(books_df["book_id"], num_ratings)
ratings = np.random.randint(1, 6, num_ratings)  # Оцінки від 1 до 5

# Створення DataFrame
ratings_df = pd.DataFrame({
    "user_id": user_ids,
    "book_id": book_ids,
    "rating": ratings
})

# Збереження у файл
ratings_df.to_csv("data/user_book_ratings.csv", index=False)
print("Synthetic ratings dataset generated and saved to data/user_book_ratings.csv")