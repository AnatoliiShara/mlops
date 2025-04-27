import pandas as pd
import sqlite3
import os

# Шлях до датасету
DATASET_PATH = "/home/anatolii-shara/Downloads/ukr_books_dataset.csv"
DB_PATH = "books.db"

# Створити директорію для даних, якщо її немає
os.makedirs("data", exist_ok=True)

# Читання CSV
df = pd.read_csv(DATASET_PATH)

# Створення SQLite бази
conn = sqlite3.connect(os.path.join("data", DB_PATH))
df.to_sql("books", conn, if_exists="replace", index=False)
conn.close()

print(f"SQLite database created at data/{DB_PATH}")