import sqlite3
import pandas as pd

# Перевірка вмісту SQLite бази
conn = sqlite3.connect("data/books.db")
df = pd.read_sql_query("SELECT * FROM books", conn)
conn.close()

print(f"Number of records: {len(df)}")
print(df.head())