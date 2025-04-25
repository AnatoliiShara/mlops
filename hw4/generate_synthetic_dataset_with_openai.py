import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Завантажити змінні середовища з файлу .env
load_dotenv()

# Отримати API-ключ із змінної середовища
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in .env file")

# Налаштування API-ключа
client = OpenAI(api_key=api_key)

# Функція для генерації одного запису про книгу через ChatGPT
def generate_book_entry():
    prompt = """Створи опис книги українською мовою для рекомендаційної системи. Дай відповідь у форматі JSON із полями:
    - "title": назва книги (унікальна, українською мовою),
    - "description": короткий опис (3–5 речень, українською мовою),
    - "genre": жанр (вибери один із: Проза, Поезія, Фентезі, Детектив, Нон-фікшн),
    - "rating": рейтинг (число від 4.1 до 5.0, з одним знаком після коми).
    Приклад:
    {
        "title": "Світло в темряві",
        "description": "Роман про молоду дівчину, яка шукає своє місце в світі після втрати сім’ї. Вона подорожує Карпатами, де відкриває таємницю стародавнього обряду. Книга сповнена містики та глибоких роздумів про сенс життя.",
        "genre": "Проза",
        "rating": 4.7
    }
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ти — експерт із літератури, який генерує описи книг українською мовою."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return eval(response.choices[0].message.content)

# Генерація 50 записів
books = []
for i in range(50):
    print(f"Generating book entry {i+1}/50...")
    try:
        book = generate_book_entry()
        books.append(book)
    except Exception as e:
        print(f"Error generating entry {i+1}: {e}")
        continue

# Створення DataFrame
df = pd.DataFrame(books)

# Збереження в CSV
os.makedirs("data", exist_ok=True)
df.to_csv("data/synthetic_books_dataset.csv", index=False)

print("Synthetic dataset created at data/synthetic_books_dataset.csv")