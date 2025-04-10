# Використовуємо офіційний образ Python
FROM python:3.9-slim

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо файли в контейнер
COPY app.py .

# Встановлюємо Flask
RUN pip install flask

# Вказуємо команду для запуску сервера
CMD ["python", "app.py"]

# Відкриваємо порт 5000
EXPOSE 5000