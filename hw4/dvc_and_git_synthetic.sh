#!/bin/bash
# Перейти до hw4
cd ~/Documents/mlops_projector/hw4

# Запустити скрипт для генерації датасету
python3 generate_synthetic_dataset_with_openai.py

# Додати файл до DVC
dvc add data/synthetic_books_dataset.csv

# Додати DVC-файли і скрипт до Git
git add data/synthetic_books_dataset.csv.dvc data/.gitignore generate_synthetic_dataset_with_openai.py

# Закомітити зміни
git commit -m "PR3: Add synthetic dataset generated with ChatGPT API"

# Запушити в гілку student
git push origin student

# Перевірити статус
git status