import pandas as pd
import numpy as np
from cleanlab.classification import CleanLearning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 1. Завантаж розмічені дані
df = pd.read_csv("data/labeled_synthetic_books_dataset_updated.csv")

# 2. Перевір розподіл жанрів
print("Розподіл жанрів у датасеті:")
print(df["genre"].value_counts())

# 3. Підготуй тексти та мітки
texts = df["description"].values
labels = df["genre"].values

# 4. Кодуй жанри в числа
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 5. TF-IDF векторизація
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts).toarray()

# 6. Побудова базової моделі
base_model = LogisticRegression(random_state=42, max_iter=1000)

# 7. Ініціалізація CleanLearning з кастомними налаштуваннями для фільтрації
cl = CleanLearning(
    base_model,
    find_label_issues_kwargs={
        "filter_by": "prune_by_noise_rate",
        "min_examples_per_class": 5
    }
)

# 8. Пошук потенційних проблем з мітками
print("Finding label issues...")
label_issues_df = cl.find_label_issues(X, labels_encoded)

# 9. Витяг булевого масиву помилок
label_issues = label_issues_df["is_label_issue"].values

# 10. Додаємо інформацію про помилки в DataFrame
df["label_issue"] = label_issues

# 11. Виводимо записи з потенційними помилками
issues = df[df["label_issue"]]
print("Records with potential label issues:")
print(issues[["description", "genre"]])

# 12. Збереження результатів
output_path = "data/labeled_synthetic_books_dataset_with_issues.csv"
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
