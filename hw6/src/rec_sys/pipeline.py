# hw6/src/rec_sys/pipeline.py

from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ---------- DATA ----------
BASE_DIR = Path(__file__).resolve().parents[2]   # hw6/
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)                   # створюємо models/, якщо нема

def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["user_id", "book_id", "rating"]]

def load_books(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "book_id" not in df.columns:
        df["book_id"] = df.index + 1
    return df

# ---------- MODEL ----------
def train_svd(ratings: pd.DataFrame, cfg: dict, seed: int = 42):
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings, reader)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=seed)

    model = SVD(**cfg)
    model.fit(trainset)

    rmse = accuracy.rmse(model.test(testset), verbose=False)
    return model, rmse

# ---------- RECOMMENDATION ----------
def get_recommendation(books_df: pd.DataFrame,
                       tfidf: TfidfVectorizer,
                       description: str,
                       genre: str):
    filtered = books_df[books_df["genre"] == genre]
    if filtered.empty:
        return None
    filtered_tfidf = tfidf.transform(filtered["description"])
    sims = cosine_similarity(tfidf.transform([description]),
                             filtered_tfidf).flatten()
    best_idx = filtered.index[np.argmax(sims)]
    return books_df.loc[best_idx]

# ---------- CLI ----------
def main():
    cfg = dict(n_factors=10, n_epochs=20, lr_all=0.01, reg_all=0.01)
    wandb.init(project="book-recommendation-system-hw5", config=cfg)

    ratings = load_ratings(DATA_DIR / "user_book_ratings.csv")
    books   = load_books(DATA_DIR / "ukr_books_dataset.csv")

    model, rmse = train_svd(ratings, cfg)
    wandb.log({"RMSE": rmse})

    with open(MODEL_DIR / "svd_model.pkl", "wb") as f:
        pickle.dump(model, f)

    wandb.finish()

    tfidf = TfidfVectorizer(max_features=5000).fit(books["description"])

    desc  = input("Введіть короткий опис книги:\n").strip()
    genre = input("Введіть жанр книги:\n").strip()

    rec = get_recommendation(books, tfidf, desc, genre)

    if rec is None:
        print(f"Жанр «{genre}» не знайдено.")
    else:
        print("\nРекомендована книга:")
        print(f"{rec['title']} — {rec['genre']}")

if __name__ == "__main__":
    main()
