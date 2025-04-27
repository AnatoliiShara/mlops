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
import argparse
import sys

# ---------- DATA ----------
BASE_DIR = Path(__file__).resolve().parents[2]   # hw6/
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # агрегуємо дублікати (user_id, book_id)
    if df.duplicated(subset=["user_id", "book_id"]).any():
        df = (
            df.groupby(["user_id", "book_id"], as_index=False)["rating"]
            .mean().round().astype(int)
        )
    df["rating"] = df["rating"].astype(float)
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
    sims = cosine_similarity(
        tfidf.transform([description]),
        tfidf.transform(filtered["description"])
    ).flatten()
    best_idx = filtered.index[np.argmax(sims)]
    return books_df.loc[best_idx]

# ---------- MAIN ----------
def main(no_cli: bool = False):
    cfg = dict(n_factors=10, n_epochs=20, lr_all=0.01, reg_all=0.01)
    run = wandb.init(project="book-recommendation-system-hw5", config=cfg)

    ratings = load_ratings(DATA_DIR / "user_book_ratings.csv")
    books   = load_books(DATA_DIR / "ukr_books_dataset.csv")

    model, rmse = train_svd(ratings, cfg)
    wandb.log({"RMSE": rmse})

    model_path = MODEL_DIR / "svd_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # ── Versioning with W&B Artifacts ───────────────────────────────
    artifact = wandb.Artifact(
        name="svd_model",
        type="model",
        metadata={**cfg, "rmse": float(rmse)},
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)

    # ── CLI interaction (skip if --no-cli) ──────────────────────────
    if not no_cli:
        tfidf = TfidfVectorizer(max_features=5000).fit(books["description"])
        desc  = input("Введіть короткий опис книги:\n").strip()
        genre = input("Введіть жанр книги:\n").strip()
        rec   = get_recommendation(books, tfidf, desc, genre)
        if rec is None:
            print(f"Жанр «{genre}» не знайдено.")
        else:
            print("\nРекомендована книга:")
            print(f"{rec['title']} — {rec['genre']}")

    run.finish()

# ---------- Entry point ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cli", action="store_true",
                        help="Skip interactive prompts (useful for CI)")
    args = parser.parse_args()
    try:
        main(no_cli=args.no_cli)
    except KeyboardInterrupt:
        sys.exit(0)
