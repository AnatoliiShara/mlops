import pytest
from pathlib import Path
from rec_sys.pipeline import load_ratings, load_books

# ── базовий каталог hw6/  ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]   # ← було [2], стало [1]
DATA_DIR = BASE_DIR / "data"

@pytest.fixture(scope="session")
def ratings_df():
    return load_ratings(DATA_DIR / "user_book_ratings.csv")

@pytest.fixture(scope="session")
def books_df():
    return load_books(DATA_DIR / "ukr_books_dataset.csv")
