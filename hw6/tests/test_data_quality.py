import numpy as np
import pandas as pd

# ---------- Ratings CSV ---------------------------------------------------

def test_ratings_no_duplicates(ratings_df):
    """(user_id, book_id) має бути унікальною парою."""
    duplicated = ratings_df.duplicated(subset=["user_id", "book_id"]).sum()
    assert duplicated == 0


def test_ratings_range_and_dtype(ratings_df):
    """Рейтинг — int або float у межах 1..5 і без NaN."""
    assert ratings_df["rating"].between(1, 5).all()
    assert pd.api.types.is_numeric_dtype(ratings_df["rating"])
    assert ratings_df["rating"].notnull().all()


# ---------- Books CSV -----------------------------------------------------

def test_books_unique_ids(books_df):
    """book_id має бути унікальним."""
    assert books_df["book_id"].is_unique


def test_books_non_empty_descr(books_df):
    """Описи книг не повинні бути порожніми чи NaN."""
    assert books_df["description"].notnull().all()
    assert (books_df["description"].str.strip() != "").all()


def test_books_genre_coverage(books_df):
    """
    Мінімум 5 різних жанрів, щоб треновані рекомендації
    не працювали лише на одному-двох жанрах.
    """
    assert books_df["genre"].nunique() >= 5
