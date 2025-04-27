import pandera as pa
from pandera import Column, Check, DataFrameSchema

ratings_schema = DataFrameSchema({
    "user_id": Column(int, Check.gt(0)),
    "book_id": Column(int, Check.gt(0)),
    # приймаємо будь-який числовий dtype і приводимо до float
    "rating":  Column(float, Check.in_range(1, 5), coerce=True),
})


def test_ratings_schema(ratings_df):
    ratings_schema.validate(ratings_df)


def test_books_basic(books_df):
    assert books_df["description"].notnull().all()
    assert books_df["genre"].nunique() <= 50
