import builtins
from rec_sys import pipeline

def test_recommendation_flow(monkeypatch, books_df):
    answers = iter(["Магічна подорож у часі", books_df["genre"].iloc[0]])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))

    tfidf = pipeline.TfidfVectorizer(max_features=100).fit(books_df["description"])
    rec = pipeline.get_recommendation(books_df, tfidf, "Магічна подорож у часі",
                                      books_df["genre"].iloc[0])

    assert rec is not None
