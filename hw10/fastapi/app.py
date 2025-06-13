# hw10/fastapi/app.py
# ------------------------------------------------------------------
#  FastAPI-сервер для рекомендаційної системи книжкового магазину.
#  POST /recommend {query,k}  →  список рекомендованих книг.
# ------------------------------------------------------------------
from __future__ import annotations

import os, sys
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize

# ─── зробимо import utils -------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, generate_embeddings, hybrid_search_with_rerank  # noqa: E402

# ─── Шлях до датасету ----------------------------------------------------------
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ukr_books_dataset.csv"))

# ─── FastAPI-додаток -----------------------------------------------------------
app = FastAPI(
    title="Book Recommender API",
    description="Гібридний пошук (BM25 + FAISS + RRF + Cross-Encoder)",
    version="0.1.0",
)

# ─── Pydantic схеми ------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    k: int = 5


class BookOut(BaseModel):
    title: str
    genre: str | None = None
    rating: float | None = None
    description: str


# ─── Одноразове завантаження ресурсів (при старті контейнера) -------------------
df_books = load_data(DATA_PATH)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

corpus = df_books["description"].astype(str).tolist()
tokenized = [t.split() for t in corpus]
bm25_model = BM25Okapi(tokenized)

doc_emb = embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
doc_emb = normalize(doc_emb)

faiss_index = faiss.IndexFlatIP(doc_emb.shape[1])
faiss_index.add(doc_emb)

# ─── Ендпоїнти ------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=List[BookOut])
def recommend(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="`query` must be non-empty")

    results = hybrid_search_with_rerank(
        query=req.query,
        bm25_model=bm25_model,
        faiss_index=faiss_index,
        embeddings=doc_emb,
        texts=df_books["combined_text"].tolist(),
        df=df_books,
        cross_encoder=cross_enc,
        top_k=req.k,
    )

    books: list[BookOut] = []
    for idx, _score in results:
        row = df_books.iloc[idx]
        books.append(
            BookOut(
                title=row["title"],
                genre=row.get("genre"),
                rating=row.get("rating"),
                description=row["description"],
            )
        )
    return books


# ─── Локальний запуск (⇢ uvicorn --reload) -------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
