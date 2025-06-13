import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import faiss, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize

from utils import load_data, generate_embeddings, hybrid_search_with_rerank

st.set_page_config(page_title="Книжкова Рекомендація", layout="wide")
st.title("📚 Книжковий Рекомендатор")

# ─────────────────── Моделі ───────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    st.info("⬇️ Завантажуємо моделі…")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker


@st.cache_data(show_spinner=False)
def prepare_index(df, _embedder):            # ← підкреслення!
    corpus = df["description"].astype(str).tolist()
    tokenized = [t.split() for t in corpus]

    bm25 = BM25Okapi(tokenized)

    emb = _embedder.encode(                  # ← змінна з підкресленням
        corpus, convert_to_numpy=True, show_progress_bar=True
    )
    emb = normalize(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, bm25, emb


# ─────────────────── DATA ─────────────────────
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ukr_books_dataset.csv")
)
df_books = load_data(DATA_PATH)

embedder, cross_enc = load_models()
faiss_index, bm25_model, doc_emb = prepare_index(df_books, embedder)

# ───────────────── UI ─────────────────────────
st.subheader("Опишіть книгу або жанр")
query = st.text_input(
    "Доброго дня! Опишіть книгу чи жанр, який шукаєте:",
    placeholder="Напр. психологічний трилер про особистісний розвиток…",
)
k = st.slider("Скільки рекомендацій показати?", 1, 20, 5)

if st.button("🔍 Знайти книги") and query:
    with st.spinner("Пошук…"):
        results = hybrid_search_with_rerank(
            query=query,
            bm25_model=bm25_model,
            faiss_index=faiss_index,
            embeddings=doc_emb,
            texts=df_books["combined_text"].tolist(),
            df=df_books,
            cross_encoder=cross_enc,
            top_k=k,
        )

    st.success(f"Знайдено {len(results)} рекомендацій:")
    for rank, (row_id, score) in enumerate(results, 1):
        row = df_books.iloc[row_id]
        st.markdown(f"**{rank}. {row['title']}**")
        st.markdown(f"_Жанр:_ {row.get('genre', 'невідомо')}")
        st.markdown(f"_Рейтинг:_ {row.get('rating', '—')}")
        st.markdown(f"_Опис:_ {row['description']}")
        st.markdown("---")
