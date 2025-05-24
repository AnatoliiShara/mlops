# hw10/utils.py
"""
Утиліти для пошуку книжок:
  • завантаження датасету
  • побудова ембедінгів
  • BM25, FAISS, RRF-ф’южн
  • фінальний rerank Cross-Encoder’ом
"""
from typing import List, Tuple

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ─────────────────────────── DATA ────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["title", "description"])
    df["combined_text"] = df["title"] + " " + df["description"]
    return df


# ────────────────────── EMBEDDINGS & INDICES ─────────────────
def generate_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True
    )
    return emb


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ─────────────────────────── BM25 (Tf-idf) ───────────────────
def bm25_score(query: str, texts: List[str]) -> np.ndarray:
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(texts)
    q_vec = vect.transform([query])
    return (tfidf @ q_vec.T).toarray().squeeze()


# ──────────────────────────  RRF fuse  ───────────────────────
def rrf(bm25_s: np.ndarray, faiss_s: np.ndarray, k: int = 60) -> np.ndarray:
    bm25_rank = np.argsort(-bm25_s)
    faiss_rank = np.argsort(-faiss_s)
    rrf_s = np.zeros(len(bm25_s))
    for rank_list in [bm25_rank, faiss_rank]:
        for r, idx in enumerate(rank_list):
            rrf_s[idx] += 1 / (k + r + 1)
    return np.argsort(-rrf_s)  # від більшого до меншого


# ───────────────────── Hybrid + Cross-Encoder ────────────────
def rerank_with_crossencoder(
    query: str, docs: List[str], top_k: int = 5
) -> List[int]:
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, d] for d in docs]
    scores = ce.predict(pairs)
    return np.argsort(-scores)[:top_k]


def hybrid_search_with_rerank(
    *,
    query: str,
    bm25_model,
    faiss_index,
    embeddings: np.ndarray,
    texts: List[str],
    df: pd.DataFrame,
    cross_encoder,
    top_k: int = 5,
    bm25_k: int = 20,
    faiss_k: int = 20,
    rrf_k: int = 60,
) -> List[Tuple[int, float]]:
    # — FAISS ­—————————————————————————
    q_emb = generate_embeddings([query])[0]
    faiss_scores, faiss_ids = faiss_index.search(
        np.expand_dims(q_emb, 0), faiss_k
    )  # (1, faiss_k)
    faiss_scores = faiss_scores.flatten()
    # put scores back to full-size vector
    faiss_score_full = np.zeros(len(texts))
    faiss_score_full[faiss_ids.flatten()] = faiss_scores

    # — BM25 ­———————————————————————————
    bm25_scores = bm25_score(query, texts)

    # — RRF fuse ­———————————————
    fused_idx = rrf(bm25_scores, faiss_score_full, k=rrf_k)

    # top candidates for cross-encoder
    candidate_idx = fused_idx[:bm25_k]
    candidate_texts = [texts[i] for i in candidate_idx]

    rerank_order = rerank_with_crossencoder(query, candidate_texts, top_k=top_k)

    final = []
    for pos in rerank_order:
        global_idx = candidate_idx[pos]
        final.append((global_idx, faiss_score_full[global_idx]))
    return final
