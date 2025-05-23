"""
Search utilities for book recommendations
"""
import numpy as np
from typing import List, Tuple
import pandas as pd

def hybrid_search_with_rerank(
    query: str,
    bm25_model,
    faiss_index,
    embeddings: np.ndarray,
    texts: List[str],
    df: pd.DataFrame,
    cross_encoder,
    embedder,
    top_k: int = 5,
    bm25_k: int = 20,
    faiss_k: int = 20,
    rrf_k: int = 60
) -> List[Tuple[int, float]]:
    """
    Perform hybrid search with reciprocal rank fusion and cross-encoder reranking
    """
    # Encode query
    query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # FAISS search
    faiss_scores, faiss_ids = faiss_index.search(
        np.expand_dims(query_embedding, 0), 
        min(faiss_k, len(texts))
    )
    
    # BM25 search
    tokenized_query = query.split()
    bm25_scores = bm25_model.get_scores(tokenized_query)
    
    # RRF fusion
    rrf_scores = np.zeros(len(texts))
    
    # Add FAISS results
    for rank, idx in enumerate(faiss_ids[0]):
        if idx < len(texts):
            rrf_scores[idx] += 1 / (rrf_k + rank + 1)
    
    # Add BM25 results
    bm25_ranking = np.argsort(-bm25_scores)[:bm25_k]
    for rank, idx in enumerate(bm25_ranking):
        rrf_scores[idx] += 1 / (rrf_k + rank + 1)
    
    # Get top candidates
    top_indices = np.argsort(-rrf_scores)[:top_k * 2]
    candidate_texts = [texts[i] for i in top_indices]
    
    # Cross-encoder reranking
    pairs = [[query, text] for text in candidate_texts]
    ce_scores = cross_encoder.predict(pairs)
    
    # Get final top-k
    ce_ranking = np.argsort(-ce_scores)[:top_k]
    
    # Return indices and scores
    results = []
    for idx in ce_ranking:
        global_idx = top_indices[idx]
        score = ce_scores[idx]
        results.append((global_idx, score))
    
    return results