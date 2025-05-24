# hw10/gradio/app.py
"""
Gradio UI для рекомендаційної системи книжок.
Запуск локально:
    python app.py
або
    gradio app.py
"""
from __future__ import annotations
import os, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # додаємо hw10 у PYTHONPATH

import gradio as gr
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize

from utils import load_data, hybrid_search_with_rerank

# ────────── константи/шляхи ──────────
DATA_PATH = pathlib.Path(__file__).with_suffix("")\
               .parent.parent / "ukr_books_dataset.csv"

# ────────── моделі ──────────
EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ────────── індекси ──────────
df_books = load_data(DATA_PATH)
corpus = df_books["description"].astype(str).tolist()
tokenized = [txt.split() for txt in corpus]
BM25_MODEL = BM25Okapi(tokenized)

embeddings = EMBEDDER.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
embeddings = normalize(embeddings)
faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)

# ────────── головна функція ──────────
def recommend(query: str, top_k: int = 5):
    """
    Повертає markdown-рядок із k рекомендаціями.
    """
    if not query.strip():
        return "Введіть, будь ласка, опис книги або жанр 🙂"

    results = hybrid_search_with_rerank(
        query=query,
        bm25_model=BM25_MODEL,
        faiss_index=faiss_index,
        embeddings=embeddings,
        texts=df_books["combined_text"].tolist(),
        df=df_books,
        cross_encoder=CROSS_ENCODER,
        top_k=top_k,
    )

    md_lines = [f"### Рекомендації для запиту **“{query}”**\n"]
    for rank, (row_id, _) in enumerate(results, 1):
        row = df_books.iloc[row_id]
        md_lines += [
            f"**{rank}. {row['title']}**",
            f"*Жанр:* {row.get('genre', 'невідомо')}",
            f"*Рейтинг:* {row.get('rating', '—')}",
            f"{row['description']}",
            "---",
        ]
    return "\n".join(md_lines)


# ────────── Gradio UI ──────────
with gr.Blocks(title="Книжковий Рекомендатор") as demo:
    gr.Markdown("# 📚 Книжковий Рекомендатор (Gradio)")
    with gr.Row():
        query_in = gr.Textbox(
            label="Опишіть книгу або жанр",
            placeholder="Напр. психологічний трилер про особистісний розвиток …",
            lines=2,
        )
        k_in = gr.Slider(1, 20, value=5, step=1, label="Кількість рекомендацій")
    run_btn = gr.Button("🔍 Знайти книги")
    out = gr.Markdown()

    run_btn.click(fn=recommend, inputs=[query_in, k_in], outputs=out)

# дозволяє `python app.py`
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
