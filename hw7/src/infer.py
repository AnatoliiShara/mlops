import pandas as pd
import joblib
import os
from sentence_transformers import SentenceTransformer

def run_inference(data_path: str, model_path: str, output_path: str):
    df = pd.read_csv(data_path)
    texts = df['description'].astype(str).tolist()

    model, _ = joblib.load(model_path)
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.to_csv(output_path, index=False)

    print(f"✅ Inference saved to {output_path}")
