import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
import os

def train_model(data_path: str, model_dir: str):
    df = pd.read_csv(data_path)
    texts = df['description'].astype(str).tolist()  # або інша колонка

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump((model, embeddings), os.path.join(model_dir, "sentence_model.joblib"))
