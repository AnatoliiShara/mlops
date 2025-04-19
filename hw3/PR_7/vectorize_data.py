import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

def generate_data(n_samples=1000):
    """Generate synthetic text data."""
    categories = ["tech", "food", "travel"]
    texts = [f"This is a {np.random.choice(categories)} article {i}" for i in range(n_samples)]
    return pd.DataFrame({"text": texts})

def vectorize_and_save(df, output_dir="vector_db"):
    """Vectorize texts and save to FAISS."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    
    # Save embeddings and metadata
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    
    print(f"VectorDB saved to {output_dir}")

if __name__ == "__main__":
    df = generate_data()
    vectorize_and_save(df)