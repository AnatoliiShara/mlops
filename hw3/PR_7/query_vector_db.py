import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os 

def query_vector_db(query_text, vector_dir="vector_db", top_k=5):
    """Query the FAISS VectorDB."""
    # Load model and data
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.load(os.path.join(vector_dir, "embeddings.npy"))
    metadata = pd.read_csv(os.path.join(vector_dir, "metadata.csv"))
    index = faiss.read_index(os.path.join(vector_dir, "faiss_index.bin"))
    
    # Encode query
    query_embedding = model.encode([query_text])
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Return results
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "text": metadata.iloc[idx]["text"],
            "distance": dist
        })
    return results

if __name__ == "__main__":
    query = "tech article"
    results = query_vector_db(query)
    for res in results:
        print(f"Text: {res['text']}, Distance: {res['distance']:.4f}")