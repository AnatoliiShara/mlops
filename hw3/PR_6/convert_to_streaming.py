import pandas as pd
import numpy as np
from streaming import MDSWriter
import os

def generate_data(n_samples=10000):
    """Generate synthetic dataset."""
    data = {
        "id": np.arange(n_samples, dtype=np.int64),  # Явно вказуємо int64
        "value": np.random.randn(n_samples),  # float64 за замовчуванням
        "category": np.random.choice(["A", "B", "C"], n_samples)
    }
    return pd.DataFrame(data)

def convert_to_streaming(df, output_dir="streaming_data"):
    """Convert DataFrame to StreamingDataset format."""
    os.makedirs(output_dir, exist_ok=True)
    
    columns = {
        "id": "int64",    # Змінено з int на int64
        "value": "float64",  # Змінено з floatcopy
        "category": "str"
    }
    
    with MDSWriter(out=output_dir, columns=columns, compression=None) as writer:
        for _, row in df.iterrows():
            sample = {
                "id": row["id"],
                "value": row["value"],
                "category": row["category"]
            }
            writer.write(sample)
    
    print(f"StreamingDataset saved to {output_dir}")

if __name__ == "__main__":
    df = generate_data()
    convert_to_streaming(df)