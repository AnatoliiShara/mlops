import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt

# Глобальна модель для уникнення проблем із серіалізацією
MODEL = None

def generate_data(n_samples=100000):
    """Generate synthetic dataset for classification."""
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def single_process_inference(model, X):
    """Inference with a single process."""
    start_time = time.time()
    predictions = model.predict(X)
    return time.time() - start_time

def predict_chunk(chunk):
    """Helper function for multi-process inference."""
    return MODEL.predict(chunk)

def multi_process_inference(model, X, n_processes=4):
    """Inference with multiple processes."""
    global MODEL
    MODEL = model  # Передаємо модель у глобальну змінну
    chunks = np.array_split(X, n_processes)
    start_time = time.time()
    with Pool(n_processes) as pool:
        predictions = pool.map(predict_chunk, chunks)
    return time.time() - start_time

def benchmark_inference():
    """Benchmark single vs multi-process inference."""
    X, y = generate_data()
    model = LogisticRegression().fit(X[:1000], y[:1000])  # Train on subset
    
    results = {"processes": [], "time": []}
    
    # Single process
    single_time = single_process_inference(model, X)
    results["processes"].append("Single")
    results["time"].append(single_time)
    print(f"Single process: {single_time:.4f}s")
    
    # Multi-process (2, 4, 8)
    for n_processes in [2, 4, 8]:
        multi_time = multi_process_inference(model, X, n_processes)
        results["processes"].append(f"Multi-{n_processes}")
        results["time"].append(multi_time)
        print(f"Multi-process ({n_processes}): {multi_time:.4f}s")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/inference_results.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.bar(results["processes"], results["time"], color="coral")
    plt.title("Inference Time by Process Count")
    plt.ylabel("Time (seconds)")
    plt.savefig("results/inference_plot.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    benchmark_inference()