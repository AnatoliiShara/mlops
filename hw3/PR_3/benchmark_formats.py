import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def generate_data(size=1000000):
    """Generate synthetic dataset."""
    data = {
        "id": np.arange(size),
        "value": np.random.randn(size),
        "category": np.random.choice(["A", "B", "C"], size),
        "text": [f"Text_{i}" for i in range(size)]
    }
    return pd.DataFrame(data)

def benchmark_formats(df, output_dir="results"):
    """Benchmark save/load times for different formats."""
    formats = {
        "csv": {"save": lambda df, path: df.to_csv(path, index=False),
                "load": lambda path: pd.read_csv(path)},
        "parquet": {"save": lambda df, path: df.to_parquet(path),
                    "load": lambda path: pd.read_parquet(path)},
        "feather": {"save": lambda df, path: df.to_feather(path),
                    "load": lambda path: pd.read_feather(path)},
        "hdf5": {"save": lambda df, path: df.to_hdf(path, key="data", mode="w"),
                 "load": lambda path: pd.read_hdf(path, key="data")},
        "pickle": {"save": lambda df, path: df.to_pickle(path),
                   "load": lambda path: pd.read_pickle(path)}
    }

    results = {"format": [], "save_time": [], "load_time": []}
    os.makedirs(output_dir, exist_ok=True)

    for fmt, funcs in formats.items():
        save_path = os.path.join(output_dir, f"data.{fmt}")
        # Benchmark save
        start_time = time.time()
        funcs["save"](df, save_path)
        save_time = time.time() - start_time
        # Benchmark load
        start_time = time.time()
        _ = funcs["load"](save_path)
        load_time = time.time() - start_time
        # Store results
        results["format"].append(fmt)
        results["save_time"].append(save_time)
        results["load_time"].append(load_time)
        print(f"{fmt}: Save={save_time:.4f}s, Load={load_time:.4f}s")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(results["format"], results["save_time"], color="skyblue")
    ax[0].set_title("Save Time")
    ax[0].set_ylabel("Time (seconds)")
    ax[1].bar(results["format"], results["load_time"], color="lightgreen")
    ax[1].set_title("Load Time")
    ax[1].set_ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_plot.png"))
    plt.close()

if __name__ == "__main__":
    df = generate_data()
    benchmark_formats(df, output_dir="results")