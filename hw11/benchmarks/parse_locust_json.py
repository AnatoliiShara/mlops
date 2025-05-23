import json, pathlib, pandas as pd

def read_json(fp):
    with open(fp) as f:
        data = json.load(f)
    row = data["stats_total"]
    return {
        "req/s": row["total_rps"],
        "avg (ms)": row["avg_response_time"],
        "p95 (ms)": row["current_response_time_percentile_95"],
        "fail %": row["fail_ratio"] * 100,
    }

def main():
    base = pathlib.Path(__file__).parent
    rest = read_json(base / "rest_result_stats.json")
    grpc = read_json(base / "grpc_result_stats.json")
    df = pd.DataFrame([rest, grpc], index=["REST", "gRPC"])
    print("## Latency / RPS benchmark\n")
    print(df.to_markdown())

if __name__ == "__main__":
    main()
