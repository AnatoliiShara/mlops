import pandas as pd, json, sys, statistics, pathlib, yaml

out = pathlib.Path("benchmarks/report.md")
rest_stats = pd.read_csv("benchmarks/rest/results_stats.csv")
grpc_raw   = json.load(open("benchmarks/grpc/raw.json"))

tbl = pd.DataFrame(
    [
        {
            "Proto": "REST",
            "RPS": round(rest_stats["Total Requests/s"].iloc[-1], 1),
            "Avg ms": round(rest_stats["Average Response Time"].iloc[-1], 1),
            "P95 ms": round(rest_stats["95%"].iloc[-1], 1),
            "Fail %": round(100 * rest_stats["Fail Ratio"].iloc[-1], 2),
        },
        {
            "Proto": "gRPC",
            "RPS": round(grpc_raw["n"] / 120, 1),     # 120 сек
            "Avg ms": round(grpc_raw["avg"], 1),
            "P95 ms": round(grpc_raw["p95"], 1),
            "Fail %": 0.0,
        },
    ]
)

md = tbl.to_markdown(index=False)
out.write_text(f"### HW11 | Benchmark\n\n{md}\n")
print(md)
