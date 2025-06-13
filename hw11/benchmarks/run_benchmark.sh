#!/usr/bin/env bash
set -euo pipefail
ROOT=$(dirname "$(realpath "$0")")
cd "$ROOT"

# 0. стартуємо Triton у docker (у фоні)
docker run -d --name tiny_bench \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v "$(realpath ../models)":/models \
  triton_python_backend_custom:latest \
  tritonserver --model-repository=/models
echo "🌟 Triton запущено"; sleep 25   # прогрів

# 1. REST benchmark
locust -f rest_locustfile.py --headless -u 20 -r 5 -t30s \
       --host http://localhost:8000 \
       --json --csv=rest_result

# 2. gRPC benchmark
locust -f grpc_locustfile.py --headless -u 20 -r 5 -t30s \
       --json --csv=grpc_result

docker stop tiny_bench
echo "✅  Benchmark finished. CSV зберігся у benchmarks/"
