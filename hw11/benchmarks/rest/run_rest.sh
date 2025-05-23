#!/usr/bin/env bash
set -e
export REST_HOST=${1:-http://localhost:8000}

locust -f rest/locustfile.py \
       --headless -u 50 -r 10 \
       --run-time 2m \
       --csv rest/results
