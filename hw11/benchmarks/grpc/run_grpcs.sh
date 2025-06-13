#!/usr/bin/env bash
set -e
python grpc/client.py | tee grpc/raw.json
