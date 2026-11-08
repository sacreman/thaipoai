#!/bin/bash
# Step 2: run untuned 8-bit Lumen on the pilot suite in all three context variants, then score it.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8765
export HF_HUB_OFFLINE=1
go build -o runs/suite ./cmd/suite

.venv/bin/python -W ignore -m mlx_lm.server --model models/lumen-8bit --port "$PORT" > runs/lumen8-server.log 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
  sleep 2
done

for variant in none correct irrelevant; do
  runs/suite --suite suites/pilot.jsonl --endpoint "http://127.0.0.1:$PORT" --model default_model \
    --system lumen8 --variant "$variant" --out "runs/gonogo-$variant.jsonl"
done

kill "$SERVER" 2>/dev/null || true
trap - EXIT

.venv/bin/python -W ignore -m train.score --suite suites/pilot.jsonl \
  --results runs/gonogo-none.jsonl runs/gonogo-correct.jsonl runs/gonogo-irrelevant.jsonl \
  --out runs/gonogo-scores.jsonl | tee runs/gonogo-summary.json
