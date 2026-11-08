#!/bin/bash
# Step 2: try other ways of sending chat history to untuned 8-bit Lumen, under the same rules.
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

for format in reference system before; do
  for variant in correct irrelevant; do
    runs/suite --suite suites/pilot.jsonl --endpoint "http://127.0.0.1:$PORT" --model default_model \
      --system lumen8 --variant "$variant" --format "$format" --out "runs/fmt-$format-$variant.jsonl"
  done
done

kill "$SERVER" 2>/dev/null || true
trap - EXIT

for format in reference system before; do
  echo "== $format"
  .venv/bin/python -W ignore -m train.score --suite suites/pilot.jsonl \
    --results "runs/fmt-$format-correct.jsonl" "runs/fmt-$format-irrelevant.jsonl" \
    --reuse-none runs/gonogo-scores.jsonl --out "runs/fmt-$format-scores.jsonl" | tee "runs/fmt-$format-summary.json"
done
