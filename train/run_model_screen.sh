#!/bin/bash
# Step 2: screen other base models on the fresh test set, each in its documented prompt format.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8766
export HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
go build -o runs/suite ./cmd/suite

for pair in hy7b:hy qingqiu9b:qingqiu gemma4e4b:chat index9b:index; do
  model=${pair%%:*}
  format=${pair##*:}
  .venv/bin/python -W ignore -m serve.mps_server --model "$model" --port "$PORT" > "runs/server-$model.log" 2>&1 &
  server=$!
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    sleep 2
  done
  if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "$model did not start, skipping it: see runs/server-$model.log"
    kill "$server" 2>/dev/null || true
    continue
  fi
  results=()
  for variant in none correct; do
    out="runs/screen-$model-$variant.jsonl"
    runs/suite --suite suites/confirm.jsonl --endpoint "http://127.0.0.1:$PORT" --model "$model" \
      --system "$model" --variant "$variant" --format "$format" --out "$out"
    results+=("$out")
  done
  kill "$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
  .venv/bin/python -W ignore -m train.score --suite suites/confirm.jsonl --results "${results[@]}" \
    --out "runs/screen-$model-scores.jsonl" > "runs/screen-$model-summary.json"
done

.venv/bin/python -W ignore -m train.compare screen
