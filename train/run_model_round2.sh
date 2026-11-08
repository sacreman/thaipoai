#!/bin/bash
# Step 2: second round. Gemma 4 E4B on the other test set, three new models on the fresh set.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8766
export HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
go build -o runs/suite ./cmd/suite

for job in gemma4e4b:chat:pilot:screen2 gemma412b:chat:confirm:screen chinda4b:chinda:confirm:screen mango9b:chat:confirm:screen; do
  IFS=: read -r model format suite tag <<< "$job"
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
    out="runs/$tag-$model-$variant.jsonl"
    runs/suite --suite "suites/$suite.jsonl" --endpoint "http://127.0.0.1:$PORT" --model "$model" \
      --system "$model" --variant "$variant" --format "$format" --out "$out"
    results+=("$out")
  done
  kill "$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
  .venv/bin/python -W ignore -m train.score --suite "suites/$suite.jsonl" --results "${results[@]}" \
    --out "runs/$tag-$model-scores.jsonl" > "runs/$tag-$model-summary.json"
done

.venv/bin/python -W ignore -m train.compare screen
.venv/bin/python -W ignore -m train.compare first
