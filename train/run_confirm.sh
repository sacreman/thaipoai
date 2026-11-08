#!/bin/bash
# Step 2: one-shot confirmation of the system-message format on fresh cases.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8765
export HF_HUB_OFFLINE=1
go build -o runs/suite ./cmd/suite

if [ ! -s suites/confirm.jsonl ]; then
  .venv/bin/python -W ignore -m train.draft_suite --model deepseek-v4-pro --exclude suites/pilot.jsonl --out suites/confirm.jsonl
fi

.venv/bin/python -W ignore -m mlx_lm.server --model models/lumen-8bit --port "$PORT" > runs/lumen8-server.log 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
  sleep 2
done

for variant in none correct irrelevant; do
  runs/suite --suite suites/confirm.jsonl --endpoint "http://127.0.0.1:$PORT" --model default_model \
    --system lumen8 --variant "$variant" --format system --out "runs/confirm-$variant.jsonl"
done

kill "$SERVER" 2>/dev/null || true
trap - EXIT

.venv/bin/python -W ignore -m train.score --suite suites/confirm.jsonl \
  --results runs/confirm-none.jsonl runs/confirm-correct.jsonl runs/confirm-irrelevant.jsonl \
  --out runs/confirm-scores.jsonl | tee runs/confirm-summary.json
