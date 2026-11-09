#!/bin/bash
# Translate the pilot suite with the base model and with a trained adapter, then compare with Jev.
# Usage: train/run_adapter_compare.sh [adapter directory, default adapters/public-night/best]
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER=$(cd "${1:-adapters/public-night/best}" && pwd)
PORT=8766
SUITE=${SUITE:-suites/pilot.jsonl}
export HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
go build -o runs/suite ./cmd/suite

.venv/bin/python -W ignore -m serve.mps_server --model gemma4e4b --port "$PORT" \
  --adapter-root "$(dirname "$ADAPTER")" > runs/server-adapter-compare.log 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/load_lora_adapter" -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"trained\",\"lora_path\":\"$ADAPTER\"}" > /dev/null

for pair in base:gemma4e4b adapter:trained; do
  label=${pair%%:*}
  for variant in none correct; do
    runs/suite --suite "$SUITE" --endpoint "http://127.0.0.1:$PORT" --model "${pair##*:}" \
      --system blind --variant "$variant" --format chat --out "runs/adapter-$label-$variant.jsonl"
  done
done

.venv/bin/python -W ignore -m train.compare_adapter --suite "$SUITE" \
  --base runs/adapter-base-none.jsonl runs/adapter-base-correct.jsonl \
  --adapter runs/adapter-adapter-none.jsonl runs/adapter-adapter-correct.jsonl \
  | tee runs/adapter-compare.json
