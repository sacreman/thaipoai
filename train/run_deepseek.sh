#!/bin/bash
# Step 2: score DeepSeek Flash on the same cases Lumen was tested on. Synthetic text only.
set -euo pipefail
cd "$(dirname "$0")/.."

go build -o runs/suite ./cmd/suite

for set in pilot confirm; do
  results=()
  for variant in none correct irrelevant; do
    out="runs/ds-$set-$variant.jsonl"
    runs/suite --suite "suites/$set.jsonl" --endpoint https://api.deepseek.com --model deepseek-flash \
      --api-key-file "$HOME/.config/thaipoai/deepseek_key" --temperature 0.7 --no-thinking \
      --system deepseek-flash --variant "$variant" --format chat --out "$out"
    results+=("$out")
  done
  .venv/bin/python -W ignore -m train.score --suite "suites/$set.jsonl" --results "${results[@]}" \
    --out "runs/ds-$set-scores.jsonl" > "runs/ds-$set-summary.json"
done

.venv/bin/python -W ignore -m train.compare
