#!/bin/bash
# Run DeepSeek Flash over the held-out suites. A stand-in for production, with no glossary.
set -euo pipefail
cd "$(dirname "$0")/.."

go build -o runs/suite ./cmd/suite

for set in dev dev2 confirm; do
  for variant in none correct; do
    runs/suite --suite "suites/$set.jsonl" --endpoint https://api.deepseek.com --model deepseek-flash \
      --api-key-file "$HOME/.config/thaipoai/deepseek_key" --temperature 0.7 --no-thinking \
      --system deepseek-flash --variant "$variant" --out "runs/$set-deepseek-$variant.jsonl"
  done
done
