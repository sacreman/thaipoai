#!/usr/bin/env bash
# Both demos with the real model on the Mac, against really trained bad adapters.
set -uo pipefail
source deploy/kind/lib.sh
k rollout status deployment/rollout-controller --timeout=180s >/dev/null
k rollout status deployment/gateway --timeout=180s >/dev/null

for adapter in flipped bloat; do
  name=demo-$adapter
  sha=$(pack "adapters/$name" "$name")
  echo "--- $name"
  rollout "$name" "$sha" 50 40 30
  new_rollout_seen "$sha"
  started=$(date +%s)
  k scale deployment/traffic --replicas=1 >/dev/null
  wait_for 2400 Promoted Blocked RolledBack Inconclusive
  k scale deployment/traffic --replicas=0 >/dev/null
  echo "phase: $(field .status.phase) after $(( $(date +%s) - started ))s"
  echo "reason: $(field .status.reason)"
  echo "golden: $(field .status.golden)"
  echo "shadow: $(field .status.shadow)"
  echo "canary: $(field .status.canary)"
  sleep 8
  k get configmap rollout-translate -o 'jsonpath=routing: {.data.routing\.json}{"\n"}'
done
