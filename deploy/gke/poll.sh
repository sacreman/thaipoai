#!/usr/bin/env bash
# Watches the training Job for one poll leg. The state is in the cluster, not on this runner,
# so the next leg picks up exactly here. Exits 0 when the Job completed or the leg ran out.
# Never prints pod logs: workflow logs on this repository are public.
set -uo pipefail

leg_seconds=19800 # five and a half hours, under the six-hour hosted job limit
max_failures=20   # about twenty minutes of an unreachable cluster

ledger=$(gcloud storage cat "gs://$BUCKET/ledger/$RUN_ID.json") || { echo "no ledger"; exit 1; }
started=$(jq -r .started_epoch <<<"$ledger")
rate=$(jq -r .rate_usd_per_hour <<<"$ledger")
limit=$(jq -r .limit_usd <<<"$ledger")

failures=0
while (( SECONDS < leg_seconds )); do
  spent=$(awk -v r="$rate" -v s="$started" -v n="$(date +%s)" 'BEGIN { printf "%.2f", r * (n - s) / 3600 }')
  if awk -v s="$spent" -v l="$limit" 'BEGIN { exit !(s >= l) }'; then
    echo "hard stop: spent=$spent limit=$limit"
    kubectl -n thaipoai delete job thaipoai-train --wait=false
    exit 1
  fi

  if ! job=$(kubectl -n thaipoai get job thaipoai-train -o json); then
    failures=$((failures + 1))
    echo "$(date -u +%FT%TZ) job unreachable ($failures/$max_failures)"
    if (( failures >= max_failures )); then
      exit 1
    fi
    sleep 60
    continue
  fi
  failures=0

  state=$(jq -r '[.status.conditions[]? | select(.status == "True") | .type] |
    if index("Complete") then "complete" elif index("Failed") then "failed"
    elif index("Suspended") then "queued" else "running" end' <<<"$job")
  retries=$(jq -r '.status.failed // 0' <<<"$job")
  echo "$(date -u +%FT%TZ) state=$state failed_pods=$retries spent_usd=$spent"
  case "$state" in
    complete) exit 0 ;;
    failed) exit 1 ;;
  esac
  sleep 60
done
echo "leg finished with the Job still running; the next leg continues"
