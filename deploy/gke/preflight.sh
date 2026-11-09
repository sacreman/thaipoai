#!/usr/bin/env bash
# Refuses to start a run without GPU quota, a healthy reaper and room under the hard stop.
# Prints numbers only: workflow logs on this repository are public.
set -euo pipefail

hard_stop_usd=150
reaper_margin_hours=6 # spec 7.1: the reaper's deadline is six hours past the Job deadline
max_job_hours=20      # four poll legs of five and a half hours, less room to finish

# Ceilings rather than estimates, so the ledger can only overstate. On-demand GPU and VM, plus
# the system node, cluster fee and disk. The owner confirms these against current prices.
case "$GPU" in
  l4)   metric=NVIDIA_L4_GPUS;   rate=1.50 ;;
  a100) metric=NVIDIA_A100_GPUS; rate=5.00 ;;
  *)    echo "unknown gpu"; exit 1 ;;
esac

if ! [[ "$JOB_HOURS" =~ ^[0-9]+$ ]] || (( JOB_HOURS < 1 || JOB_HOURS > max_job_hours )); then
  echo "job_hours must be a whole number from 1 to $max_job_hours"
  exit 1
fi
if ! [[ "$DATASET" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "dataset must be a plain name under datasets/ in the bucket"
  exit 1
fi

# Spot uses the preemptible quota where one has been granted, and the regular one otherwise.
free=$(gcloud compute regions describe "$REGION" --project "$PROJECT" --format=json |
  jq --arg m "$metric" --arg s "PREEMPTIBLE_$metric" '[.quotas[] | select(.metric == $m or .metric == $s) | .limit - .usage] | max // 0')
global=$(gcloud compute project-info describe --project "$PROJECT" --format=json |
  jq '[.quotas[] | select(.metric == "GPUS_ALL_REGIONS") | .limit - .usage] | max // 0')
echo "gpu quota free: regional=$free global=$global"
if ! jq -en --argjson a "$free" --argjson b "$global" '$a >= 1 and $b >= 1' >/dev/null; then
  echo "not enough GPU quota"
  exit 1
fi

state=$(gcloud scheduler jobs describe thaipoai-reaper --project "$PROJECT" --location "$REGION" --format='value(state)')
if [[ "$state" != ENABLED ]]; then
  echo "reaper schedule is not enabled"
  exit 1
fi
latest=$(gcloud run jobs executions list --job thaipoai-reaper --project "$PROJECT" --region "$REGION" --limit 10 --format=json |
  jq -r 'map(select(.status.completionTime != null)) | sort_by(.status.completionTime) | last // empty | "\(.status.completionTime) \(.status.succeededCount // 0)"')
if [[ -z "$latest" ]]; then
  echo "reaper has never finished a pass"
  exit 1
fi
read -r finished succeeded <<<"$latest"
age=$(( $(date +%s) - $(date -d "$finished" +%s) ))
echo "reaper last finished ${age}s ago, succeeded=$succeeded"
if (( succeeded < 1 || age > 2700 )); then # three missed passes
  echo "reaper is not healthy"
  exit 1
fi

gcloud artifacts docker images describe "$TRAINER_IMAGE:$GITHUB_SHA" --project "$PROJECT" >/dev/null

# The ledger keeps its first start time, so re-running a workflow cannot reset the spend.
now=$(date +%s)
path="gs://$BUCKET/ledger/$RUN_ID.json"
started=$now
if gcloud storage ls "$path" >/dev/null 2>&1; then
  started=$(gcloud storage cat "$path" | jq -r .started_epoch)
fi
worst=$(awk -v r="$rate" -v s="$started" -v n="$now" -v j="$JOB_HOURS" -v m="$reaper_margin_hours" 'BEGIN { printf "%.2f", r * ((n - s) / 3600 + j + m) }')
echo "worst case for this run: $worst of $hard_stop_usd USD"
if awk -v w="$worst" -v l="$hard_stop_usd" 'BEGIN { exit !(w > l) }'; then
  echo "over the hard stop"
  exit 1
fi
jq -n --arg run "$RUN_ID" --arg gpu "$GPU" --argjson started "$started" --argjson rate "$rate" --argjson limit "$hard_stop_usd" \
  '{run_id: $run, gpu: $gpu, started_epoch: $started, rate_usd_per_hour: $rate, limit_usd: $limit}' |
  gcloud storage cp - "$path"

echo "deadline_epoch=$((now + (JOB_HOURS + reaper_margin_hours) * 3600))" >>"$GITHUB_OUTPUT"
