#!/usr/bin/env bash
# Deletes everything labelled with this run's ID, finding it by label rather than from
# Terraform state, so it works after a half-finished apply. Safe to run twice.
set -euo pipefail

filter="labels.thaipoai-run-id=$RUN_ID"

clusters() {
  gcloud container clusters list --project "$PROJECT" \
    --filter="resourceLabels.thaipoai-run-id=$RUN_ID" --format='csv[no-heading](name,location,status)'
}
instances() {
  gcloud compute instances list --project "$PROJECT" --filter="$filter" --format='csv[no-heading](name,zone.basename())'
}
disks() {
  gcloud compute disks list --project "$PROJECT" --filter="$filter" --format='csv[no-heading](name,zone.basename(),region.basename())'
}

listed=$(clusters)
for row in $listed; do
  IFS=, read -r name location status <<<"$row"
  if [[ "$status" != STOPPING ]]; then
    gcloud container clusters delete "$name" --project "$PROJECT" --location "$location" --quiet
  fi
done

# A delete started by the reaper or an earlier attempt may still be finishing.
for _ in $(seq 60); do
  listed=$(clusters)
  if [[ -z "$listed" ]]; then
    break
  fi
  sleep 30
done

listed=$(instances)
for row in $listed; do
  IFS=, read -r name zone <<<"$row"
  gcloud compute instances delete "$name" --project "$PROJECT" --zone "$zone" --quiet
done

listed=$(disks)
for row in $listed; do
  IFS=, read -r name zone region <<<"$row"
  if [[ -n "$zone" ]]; then
    gcloud compute disks delete "$name" --project "$PROJECT" --zone "$zone" --quiet
  else
    gcloud compute disks delete "$name" --project "$PROJECT" --region "$region" --quiet
  fi
done

left=$(clusters)
left+=$(instances)
left+=$(disks)
if [[ -n "$left" ]]; then
  echo "resources remain for this run; the reaper will retry them"
  exit 1
fi
echo "nothing left for this run"

path="gs://$BUCKET/ledger/$RUN_ID.json"
if gcloud storage ls "$path" >/dev/null 2>&1; then
  gcloud storage cat "$path" |
    jq --argjson ended "$(date +%s)" '. + {ended_epoch: $ended, spent_usd: (((.rate_usd_per_hour * ($ended - .started_epoch) / 3600) * 100 | round) / 100)}' |
    gcloud storage cp - "$path"
fi
gcloud storage rm --recursive "gs://$BUCKET/tfstate/$RUN_ID" 2>/dev/null || true
