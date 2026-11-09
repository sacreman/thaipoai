#!/usr/bin/env bash
# The fault matrix and both demos on kind, with three deterministic replicas. Exits non-zero if
# any outcome is not the one the design promises.
set -uo pipefail
source deploy/kind/lib.sh
failures=0

check() { # check <what> <expected> <actual>
  if [ "$2" = "$3" ]; then echo "PASS  $1: $3"; else echo "FAIL  $1: wanted '$2', got '$3' ($(field .status.reason))"; failures=$((failures + 1)); fi
}

digest() { printf '%s' "$1" | shasum -a 256 | cut -d' ' -f1; } # a fake model's identity

traffic() { k scale deployment/traffic --replicas="$1" >/dev/null; }
routing() { k get configmap rollout-translate -o 'jsonpath={.data.routing\.json}'; }
die() { echo "FAIL  $1"; exit 1; }
start() {
  sha=$(digest "$1")
  rollout "$1" "$sha" 50 150 || die "rollout $1 was refused"
  new_rollout_seen "$sha" || die "controller never picked up $1"
}
finish() { wait_for 180 Promoted Blocked RolledBack Inconclusive || die "rollout never finished"; field .status.phase; }
in_canary() { # waits until the candidate is really receiving traffic, so a fault lands mid-canary
  wait_for 120 Canary || die "never reached canary"
  for _ in $(seq 1 90); do
    [ "$(field .status.canary.candidate.requests)" -gt 0 ] 2>/dev/null && return 0
    sleep 1
  done
  die "candidate never received traffic"
}
gateway_errors() { k logs deploy/gateway --tail=-1 | grep -c 'status=error'; }
gateway_candidate_requests() { # asks a running gateway directly, not the controller
  k port-forward deploy/gateway 18088:8088 >/dev/null 2>&1 &
  local forward=$!
  sleep 2
  curl -s -m 5 localhost:18088/stats | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['models'].get('$1',{}).get('requests',0))"
  kill $forward 2>/dev/null
  wait $forward 2>/dev/null
}
ready_replicas() { k get deployment model -o 'jsonpath={.status.readyReplicas}'; }

k rollout status deployment/rollout-controller --timeout=180s >/dev/null
k rollout status deployment/gateway --timeout=180s >/dev/null
k rollout status deployment/model --timeout=180s >/dev/null
traffic 1

echo "--- first rollout, from the base model"
start good1
check "first rollout" Promoted "$(finish)"
check "stable pointer moved" good1 "$(field .status.stable.name)"

echo "--- two rollouts on one target, and controller killed mid-canary"
start good2
second=$(digest good3)
if rollout good3 "$second" 50 30 2>/tmp/thaipoai-second.$$; then
  check "second rollout refused by the API server" refused accepted
else
  check "second rollout refused by the API server" "in progress" "$(grep -o 'in progress' /tmp/thaipoai-second.$$ | head -1)"
fi
rm -f /tmp/thaipoai-second.$$
in_canary
if rollout good2 "$sha" 25 150 2>/tmp/thaipoai-edit.$$; then
  check "editing a running canary refused" refused accepted
else
  check "editing a running canary refused" "in progress" "$(grep -o 'in progress' /tmp/thaipoai-edit.$$ | head -1)"
fi
rm -f /tmp/thaipoai-edit.$$
errors_before=$(gateway_errors)
k scale deployment/rollout-controller --replicas=0 >/dev/null
k delete pod -l app=rollout-controller --grace-period=0 --force >/dev/null 2>&1
k wait --for=delete pod -l app=rollout-controller --timeout=60s >/dev/null 2>&1
candidate=good2-${sha:0:12}
count_before=$(gateway_candidate_requests "$candidate")
sleep 4
count_after=$(gateway_candidate_requests "$candidate")
if [ "$count_after" -gt "$count_before" ]; then check "canary kept serving with the controller gone" yes yes; else check "canary kept serving with the controller gone" yes "no: $count_before then $count_after"; fi
k scale deployment/rollout-controller --replicas=1 >/dev/null
check "controller killed mid-canary" Promoted "$(finish)"
sleep 12
check "no user errors across a promotion" "$errors_before" "$(gateway_errors)"

echo "--- gateway restart mid-canary"
start good3
in_canary
k rollout restart deployment/gateway >/dev/null
check "gateway restart" Promoted "$(finish)"

echo "--- replica lost mid-canary"
start good4
in_canary
k delete pod "$(k get pods -l app=model -o 'jsonpath={.items[0].metadata.name}')" --wait=false >/dev/null
outcome=$(finish)
check "replica lost is never a rollback" Promoted "$outcome"
k rollout status deployment/model --timeout=180s >/dev/null

echo "--- candidate missing on one replica of three"
victim=$(k get pods -l app=model -o 'jsonpath={.items[0].metadata.name}')
k port-forward "pod/$victim" 18000:8000 >/dev/null 2>&1 &
forward=$!
sleep 2
curl -s -m 5 -X POST 'localhost:18000/fault?hide=good5' >/dev/null
stable_before=$(field .status.stable.name)
start good5
check "candidate missing on one replica" Inconclusive "$(finish)"
check "stable unchanged" "$stable_before" "$(field .status.stable.name)"
sleep 2
check "all replicas still serving" 3 "$(ready_replicas)"
curl -s -m 5 -X POST 'localhost:18000/fault?hide=' >/dev/null
kill $forward 2>/dev/null

echo "--- digest no replica holds"
start corrupt
check "digest no replica holds" Inconclusive "$(finish)"
check "never reached golden" "" "$(field .status.golden.cases)"
case "$(routing)" in *candidate*) check "no traffic to it" none present ;; *) check "no traffic to it" none none ;; esac
check "stable unchanged" "$stable_before" "$(field .status.stable.name)"

echo "--- demo: flipped gender particles"
start flipped
check "flipped model" Blocked "$(finish)"
echo "      $(field .status.reason)"
check "never reached shadow" "" "$(field .status.shadow.cases)"

echo "--- demo: output bloat"
start bloat
check "bloat model" RolledBack "$(finish)"
echo "      $(field .status.reason)"
sleep 2
case "$(routing)" in *candidate*) check "traffic back on stable" none present ;; *) check "traffic back on stable" none none ;; esac
check "stable unchanged" "$stable_before" "$(field .status.stable.name)"

traffic 0
echo "--- history"
k get modelrollout translate
[ "$failures" -eq 0 ] && echo "fault matrix passed" || echo "fault matrix FAILED: $failures"
exit "$failures"
