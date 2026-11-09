# Helpers for driving rollouts on kind. Source this file.
k() { kubectl --context kind-thaipoai "$@"; }

# rollout <name> <sha256> <percent> <samples>
rollout_yaml() {
  cat <<YAML
apiVersion: thaipoai.dev/v1alpha1
kind: ModelRollout
metadata: {name: translate}
spec:
  model: {name: $1, sha256: "$2"}
  canary: {percent: $3, minSamples: $4, timeoutMinutes: 5}
YAML
}
rollout() { rollout_yaml "$@" | k apply -f - >/dev/null; }

field() { k get modelrollout translate -o "jsonpath={$1}" 2>/dev/null; }

# wait_for <seconds> <phase>... returns when the rollout reaches any of the phases.
wait_for() {
  local limit=$1 now
  shift
  for _ in $(seq 1 "$limit"); do
    now=$(field .status.phase)
    for want in "$@"; do [ "$now" = "$want" ] && return 0; done
    sleep 1
  done
  return 1
}

# new_rollout_seen <sha256> waits until the controller has picked up this spec.
new_rollout_seen() {
  for _ in $(seq 1 30); do
    [ "$(field .status.observed)" = "$1" ] && return 0
    sleep 1
  done
  return 1
}
