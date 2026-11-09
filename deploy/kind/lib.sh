# Helpers for driving rollouts on kind. Source this file.
k() { kubectl --context kind-thaipoai "$@"; }
state=$PWD/runs/kind

# pack <directory> <name> stores an adapter archive and prints its sha256.
pack() {
  COPYFILE_DISABLE=1 tar -C "$1" -czf "$state/store/$2.tar.gz" adapter_config.json adapter_model.safetensors
  shasum -a 256 "$state/store/$2.tar.gz" | cut -d' ' -f1
}

# rollout <name> <sha256> <percent> <samples> [timeout minutes] [attempt]
rollout_yaml() {
  cat <<YAML
apiVersion: thaipoai.dev/v1alpha1
kind: AdapterRollout
metadata: {name: translate}
spec:
  adapter: {name: $1, url: "http://adapter-store/$1.tar.gz", sha256: "$2"}
  attempt: ${6:-0}
  canary: {percent: $3, minSamples: $4, timeoutMinutes: ${5:-5}}
YAML
}
rollout() { rollout_yaml "$@" | k apply -f - >/dev/null; }

field() { k get adapterrollout translate -o "jsonpath={$1}" 2>/dev/null; }

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
