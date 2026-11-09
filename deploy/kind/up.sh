#!/usr/bin/env bash
# Starts the stack on kind. usage: deploy/kind/up.sh fake|mac
# "mac" needs the model server running with --adapter-root runs/kind/adapters.
set -euo pipefail
mode=$1
kind=${KIND:-kind}
state=$PWD/runs/kind
mkdir -p "$state/adapters" "$state/store"
if [ "$mode" = fake ]; then base=base verify=45s; else base=gemma4e4b verify=10m; fi
k() { kubectl --context kind-thaipoai "$@"; }

$kind get clusters | grep -qx thaipoai || sed "s#__STATE__#$state#g" deploy/kind/cluster.yaml | $kind create cluster --name thaipoai --config -
commands="adapter-sync gateway rollout-controller"
[ "$mode" = fake ] && commands="$commands fake-model"
for cmd in $commands; do
  docker build -q --build-arg CMD=$cmd -f deploy/Dockerfile -t thaipoai/$cmd:dev . >/dev/null
  for attempt in 1 2 3; do
    $kind load docker-image --name thaipoai thaipoai/$cmd:dev && break
    sleep 5 # containerd may still be starting on a new node
  done
done

k apply -f deploy/crd.yaml
k wait --for=condition=Established crd/adapterrollouts.thaipoai.dev --timeout=60s
k create configmap suites --from-file=golden.jsonl=suites/golden.jsonl --from-file=replay.jsonl=suites/confirm.jsonl --dry-run=client -o yaml | k apply -f -
sed -e "s#__BASE__#$base#g" -e "s#__VERIFY__#$verify#g" deploy/kind/stack.yaml | k apply -f -
sed "s#__STATE__#$state#g" "deploy/kind/model-$mode.yaml" | k apply -f -
