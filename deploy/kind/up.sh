#!/usr/bin/env bash
# Starts the stack on kind. usage: deploy/kind/up.sh fake|mac
# "mac" needs an OpenAI-style model server on the host at port 8766.
set -euo pipefail
mode=$1
kind=${KIND:-kind}
if [ "$mode" = fake ]; then verify=45s; else verify=10m; fi
k() { kubectl --context kind-thaipoai "$@"; }

$kind get clusters | grep -qx thaipoai || $kind create cluster --name thaipoai --config deploy/kind/cluster.yaml
commands="gateway rollout-controller"
[ "$mode" = fake ] && commands="$commands fake-model"
for cmd in $commands; do
  docker build -q --build-arg CMD=$cmd -f deploy/Dockerfile -t thaipoai/$cmd:dev . >/dev/null
  for attempt in 1 2 3; do
    $kind load docker-image --name thaipoai thaipoai/$cmd:dev && break
    sleep 5 # containerd may still be starting on a new node
  done
done

k apply -f deploy/crd.yaml
k wait --for=condition=Established crd/modelrollouts.thaipoai.dev --timeout=60s
k create configmap suites --from-file=golden.jsonl=suites/golden.jsonl --from-file=replay.jsonl=suites/confirm.jsonl --dry-run=client -o yaml | k apply -f -
sed "s#__VERIFY__#$verify#g" deploy/kind/stack.yaml | k apply -f -
k apply -f "deploy/kind/model-$mode.yaml"
