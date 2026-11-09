#!/usr/bin/env bash
# Starts the stack on kind with the fake model. usage: deploy/kind/up.sh
set -euo pipefail
kind=${KIND:-kind}
k() { kubectl --context kind-thaipoai "$@"; }

$kind get clusters | grep -qx thaipoai || $kind create cluster --name thaipoai --config deploy/kind/cluster.yaml
for cmd in gateway rollout-controller fake-model; do
  docker build -q --build-arg CMD=$cmd -f deploy/Dockerfile -t thaipoai/$cmd:dev . >/dev/null
  for _ in 1 2 3; do
    $kind load docker-image --name thaipoai thaipoai/$cmd:dev && break
    sleep 5 # containerd may still be starting on a new node
  done
done

k apply -f deploy/crd.yaml
k wait --for=condition=Established crd/modelrollouts.thaipoai.dev --timeout=60s
k create configmap suites --from-file=golden.jsonl=suites/golden.jsonl --from-file=replay.jsonl=suites/confirm.jsonl --dry-run=client -o yaml | k apply -f -
k apply -f deploy/kind/stack.yaml
k apply -f deploy/kind/model-fake.yaml
