# Rollout states and gates

A `ModelRollout` moves one target to a new model. The decisions are pure functions in
`internal/rollout`, and `cmd/rollout-controller` only reads the world and writes the result.

```
Verifying --> Golden --> Shadow --> Canary --> Promoted
    |            |          |          |
    |            |          |          +--> RolledBack   the candidate is bad and had traffic
    +------------+----------+-------------> Blocked      the candidate is bad and had no traffic
    +------------+----------+----------+--> Inconclusive the infrastructure failed
```

| Phase | Passes when | Otherwise |
|---|---|---|
| Verifying | Every running replica answers `GET /v1/models/{id}` for the candidate, whose id carries its sha256 | Not all serving it in time: Inconclusive |
| Golden | At most 1 must-pass case newly fails | Blocked |
| Shadow | Replay corpus shows no significant harm (McNemar, p < 0.01) | Blocked |
| Canary | Enough samples, guard failures and output length within limits | RolledBack, or Inconclusive for outages or too little traffic |

Candidate traffic exists only in Canary. Every other phase serves stable alone.

## Three rules that shape the design

**An outage is never a rollback.** A rollback says "this model is bad". An outage says nothing
about the model, so it ends in `Inconclusive` and traffic returns to stable. The transition
table has no edge from an infrastructure failure to `RolledBack`, and the canary checks
infrastructure before quality. A candidate request that fails falls back to stable, so the user
sees no error, and it counts as an infrastructure error.

**What is deployed is a function of status.** `rollout.Plan` turns the durable status into the
model list and routing. The controller reapplies it on every reconcile. A promotion writes
the phase and `status.stable` in one update. If the controller dies right after, the next
reconcile publishes the routing. There is no half-done state to recover from. The previous
stable stays in the list until the next rollout starts, because replicas and gateways pick up a change
at different moments, and a gateway that is behind must still find the model it asks for.

**The API server enforces the invariants, not a webhook.** One object per target, named after
the target, so uniqueness comes from Kubernetes. A rule in the CRD refuses a new spec until the
controller has observed the current one and finished it. This cannot race, because the check
and the write are one operation on one object. `status` is a subresource, so permission to
edit a rollout does not include permission to type a stable model.

## Gate statistics

**Golden and Shadow are paired.** Both models translate the same cases. Only discordant pairs
count: `worse` where only the candidate failed, `better` where only stable failed. A case both
pass or both fail says nothing about the difference. An overall failure rate would let a large
regression hide inside a large corpus.

Shadow's p-value is the exact one-sided McNemar test: the chance of at least `worse` heads in
`worse + better` fair coin flips. Seven worse and none better gives 1/128 = 0.0078, which
blocks, whatever the size of the corpus.

Golden is a fixed tolerance rather than a test, because a must-pass suite is too small for
statistics. More than one new failure blocks.

**Canary is counts only.** Live messages are never read, stored or sent to a judge. Each gateway
keeps per-model counts of requests, guard failures, infrastructure errors, characters and
time. Live traffic is not paired, so guard failures use the exact one-sided Fisher test at
p < 0.01. Output length rolls back above 1.5 times stable's characters-per-source-character.

Latency is reported but not gated, because both models share one queue. A slow candidate slows
stable too, so the ratio cannot separate them. Output length has no such problem, and it is what
causes the latency anyway.

Neither side is judged until both have `minSamples` requests. Counts start from zero when the
canary begins and are held by the gateways, so a gateway restart loses its share and the canary
simply waits for more. A canary that cannot collect enough before its timeout is `Inconclusive`,
and so is one whose candidate-only error share is above 5% when it would otherwise be promoted.

## Limits

- Guards see format and a narrow set of factual slips. The gates
  catch what guards can see, and they are not a judgment of translation quality. That is the
  frozen benchmark's job, before a rollout begins.
- The gateway reads routing from a mounted ConfigMap, so a change can take up to a minute to
  arrive. A rollback is not instant.
- Deleting a `ModelRollout` ends any canary through a finalizer and returns traffic to stable.
  The published list survives, and a recreated object reads its stable model back from it. Only
  its history is lost.
- One controller replica, with no leader election. It resumes from status after a restart.
