# Step 17: rollout controller, fault matrix and both demos

9 November 2026. An `AdapterRollout` moves one target to a new adapter through Verifying, Golden, Shadow and
Canary, and ends Promoted, Blocked, RolledBack or Inconclusive. The states, gates and statistics
are in [`docs/rollout-states.md`](../rollout-states.md).

## The design choice worth reading

A rollback has to mean "this adapter is bad". If an outage can cause one, the signal is
worthless, so infrastructure failures end in `Inconclusive` and the transition table has no edge
from one to `RolledBack`. The canary checks infrastructure before it checks quality.

The controller holds no state of its own. `rollout.Plan` turns the durable status into the
adapter list and the routing, and every reconcile reapplies the result. A promotion is one status
update, so a controller killed at any moment resumes with nothing to repair.

The API server enforces the rule against two rollouts on one target. There is one object per
target, and a CEL rule on the CRD refuses a new spec until the current one has finished.

## Fault matrix

`deploy/kind/fault-matrix.sh` on kind, with three replicas of a deterministic fake model that
implements the same adapter API. 19 checks passed and none failed.

| Fault | Result |
|---|---|
| Second rollout, or an edit to a running canary | Refused by the API server |
| Controller killed mid-canary | Canary kept serving, then Promoted |
| Promotion under traffic | 0 user errors |
| Gateway restarted mid-canary | Promoted |
| Replica lost mid-canary | Promoted, never a rollback |
| Adapter load fails on one replica of three | Inconclusive, stable unchanged, 3 replicas serving |
| Checksum mismatch | Blocked |
| Adapter that swaps gendered particles | Blocked at Golden, 30 regressions with 1 allowed, never reached Shadow |
| Adapter that triples its output | RolledBack at Canary, 3.15 times stable's length, traffic back on stable |

## Real model

Gemma 4 E4B on the Mac behind the adapter API, one replica, with adapters trained for the demo.
The first runs found two things the fake model could not.

| Adapter | First result | What it meant |
|---|---|---|
| Swapped particles | Inconclusive, despite 25 regressions | A real bug. The sidecar's model listing queued behind slow translations, so the replica went unready for 6 s and 4 of 36 cases errored. The server now answers listings while translating |
| Tripled output | Blocked at Golden, 3 regressions, length ratio 2.32 | A correct catch. It was trained without gender notes and hedged the particles, so it failed for a reason nobody planned. It was retrained so that length is its only fault |

After the fix and the retrain, both adapters ended where they should:

| Adapter | Golden (36 cases) | Shadow (120 cases) | Canary | Result |
|---|---|---|---|---|
| Swapped particles | 29 worse, 0 errors | Not reached | Not reached | **Blocked** in 95 s, never had traffic |
| Output bloat | 0 worse, 0 errors | 0 worse, 6 better | 1.96 times stable's output per source character, 40 candidate and 53 stable requests, 0 errors | **RolledBack** after 642 s, traffic back on stable |

The bloat adapter is the better demonstration. Its translations are good enough to pass both
offline gates, and only live output length shows the fault, which is why the canary exists.

## Independent review

A reviewer who had not written the code found 11 defects and 4 test gaps, all now fixed with
tests. The ones that mattered:

- A promotion unloaded the old stable adapter while gateways still asked for it, causing 502s.
  The previous stable now stays loaded until the next rollout.
- A hung candidate used the whole timeout and left none for the fallback to stable.
- Infrastructure errors under 20% could still be promoted. Above 5% is now Inconclusive.
- Deleting a rollout mid-canary left traffic on the candidate. A finalizer now returns it.
- A checksum mismatch on one replica blocked the adapter. It now needs every replica to agree.
- Stable samples from before the canary counted in the comparison.

## Tests

70 Go tests across `internal/rollout`, `cmd/rollout-controller`, `cmd/gateway` and
`cmd/adapter-sync`, and the 36-case golden suite. `.github/workflows/fault-matrix.yml` runs the
matrix in CI, but it has not run yet because nothing has been pushed.

## What this does not show

- This is not real vLLM. Step 18 runs it on real GPUs.
- The three-replica faults used the fake model. One Mac can be only one real model replica.
- The gates catch what guards and the golden suite can see. They do not judge translation
  quality, which is the frozen benchmark's job before a rollout begins.
- A rollback is not instant. Routing reaches the gateway through a mounted ConfigMap, which can
  take up to a minute.
- One controller replica, with no leader election. It resumes from status after a restart.
