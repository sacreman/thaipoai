# Step 16: adapter-sync on kind

9 November 2026. `cmd/adapter-sync` is a sidecar that sits next to the model server. It reads
a list of desired adapters, downloads each one, verifies its sha256, loads it, confirms it
produces a real translation and publishes the result for that replica.

## The design choice worth reading

vLLM keeps adapters in memory and exposes no hash, so nothing can ask a replica which weights
it holds. The sidecar registers each adapter under a name that contains its hash, such as
`gemma4e4b-demo-bd355a1e0ed9`. The server's own `/v1/models` list then says exactly which
weights are loaded. The sidecar keeps no durable state, so either side can restart and the
next check puts things right.

Routing uses Kubernetes readiness. `/readyz` passes only when the replica holds every adapter
marked `required`, so a Service sends traffic only to replicas that can serve it. A candidate
that is not required can fail without taking the replica out of service, and `/state` reports
it. The rollout controller in step 17 will read that.

## It ran for real

kind cluster, with the real Gemma 4 E4B on the Mac behind vLLM's adapter API, and the adapter
archive served from inside the cluster. The adapter came from the feasibility run and adds a
` #ok` marker to its output, which proves requests really go through it. It says nothing
about translation quality.

| What happened | Result |
|---|---|
| Fresh deploy | Downloaded, verified, loaded and confirmed. Replica ready |
| Request through gateway and Service | 200, translated, marker present |
| Base model asked directly | No marker |
| Unknown adapter name | 404, never the base model |
| Candidate with a wrong checksum | `checksum mismatch`, never loaded. Replica stayed ready, gateway kept serving |
| Candidate removed from the list | Unloaded and its files deleted |
| Model server killed | Replica unready in about 8 s, gateway returned an honest 502 |
| Model server restarted, holding no adapters | Adapter reloaded without a download. Ready 24 s after the restart began, no manual step |
| Sidecar pod deleted | New pod ready in about 3 s, server's adapter untouched |

## Tests

15 Go tests run against a fake server that behaves like vLLM: checksum mismatch, bad candidate,
adapter that translates badly, failed load, restarts of either side, unreachable server, new
weights under an old name, a broken list keeping the last good one, retry delay and unsafe
archives. Archives may hold only flat JSON and safetensors files, so no pickle can reach the
model server.

## What this does not show

- This is not real vLLM. The Mac server copies vLLM's documented adapter API, and step 18 tests
  it against the real thing.
- One replica, because one Mac is one model server. Three-replica faults come in step 17.
- The gateway is pinned to one adapter by a flag. Choosing between stable and candidate for
  each request is step 17.
- A change to the desired list took 48 to 72 seconds to arrive, because Kubernetes refreshes
  mounted ConfigMaps slowly. That is acceptable for a rollout that takes hours.

Approved by the owner on 9 November 2026.
