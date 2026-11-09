# Step 9: cloud pipeline

9 November 2026. The code for the temporary GPU pipeline in section 7.1 of the spec is written.
**None of it has run against a cloud, so this step is not done.** A step is done when it runs
for real.

## What exists

| Piece | Where | What it does |
|---|---|---|
| Reaper | `cmd/reaper` | One-shot pass. Deletes labelled GKE clusters, VMs and disks past their deadline |
| Long-lived infra | `infra/base` | Reaper on Cloud Run and Cloud Scheduler, GitHub OIDC, network, bucket, service accounts |
| Per-run infra | `infra/run` | Zonal cluster, CPU system pool, Spot GPU pool from zero. L4 by default, A100 by option |
| Workflow | `.github/workflows/train.yml` | Preflight, provision, poll, eval gate, upload, teardown |
| Job and queue | `deploy/gke/job.yaml`, `queue.yaml` | Kueue admits the training Job, with `activeDeadlineSeconds` |
| Helpers | `deploy/gke/*.sh` | Preflight, one poll leg, teardown by label |

Decisions worth reading:

- **Labels carry the guarantee.** A resource needs `thaipoai-run-id` and `thaipoai-deadline`
  (Unix seconds) to be reaped. With the run label and an unreadable deadline it is never deleted
  and the pass fails, so preflight notices.
- **The reaper cannot be removed by a run.** It sits in a separate Terraform root and state, and
  the workflow's identity only has viewer rights on it.
- **A run's VMs are left to their cluster while it exists**, because nodes in a managed group
  are replaced if deleted. A later pass clears anything the cluster left behind.
- **Two GitHub environments, not one.** Each job naming an environment with a required reviewer
  stops for approval, so a single `gpu` environment would make teardown wait for a person. `gpu`
  is the reviewed start and `gpu-run` carries the later jobs. OIDC also requires this exact
  workflow file on `main`, and later jobs only run after the reviewed one.
- **Polling holds no state.** Four sequential legs of 5.5 hours watch the Job in the cluster, so
  the Job deadline is capped at 20 hours.
- **The ledger is time multiplied by a ceiling rate**, not billing data. It can only overstate.
  Preflight refuses a run whose worst case, including the six-hour reaper margin, is over $150.
- **Teardown finds resources by label**, not from Terraform state, so it works after a
  half-finished apply.
- **"Upload" means the `adapters/` prefix in our private bucket.** Published weights cannot be
  withdrawn, so Hugging Face stays a deliberate step at step 19.

## What has been checked

- 17 Go tests against an httptest fake of the metadata server and both APIs: unlabelled resources
  untouched, unreadable deadlines, already gone, second pass, paging, stopping clusters, nodes
  left to their cluster, attached disks, and failed or incomplete listings.
- `terraform fmt` and `terraform validate` pass on both roots.
- `actionlint` passes on the workflow. The YAML parses, and the manifests render.
- `poll.sh` and `teardown.sh` ran once against stub `gcloud` and `kubectl` scripts, covering
  complete, failed, unreachable and hard stop, and a repeat teardown. These stubs are not in the repo.

## What has NOT been run

Nothing below has happened, and each is a real unknown:

- No `terraform plan` or `apply`, no workflow run, no GCP or GitHub Actions call of any kind.
- The reaper has never met the real APIs or a real metadata server. The fakes are built from the
  documented response shapes.
- IAM roles are worked out from documentation and may be short of a permission.
- `max_run_duration` on a Spot GKE node pool is accepted by the provider schema. Google may still
  reject it. If so, remove that one line.
- Quota metric names, the `gcloud` output `preflight.sh` parses, and the Kueue, GPU node and
  autoscaler behaviour are all untested.
- `kubectl` has not validated the manifests against a cluster.

## What the trainer still needs

`deploy/gke/job.yaml` runs `python -m train.cloud_job`, which does not exist. `train/overnight.py`
has the right checkpoint and resume logic, but it cannot run in this Job yet:

1. `train/feasibility.py` hard-codes `DEVICE = "mps"` and calls `torch.mps` directly. It needs CUDA.
2. Checkpoints are local. They need to reach `$OUTPUT_URI/checkpoints` at least every 30 minutes
   with a completion marker, because the pod's disk is lost on a Spot reclaim.
3. On start, resume from the newest complete checkpoint in the bucket.
4. Read the dataset from `$DATASET_URI`. Scratch space is at `$SCRATCH_DIR`.
5. After training, run the eval and write the adapter to `$OUTPUT_URI/adapter/`, then
   `$OUTPUT_URI/eval.json` containing `run_id`, `passed`, `dataset_hash` and `metrics`.
6. If restarted after finishing training, go straight to eval.
7. A trainer image pushed as `$TRAINER_IMAGE:<git sha>`. There is no Dockerfile for it yet.

Never print message text. Workflow logs on this repository are public.

## Before the first run

1. Add `.terraform/` and `*.tfstate*` to `.gitignore`.
2. Request GPU quota, including the global `GPUS_ALL_REGIONS`.
3. Apply `infra/base`: first `-target=google_artifact_registry_repository.images`, then push the
   reaper image built from `deploy/Dockerfile` with `CMD=reaper`, then the full apply.
4. In GitHub, create environments `gpu` (required reviewer) and `gpu-run` (no reviewer), both
   limited to `main`.
5. Set secrets `GCP_PROJECT_ID`, `GCP_BUCKET`, `TRAINER_IMAGE`, `GCP_WORKLOAD_IDENTITY_PROVIDER`
   and `GCP_DEPLOYER_SERVICE_ACCOUNT`, and variables `GCP_REGION` and `GCP_ZONE`. The project ID
   is a secret only so that it is masked in public logs.
6. Check the two rate ceilings in `deploy/gke/preflight.sh` against current prices.
