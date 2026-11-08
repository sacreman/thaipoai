# ThaiPoAI: replacing hosted translation with a model we run ourselves

Version 6.0, 8 November 2026. Personal open-source project by Steven Acreman.
Public repository `sacreman/thaipoai`, model `sacreman/thaipoai` on Hugging Face,
Apache-2.0 for code and weights.

Evidence labels: **[primary]** vendor or legal source, **[estimate]** arithmetic, not
measured.

---

## 1. The story

ThaiPo has been a production English-Thai chat translation service since May 2026. It
sends messages to hosted models (Claude and DeepSeek), together with a glossary, hints
and context that a continuous learning loop mines from each conversation. Every
translation is archived, and Opus 5.5 audits samples for meaning, so we know how good
production really is.

That quality comes at a price: every message leaves our network.

**This project is an experiment: can we stop sending messages out, and replace the
whole system, model and glossary together, with a model we trained and run on our own
hardware?**

The same model is released openly, so any company needing English-Thai translation can
run it in-house and truthfully tell its users that messages stay inside.

### What "good" means

This is a **meaning translator**, not a word-for-word one. Idioms, slang, allegory,
sarcasm and sayings must carry their intended meaning. It matches the sender's tone, and
when the source is genuinely ambiguous it keeps the ambiguity instead of inventing facts.

The score is about meaning.

---

## 2. Success

| Level | Meaning |
|---|---|
| **M1 Trained** | A LoRA adapter beats its untuned base on the dev suite |
| **M2 Not worse** | Our model alone is not meaningfully worse than production (section 8) |
| **M3 Replaced** | It serves real ThaiPo typed English-Thai traffic |

**v1 scope:** typed chat, English to Thai and Thai to English.

**If a benchmark misses the margin:** train again, with a fresh held-out window, until it
passes or the budget is spent. Only then report whatever the final result is.

Fine-tuned means fine-tuned. The base model is Lumen, from Alibaba Token Hub, and the README says so.

---

## 3. System

```
ThaiPo Kappa log (raw events + delivered translations, checksummed per chat-day)
        |  replay
        v
dataset-projector (Go): filters, PII scrub, lineage manifest
        |  + relabelled rows + free public data
        v
LoRA training: 8-bit Lumen on the Mac for rehearsal, 16-bit on a temporary GKE A100 (section 7.1)
        |
        v
Adapter with sha256 manifest  ->  leak test  ->  frozen-candidate benchmark
        |
        v
AdapterRollout CR  ->  rollout-controller (Go)
        |
        v
Client -> gateway (Go, guards) -> vLLM on GKE (Mac model server under kind) with adapter-sync
```

Open weights run on our hardware send nothing to anyone. Lumen's Chinese origin does not
change that, though some procurement teams refuse Chinese-origin models, and the README says
so plainly.

Go for everything that is not training or statistics.

---

## 4. Data

### 4.1 The dataset is a projection of the log

ThaiPo is a Kappa architecture: raw events and delivered translations are stored
separately per chat-day, each with a sha256 manifest. A training set is one more read
model, built by replay.

**All data in ThaiPo's Tigris archive is available for training.** The owner collects
consent for this before the first real-data step. The dataset manifest records the
consent basis and the build refuses to run without it.

Real data work begins only after that. Until then every step uses synthetic fixtures.

Delivered translations record `model`, `prompt_version`, `glossary_version`, `pack`,
`native`, `untranslated` and the source type, so filters are field checks.

### 4.2 Row rules

| Rule | How |
|---|---|
| Typed text only | Not voice, image, video or PDF |
| Literal translation path only | `native` and `untranslated` both false. Native Reply rows are a later experiment |
| English-Thai pair | `chat_lang` empty, source and target are `en` and `th` |
| Not flagged | Absent from the daily quality sweep's flagged rows, and passes Go guards |
| Family-friendly | Sexually explicit rows and adult-pack rows removed. Slang, swearing, flirting and idiom are kept |
| One-to-one and groups | Both, with speaker labels |

### 4.3 Labels

| Row was written by | Use |
|---|---|
| DeepSeek | Delivered output is the label. DeepSeek's terms allow training and distillation [primary] |
| Claude | **Source kept, label replaced.** Rerun through the full production pipeline on DeepSeek. Anthropic's terms bar training competing models on Claude output [primary] |

Labels carry the glossary, hints and six months of tuning, which is how the model absorbs
them. Production output is the teacher, deliberately.

Anything derived from Claude output, including mined chat summaries, is not training
data.

### 4.4 Personal details

Before training, phone numbers, LINE IDs, emails, bank numbers, URLs, addresses and
personal names become typed placeholders, aligned across source and label.

`cmd/extract-audit` is a release gate: fake canary phone numbers and names are planted in
training at known rates, then the trained adapter is probed for them. Any recall over
the threshold blocks publication.

### 4.5 Splits

Newest four weeks and a random 10% of chats are held out, never trained on. A later
benchmark round after retraining uses a newer window.

### 4.6 Size

Start at 20,000 rows, stratified by month and chat type. Grow only when scores improve.

### 4.7 Other sources

Free first. Buy only if a specific gap appears and is shown to matter.

| Source | Licence | Notes |
|---|---|---|
| Common Voice pairs in scb-mt | CC0 | Confirm the published file |
| ALT Parallel Corpus | CC BY 4.0 | CC BY files only |
| TALPCo | CC BY 4.0 | Conversational |
| Tatoeba tha-eng | Per sentence | Credit each author |

**Excluded:** scb-mt share-alike, crawled and Google-translated parts, OpenSubtitles,
FLORES+, TranslateGemma output.

### 4.8 Handling

The dataset lives in a gitignored directory on the Mac and, for cloud runs, in a private
GCS bucket in the training region. Coding agents may read it once real data work has begun.
CI scans every commit and fails on anything resembling private message content. Job logs
never print text, because the workflow logs of a public repository are public. The dataset
itself is never published; the code that builds it is.

---

## 5. Model and training

**Base: Lumen** ([`ATH-MaaS/Qwen-MT-Lumen`](https://huggingface.co/ATH-MaaS/Qwen-MT-Lumen),
revision `3bf800f`), a fine-tuned Qwen3 14B from Alibaba Token Hub. 14.77B parameters, 29.5
GB at 16-bit. It scored 84.0 on English to Thai with human judges at WMT26, the highest of
any open-weight system there. [primary]

**Known gaps, tested first (step 2):** no published Thai-to-English result, chat context
never evaluated, and its Apache-2.0 grant is stated in its README with no LICENSE file. If
Lumen fails either direction or context on the untuned pilot, pause and decide whether to
change base before any paid training.

Its paper also describes some model-generated training data without naming the generator.
The model card says so.

### 5.1 Two environments, one recipe

| | Laptop rehearsal | Real training |
|---|---|---|
| Purpose | Find every bug cheaply | Produce the released model |
| Weights | Lumen 8-bit | Lumen 16-bit (`bfloat16`) |
| Machine | M4 Max, 36 GB | Google A100 80 GB Spot (`a2-ultragpu-1g`), through section 7.1 |
| Stack | MLX, `mlx_lm.lora` | PyTorch, PEFT, CUDA |
| Data | Small chunks: 1,000, 5,000, then 20,000 rows | Full usable dataset |
| Cost | Free, about 1,600 rows an hour (measured) | Target about 20 hours and $60 at the midpoint [estimate] |

**The two stacks are different code.** An MLX rehearsal cannot find a bug in the PEFT cloud
script, so the cloud script gets its own rehearsal: the exact script, on the Mac, against a
tiny Qwen3 model on MPS. Labels, masking, resume, checkpoints and the adapter format are all
exercised before any paid GPU hour.

Shared settings: LoRA rank 64 on the attention projections of all 40 layers, batch 4 to
start. Training order: label and masking checks, overfit one batch, 100-step run for memory
and speed, then longer runs.

Learning rates differ, and each is found by experiment. MLX diverged at 2e-4 and trained at
5e-5, because it applies no gradient clipping. The PEFT script clips gradients at 1.0 and
starts at 2e-4.

**Experiments**, run in small chunks on the laptop first:
- Trained vs untuned Lumen
- 0, 4 and 8 turns of context
- Real ThaiPo data vs public data only, same size
- 8-bit vs 16-bit quality on the same adapter recipe

Dev decisions use the dev suite and Jev. Opus is not used to choose checkpoints.

### 5.2 What gets released

Three artefacts on Hugging Face, all from one training run:

| Repository | Contents | Size | For |
|---|---|---|---|
| `sacreman/thaipoai` | Adapter merged into 16-bit weights | About 30 GB | In-house use on a GPU with 40 GB or more, with vLLM |
| `sacreman/thaipoai-8bit-mlx` | The merged model converted to 8-bit MLX | About 15 GB | Macs |
| `sacreman/thaipoai-adapter` | PEFT LoRA adapter alone | About 0.3 GB | People building on base Lumen |

The 8-bit build is converted from the merged 16-bit model, so it contains the training. Every
artefact carries a sha256 manifest, the training run ID and the dataset manifest hash.

**Before upload:** `extract-audit` passes, the model card credits Lumen and its base Qwen3,
keeps Apache-2.0 notices and marks the weights as modified, and the Thai and English
benchmark results are recorded. Published weights cannot be withdrawn once downloaded.

---

## 6. Replacing the glossary

**End state: no glossary at all.** The model alone must carry the shared terms and hints
it learned from production. Names and terms that only exist in one chat are the expected
weak spot, and the benchmark reports them separately.

API:

```
POST /v1/translate
source_lang, target_lang, text, context[] (ordered, with speaker),
speaker{gender?}, addressee{gender?}
```

Gender is optional. When absent, the model infers it from the conversation and stays
neutral if unsure. Training data derives gender from each speaker's own particles.

---

## 7. Runtime and Kubernetes

Two environments, one codebase.

- **kind on the laptop** is for development. Real translations come from a model server on
  the Mac host behind an OpenAI-style API, which pods call. A fake backend gives
  deterministic multi-replica behaviour in CI.
- **A temporary GKE GPU pool** (section 7.1) is for real training, real vLLM serving and
  the demos on real GPUs. It exists only while it is being used.

**`cmd/gateway`.** Implements section 6. Guards: gender agreement, numbers and times,
negation, source-script leak, commentary and refusal, markup. One repair. First-try and
post-repair quality are reported separately. Never logs text.

**`cmd/adapter-sync`.** vLLM exposes no adapter hash and keeps its registry in memory.
[primary] This sidecar syncs desired adapters, verifies sha256, confirms the adapter
serves a test request and publishes per-replica state. The gateway routes only to ready
replicas.

**`cmd/rollout-controller`.** `AdapterRollout` CR: verify, golden suite, paired shadow,
canary, then promoted or rolled back.

- One active rollout per target, enforced at admission. `spec` is immutable.
- `status.stable` is a durable pointer, never hand-typed.
- Infrastructure faults give `Inconclusive`, never rollback.
- Quality gates on discordant pairs (candidate fails where stable passes), exact McNemar.
- Per-adapter gateway metrics, with minimum sample counts.
- If the controller dies, the gateway holds its last weights.
- State transitions and gate statistics are in `docs/rollout-states.md`.

**`cmd/suite`.** Runs a contrastive suite against any endpoint. **`cmd/dataset-projector`**
and **`cmd/extract-audit`** are in section 4.

**Fault matrix, in CI on kind:** controller killed mid-canary, gateway restart, replica
lost mid-canary, adapter load fails on one replica of three, checksum mismatch, two
rollouts on one target.

**Demos:** an adapter trained with flipped gender particles blocked at the golden gate,
and an output-bloat adapter rolled back at canary.

**vLLM on real GPUs.** Serve the model on vLLM in the temporary pool and publish load
numbers: p95, time to first token, and tokens per dollar. At 16-bit it needs a GPU with at
least 40 GB [estimate], so an A100 40 GB is the smallest Google option. vLLM is the production serving
standard and needs an NVIDIA GPU.

### 7.1 Temporary GPU pipeline

A GitHub Actions run creates a GPU pool, trains, evaluates, uploads and destroys everything.
Plain "rent a GPU from CI" is well trodden (ec2-github-runner, CML, SkyPilot), so the claim
is the guarantees below, not the idea.

**Who can start it.** `workflow_dispatch` only, through a protected `gpu` environment that
needs a required reviewer. Pull requests get no GPU credentials.

**No runner on the GPU.** A GitHub-hosted runner submits the Job and polls it, because
self-hosted runners on public repositories are unsafe. [primary] Polling is resumable, to
stay under the 6-hour hosted job limit.

**Identity.** OIDC to Google Workload Identity Federation, conditioned on repository, ref
and environment. No long-lived keys.

**Flow:**
1. Preflight: GPU quota, run ledger under its hard spending limit, reaper healthy.
2. Terraform creates a zonal GKE cluster with a CPU system pool and a Spot GPU pool scaled
   from zero: A100 80 GB (`a2-ultragpu-1g`) for 16-bit Lumen, and L4 for cheap rehearsals with
   the small stand-in model. Everything is labelled with the run ID.
3. Kueue admits a Kubernetes Job with `activeDeadlineSeconds`.
4. Training writes atomic checkpoints with a completion marker at least every 30 minutes,
   and resumes after preemption. GKE gives only 15 seconds' notice by default. [primary] A
   full run is expected to be interrupted at least once.
5. The eval gate runs. Only passing adapters are uploaded, with git SHA, dataset hash,
   metrics and cost.
6. Teardown runs on `if: always()`, idempotently, finding resources by label.

**Termination is not trusted to CI.** `cmd/reaper`, run on Cloud Scheduler outside the
workflow, deletes any labelled resource past its deadline. Provider-side maximum runtime is
set where available. A workflow that dies cannot leave a GPU running.

**Budgets.** A billing alert does not cap spending and it lags, so the run ledger is the
hard stop and the alert is only a backstop. The full training run is planned at about 20
hours and about $60 at the midpoint [estimate], with a hard stop of $150 per run. The Job
deadline is 50 hours and the reaper's deadline 56 hours.

**Published drills:** kill a run mid-training and show the reaper clearing it, and a Spot
reclaim with no lost work. Report cost per passing adapter.

**Position against existing projects.** llm-d, AIBrix, NVIDIA Dynamo, Kthena, OME, KServe,
Argo Rollouts, Flagger and Seldon each cover parts of this. What we did not find is
eval-gated promotion on paired outputs with per-replica adapter truth. Claim that and
nothing wider.

---

## 8. Benchmark

**Competitors:**
1. Our model alone, at 16-bit
2. **Production: exactly what ThaiPo delivered, from the archive**
3. Opus 5.5 translating fresh, as a quality ceiling

Opus scores its own translations in slot 3, so that slot is labelled self-marked and is
reference only. The headline is 1 against 2, where the judge wrote neither.

The 8-bit copy is benchmarked separately, so the quality cost of compression is on record.
The pass below is judged on the 16-bit model, because that is what is released.

**Scoring** (blind, shuffled):
- **Opus 5.5:** 1-5 on meaning with a defect class (actor, polarity, question, number,
  gender, invented, dropped, typo, language, commentary, leak, other), plus which of two
  better conveys the meaning.
- **Jev:** independent yes/no on "do these mean the same?". Disagreements with Opus are
  published.

**Opus only ever scores frozen candidates.** Its scores never feed training data, in
line with Anthropic's terms. [primary]

**Suites:**
- **Real-chat benchmark:** 2,000 held-out lines per direction.
- **Meaning suite:** 300 cases of idiom, slang, sarcasm, euphemism and sayings in both
  directions, each with intended meaning and a literal trap that must not appear. Drafted
  by DeepSeek, reviewed by Opus 5.5, used on frozen candidates only.
- **Dev suite:** about 200 cases drafted by DeepSeek, for training decisions only.

**Pass (M2), per direction:**
- Misleading rate (score 2 or lower) no more than **1.5 percentage points** above
  production, by one-sided 95% upper bound on the paired difference.
- A/B preference not clearly behind: lower bound of preference share over 0.40.

Thresholds are fixed before the sealed benchmark is opened. [estimate] 2,000 per direction
resolves a 1.5-point margin at the error rates seen so far; recompute from the first run.

**Limitation stated in the README:** all judging is by AI. No human has validated the
scorers, and Opus's blind spots would be invisible here.

---

## 9. Build process

One small piece at a time. A step is done when it **runs for real, has a test that fails
if it breaks, has a short write-up of what we learned, and the owner approves.**

The rule for the GPU bill: **nothing runs on the A100 until the same code has already run
on the laptop.**

**Phase 1: Laptop, free**

1. Prove the models train on the Mac. Done, including Lumen at 8-bit. Repo goes public after this.
2. Suites and scorers on untuned 8-bit Lumen, covering both directions and chat context. Go or no-go on Lumen.
3. `dataset-projector` on fixtures, then a count-only inventory of the archive.
4. Consent basis recorded. Relabelling, filters, PII scrub, hold-out.
5. Score production to set the baseline.
6. 8-bit rehearsals in growing chunks: 1,000, 5,000, then 20,000 rows, each scored and leak tested. Fix what breaks.
7. Experiments from section 5, at 8-bit.
8. PEFT cloud script rehearsed on the Mac against a tiny Qwen3: training, interruption, resume, adapter reload, merge and 8-bit conversion.

**Phase 2: Cloud training**

9. Terraform, Actions pipeline and `cmd/reaper`, run with the tiny model on a cheap L4 Spot node. Request GPU quota well before this step.
10. Leak and Spot drills on that cheap node.
11. A 100-step timing run on the A100 80 GB, under $1. Confirms memory and real speed, and reprices the full run.
12. A 20,000-row run on the A100, about $6 [estimate], end to end.
13. The full training run, target about 20 hours, then the leak test.
14. Frozen benchmark on the 16-bit model. Retrain until it passes or the budget runs out.

**Phase 3: Platform**

15. Gateway with guards.
16. `adapter-sync` on kind with the Mac model server.
17. `rollout-controller`, fault matrix, both demos.
18. vLLM on real GPUs: load numbers and both demos.

**Phase 4: Launch**

19. Merge, build the three Hugging Face artefacts, verify each in a clean environment.
20. README, recorded demo, then one launch across Kubedex and LinkedIn.

Nothing is announced before step 20.

---

## 10. README requirements

The README must explain, in this order:

1. ThaiPo: what it is, how long it has run, how hosted models, glossary and the learning loop work.
2. Why we want to stop sending messages out.
3. What meaning translation is, with examples.
4. How production quality is measured: Opus 5.5 rubric, aggregates and dates, never a message.
5. How the model was made: base model, data sources, what was filtered out.
6. Benchmark results, AI-only judging limitation stated.
7. How to run it in-house.
8. Link to the model on Hugging Face.

---

## 11. Public claims

| Safe | Not safe |
|---|---|
| "Fine-tuned from Lumen, itself fine-tuned from Qwen3" | "Built from scratch" |
| "Trained on ThaiPo conversations used with consent, plus the public data in this manifest" | Describing individual conversations |
| "Within X points of production on held-out chats, this date" | "Better than DeepSeek" unscoped |
| "Runs in your cluster, sends nothing out" | "Your data never reached China", about any row DeepSeek labelled |
| "A continuous learning loop mines glossaries" | "Reinforcement learning" |
| "In production since May 2026" | "Almost a year" |
| "Judged by Opus 5.5 and Jev" | "Human-validated" |
| "Built with AI coding agents, designed and reviewed by me" | Implying it was hand-typed |

**Out of scope for this model:** impersonation, romance or recruitment scams, safety,
legal or medical translation without human review.

---

## 12. Layout

```
cmd/{gateway,rollout-controller,adapter-sync,suite,dataset-projector,extract-audit,reaper}
api/  internal/guards/  internal/stats/
.github/workflows/  infra/terraform/
train/  serve/  chart/  data/manifests/  suites/
docs/{rollout-states.md,steps/,interview/}  reports/
```

## Sources

[DeepSeek terms](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html), [Anthropic terms](https://www.anthropic.com/legal/commercial-terms), [Lumen](https://huggingface.co/ATH-MaaS/Qwen-MT-Lumen), [Lumen paper](https://arxiv.org/abs/2610.19758), [WMT26 Findings](https://www2.statmt.org/wmt26/pdf/2026.wmt-1.48.pdf), [ALT](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/), [TALPCo](https://github.com/matbahasa/TALPCo), [Tatoeba](https://tatoeba.org/en/terms_of_use), [scb-mt](https://github.com/vistec-AI/dataset-releases/releases), [vLLM LoRA](https://docs.vllm.ai/en/stable/features/lora/), [vLLM metrics](https://docs.vllm.ai/en/stable/design/metrics/), [llm-d adapter rollout](https://llm-d.ai/docs/operations/rollouts/adapter-rollout), [Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension).
