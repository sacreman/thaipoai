# ThaiPoAI: a Thai-English translator trained from scratch

Version 8.3, 12 November 2026. Personal open-source project by Steven Acreman.
Public repository `sacreman/thaipoai`. Code is Apache-2.0. The weights' licence follows the
training data (section 4.4).

Earlier work downloaded pretrained models and fine-tuned them. It did not look promising and has
been removed. `docs/pretrained-experiments.md` records what was done and what it showed.

Evidence labels: **[measured]** run on our hardware, **[estimate]** arithmetic or a sizing run,
**[inferred]** read from documentation.

---

## 1. The experiment

ThaiPo is a production English-Thai chat translator. It uses a hosted DeepSeek pipeline with a
glossary, scoring and corrections.

**What can a model of under 300 million parameters, trained from random initialisation on an M4 Max
laptop, achieve against that system, and where does it fall short?**

This is an experiment in learning and in practical quality. We do not assume it will beat DeepSeek.
A clear measurement of the gap counts as success.

### The outcome aimed for

Restated on 11 November 2026: the best translator for informal one-to-one chat between Thai and
English in a model under 300 million parameters. The proof is Thai speakers preferring its output,
blind, to DeepSeek's, Typhoon Translate's and Google's on chat messages none of them has seen
(section 5). FLORES-200 is a secondary line: match or beat Typhoon Translate 1.5.

On general text a model trained on DeepSeek's translations is expected to end below DeepSeek. A
review on 11 November estimated 46 to 55 chrF on FLORES-200 for such a model, against the teacher's
59 and 64. The wins looked for are per size and on chat.

Measured so far, chrF on FLORES-200 devtest with the project's own script, English to Thai and Thai to
English: DeepSeek Flash 59.08 and 63.81; Typhoon Translate 1.5 (4B) 54.22 and 52.46; this project's
75M model on the human-translated data about 26 and 30 [measured].

### What "good" means

This is a **meaning translator**, not a word-for-word one. Idioms, slang, sarcasm and sayings must
carry their intended meaning. It matches the sender's tone, and when the source is ambiguous it
keeps the ambiguity instead of inventing facts.

---

## 2. Deliverables

1. A reproducible MLX training pipeline.
2. Documented dataset selection (`docs/data-sources.md`, `docs/distillation-plan.md`,
   `docs/distillation-sources.md`).
3. A tokenizer.
4. Model checkpoints.
5. An honest comparison with the existing system, per direction.

---

## 3. Model

One bilingual encoder-decoder Transformer in core MLX, not `mlx-lm`. Target-language markers select
the direction. Every parameter is trainable from random initialisation, embeddings and output
weights included.

| Config | Encoder/decoder layers | d_model | Heads | Feed-forward | Parameters |
|---|---|---|---|---|---|
| **full** | 9/9 | 1024 | 16 | 4096 | 297,197,568 |
| baseline | 8/8 | 512 | 8 | 2048 | 75,188,224 |
| probe | 6/6 | 256 | 4 | 1024 | about 19 million |

Counts are from instantiated models at a vocabulary of 32,000 [measured]. All share one matrix for
the source embedding, target embedding and output projection, and that matrix is fully trained. The
probe is for quick comparisons only.

**Size risk.** The first corpus is about 300,000 pairs, far below what a model of this size usually
needs, so the baseline trains on the same data for comparison. On that corpus the full model was
behind the baseline when its run was stopped a third of the way through
(`docs/steps/05-first-training-runs.md`).

**Implementation.**
- Hand-written layers. `nn.Transformer` has no decode cache, embeddings or positions.
- Fixed batch buckets, because `mx.compile` retraces on new shapes.
- Train in 32-bit float. 16-bit halved memory but left 43.9% of weights unchanged after 10 steps at
  a learning rate of 3e-5, against 0% for 32-bit [measured]. Consider quantisation for deployment
  only after translation quality is established.
- Checkpoints hold weights, optimizer, RNG, sampler state and step. The full model's is 3.32 GB [measured].
- A side is at most 256 pieces, raised from 128 for the long paragraph rows of the distilled corpus
  (section 4.6). Longer examples are dropped and counted. Runs so far used 128.

**Measured speed.** On the first corpus, 500 real steps with other apps open: the full model at about
2,500 tokens per second and 1.6 hours per epoch, the baseline at 8,783 and 0.47 hours [measured]. With
MLX's cache capped at 2 GB the full model peaks at 12.3 GB. Eight epochs is about 13 hours for the full
model and 4 for the baseline [estimate]. Runs lasted minutes, so sustained speed is untested
(`docs/steps/04-laptop-benchmark.md`).

### Tokenizer

One joint SentencePiece unigram model at 32,000 pieces, trained on the training split of the first
corpus (section 4.1) and published with it.
- `normalization_rule_name=identity` plus our own NFC. The default `nmt_nfkc` broke sara am: only
  54.9% of Thai round-tripped, against 100% with `identity` [measured]. Piece counts do not reveal
  this, so a test checks the round trip.
- Pad 0, bos 1, eos 2, unk 3, with control symbols `<en>`, `<th>`, `<sep>`, `<ctx>`, `<gloss>`.
  Literal `<th>` in user text must not produce the marker.
- On held-out text at 32,000 pieces: 9.65 Thai and 15.18 English per message, no unknown tokens, and
  every line round-trips [measured on 5,000 test rows].

### Context and glossary

A from-scratch model cannot follow English instructions, so context and glossary entries go in
through `<ctx>` and `<gloss>` spans in the source. Rows carry them only when the expected
translation depends on them. The trainer writes a context line between two `<ctx>` markers, and opens a
teacher-written source with `<sep>`; `<gloss>` is not used yet.

---

## 4. Data

No dataset is treated as perfect because it is paid, published or described as human translated.
Every row keeps its source metadata. Collections are deduplicated against each other. Representative
samples are audited before substantial training time is committed. Split by conversation or document
**before** creating examples in both directions, so one exchange never lands on both sides.

### 4.1 The first training mix

Built by `train/corpus.py`. Each row records its source, subset, review process and licence, so a run
selects exactly what it trains on. Each source's published validation and test rows stay held out.

| Source | Training pairs | Held out | Quality control | Licence |
|---|---:|---:|---|---|
| SCB task dialogs | 182,213 | 36,937 | Professional translators | CC BY-SA 4.0 |
| SCB generated reviews | 105,102 | 27,696 | Professional translators | CC BY-SA 4.0 |
| MTOP Thai | 10,483 | 4,394 | Two translators, one reviewer, annotation review | CC BY-SA 4.0 |
| MASSIVE Thai | 1,132 | 654 | Three human ratings, then a meaning check | CC BY 4.0 |
| **Total** | **298,930** | **69,681** | | |

**MASSIVE** keeps only rows with unanimous top ratings, no localized slots, the same slots as the
English, and a passing Jev majority (`data/massive-th-verified.json`). Spaces between Thai words,
which annotators added, are removed. 2,015 of 16,521 rows survive before deduplication.

**SCB** names its professional subsets differently in the training split, `task_master_corpus` and
`fake_reviews_hope`. Filtering only on the documented names would silently drop them.

**Audit [measured].** Three Jev votes on 150 random training rows per source. Rows judged wrong: task
dialogs 4, generated reviews 18, MTOP 2, MASSIVE 0. The review errors read as real when inspected:
dropped sentences and changed meaning. MASSIVE's 0 is expected, since Jev already filtered it.

**Comparison before commitment.** Train SCB-only and combined runs, and compare with and without the
generated reviews, before adding anything else.

**Not in the first mix:** SCB `generated_reviews_yn`, which is Google output and carries a licence
risk. ALT and TALPCo, whose review processes are not established for each pair. Crawled data and
subtitles (`docs/data-sources.md`). FLORES+ is benchmark only and never trained on.

### 4.2 Published corpus

The combined corpus and its tokenizer are published as a versioned GitHub release, so a laptop or a GPU
box can pull exactly the same data. `.github/workflows/corpus-release.yml` runs on demand and needs no
secrets. It downloads each source from `train/sources.py`, where every file is pinned by sha256, builds
the corpus and tokenizer, and publishes `corpus.tgz`, its checksum, the manifest and an attribution
notice. The package is CC BY-SA 4.0 and a version is never replaced. The build stops if any row has an
unregistered source or licence. A new source must allow redistribution as well as training.

### 4.3 The biggest gap

SCB's human-translated data is English-origin, and no usable public Thai-origin set turned up.
Commissioned Thai-written chat translated into English closes this, with part held out for testing.
Estimated from advertised rates: $3,000 to $12,000 per 10,000 ten-word sentences for translation,
and about half that for review.

### 4.4 Weights licence

ShareAlike on model weights is legally unsettled. If SCB is in, CC BY-SA on the weights is the
safest option. A licence on the weights cannot fix unknown source rights. The tier 2 text of the
distilled corpus (section 4.6) has such rights, so whether it stays is decided before any release of
that corpus or of weights trained on it.

### 4.5 App data

**This project trains on public data only.** No ThaiPo conversation, and nothing derived from one, is
in this repository, in a corpus release or in the published weights. The corpus build stops on any
source that is not registered with a licence (section 4.2), which keeps it that way.

The operator of ThaiPo may separately adapt a private copy of the model to its own conversations.
That work is outside this project. Its data and weights are not published, and no result reported
here depends on it.

Anyone adapting the model to personal text should replace contact details and names with typed
placeholders first, and pass `cmd/extract-audit` before the model serves anyone.

---

### 4.6 Distilled corpus

The human-translated mix covers commands, reviews and task dialogue, and a model trained on it is 25
to 33 chrF behind the teacher on anything else (measured 10 November). A second corpus is meant to
close that gap: a billion clean words of English and Thai text in one language, cleaned, then
translated once by DeepSeek Flash. [`docs/distillation-plan.md`](docs/distillation-plan.md) holds the
plan and [`docs/distillation-sources.md`](docs/distillation-sources.md) the record of every source
and rule.

**Sources are in two licence tiers, and every row carries its tier.** Tier 1 has a licence that
allows commercial use, redistribution and derived works, from whoever holds the rights. Tier 2, added
on 11 November because tier 1 Thai ran out near 500,000 rows, is public Thai text with no stated
licence or a licence that covers only the packaging, web-crawl text included. Tier 2 can be removed in
one step, and whether it stays is decided before any release. Out of both tiers: a dataset whose own
licence forbids the use (non-commercial, no-derivatives, no-training), a source behind an access
agreement, and the output of any model whose terms forbid training on it. The web-crawl text is in
tier 2 on the collection's licence alone; the terms of the individual sites have not been cleared, so
tier 2 is for experiments and is not released before that is settled.

**State on 12 November.** About 900 million Thai words pass the local rules and about 684 million are
counted clean; nearly all are tier 2 web text and news, and conversational Thai is under 2%. English
stands at roughly 54 million words from tier 1 sources and is collected after the Thai. Rows are
cleaned by rules and then scored by small local filters. Nothing has been translated: no paid AI use
is allowed until the billion words are collected, and a priced pilot of 1,000 rows comes first.

**Not yet reproducible.** The reduced set of 11 November was rebuilt byte for byte from pinned
downloads. The crawl collection, the filter scores and the filters themselves are newer, are not all
in this repository, and cannot yet be rebuilt from it. No workflow builds this corpus.

## 5. Evaluation

Build the evaluation sets first, and keep them out of training.

| Set | Use |
|---|---|
| `suites/dev`, `dev2`, `confirm` | Held-out final tests. Synthetic |
| `suites/pilot` | Checkpoint tracking during training |
| `suites/golden` | Must-pass cases for the rollout gate |
| FLORES-200 devtest | Public benchmark, 1,012 sentences. Never trained on. FLORES+ is access-controlled and holds the same sentences with later corrections; it has not been used |
| Chat test set, to be made | 500 to 1,000 messages each way, human-translated, never published or trained on. The proof of section 1, with a blind ranking by Thai speakers (see also section 4.3) |

Training data never contains a suite message: every row whose normalised source appears in any
suite is dropped when the corpus is built. A few short public sentences match today.

Two limits on the suites. DeepSeek drafted every case, and DeepSeek is also the teacher and a system
under comparison, which may favour it and its student. And dev, dev2 and confirm were used to choose
among pretrained models earlier (`docs/pretrained-experiments.md`), and the kind cluster replays
confirm as its shadow corpus, so they are unseen by this model but not untouched. The chat test set
is the one that settles the claim.

**Baseline.** The existing DeepSeek system with its glossary and available context. Both systems
receive equivalent relevant information. `evaluation/run_baseline.sh` is only a stand-in, with no
glossary, and step 7 needs the real one.

**Report per direction, never pooled alone:** meaning preservation, omissions, invented content,
names, numbers, negation, Thai naturalness, politeness, pronouns and glossary compliance. Also
translation latency (p50 and p95 for encoder, first token and per token, at beam 1 and 4), memory,
training time and operating cost.

**Scoring.**
- Blind bilingual human review. None has been done yet.
- chrF as the repeatable automatic metric. Today it is the project's own implementation with greedy
  decoding (`train/evaluate.py`). Moving to sacreBLEU chrF and chrF++ with signatures, and to beam
  search, is an open item. Until then, scores here should not be set beside published ones.
- Jev's three-vote majority for meaning (`evaluation/compare_systems.py`), paired on discordant cases
  with a one-sided 95% upper bound.

**During training**, every half epoch scores chrF on the same 2,000-row validation sample for every run, both ways,
per direction and per source. Training keeps the best checkpoint and stops after three scores without
a gain of at least 0.1 (`train/evaluate.py`). The test split is not touched. Suites hold intended
meanings, not reference translations, so they are judged by Jev on the best checkpoint.

A gap of 1.5 points or less is reported as within reach of production. It is a reference, not a
gate on publishing the result. Thresholds are set before a held-out set is opened.

---

## 6. Sequence

A step is done when it **runs for real, has a test that fails if it breaks, has a short write-up in
`docs/steps/`, and the owner approves.**

1. Inspect the repository, keep what is reusable, replace the plan. Done with this version.
2. Evaluation set first. Section 5.
3. Validate training: learn a tiny dataset, every intended weight updates, a checkpoint restores state
   bit for bit and then tracks a control run within floating-point noise. Built, awaiting approval.
4. Benchmark the laptop: several hundred representative steps, measuring GPU use, memory and
   sustained throughput, before estimating a full run. Measured, awaiting approval.
5. Train on selected public data. Evaluate checkpoints regularly and use measured progress to decide
   whether more training is worth it. The human-translated mix is trained. For the distilled corpus
   (section 4.6) the order is: collect and filter the Thai; collect English to 30 to 40% of the
   total; a priced 1,000-row pilot; distil in steps, training and scoring after each; then training
   that can pass the teacher on chat (the best of several teacher outputs per row, human-translated
   chat pairs weighted at the end, and reinforcement learning with a reward from human judgement).
6. Removed. Adapting to application data is outside this project (section 4.5).
7. Compare against the production system, each direction separately, and run the blind ranking of
   section 1 on the chat test set.

---

## 7. Runtime

Go serves whatever model passes evaluation. It is independent of how the model was built.

- **`cmd/gateway`**: `POST /v1/translate` with guards, one repair, and no text in logs.
- **`cmd/rollout-controller`**: `ModelRollout` resource: verify, golden suite, paired shadow, canary,
  then promoted or rolled back. Infrastructure faults end in `Inconclusive`, never a rollback.
  States and statistics are in `docs/rollout-states.md`.
- **`cmd/suite`** runs a suite against an endpoint. **`cmd/extract-audit`** is the release gate for
  memorised personal details (`docs/extract-audit.md`).
- **kind** runs the stack with a fake model in CI, including the fault matrix.

**Known mismatch.** The gateway's repair turn and `ChatMessages` assume a model that follows English
instructions. The from-scratch model will not. Before it goes behind the gateway, repair becomes a
resample or a failure, and context and gender move to the `<ctx>` channel.

A model server must answer `GET /v1/models/{id}` for the rollout's verification step to pass.

The rollout stack has only been proven against a fake model and the earlier pretrained models, and
claims about it say so.

---

## 8. Public claims

| Safe | Not safe |
|---|---|
| "Trained from random initialisation in MLX on a laptop" | "Better than DeepSeek" unscoped |
| "Within X points of production on this held-out set, this date" | "Human-validated" without a human review |
| The exact data mix, with licences | "Trained on real conversations": the published model is not |
| "Built with AI coding agents, designed and reviewed by me" | Implying it was hand-typed |

**Out of scope:** impersonation, romance or recruitment scams, safety, legal or medical translation
without human review.

---

## 9. Layout

```
cmd/{gateway,rollout-controller,fake-model,suite,extract-audit}
internal/{guards,llm,rollout}  deploy/{crd.yaml,Dockerfile,kind}
train/  evaluation/  suites/  data/{synthetic-chat-v1/,massive-th-verified.json}  tests/
docs/{pretrained-experiments.md,data-sources.md,distillation-plan.md,distillation-sources.md,rollout-states.md,extract-audit.md,steps/}
```

## Sources

[SCB English-Thai 2020](https://huggingface.co/datasets/pythainlp/scb_mt_enth_2020), [ALT](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/index.html), [ChindaMT-Grounded](https://huggingface.co/datasets/iapp/ChindaMT-Grounded), [LDC LORELEI Thai](https://catalog.ldc.upenn.edu/LDC2023T08), [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus), [TALPCo](https://github.com/matbahasa/TALPCo), [Tatoeba](https://tatoeba.org/en/terms_of_use), [DeepSeek terms](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html), [MLX](https://github.com/ml-explore/mlx).
