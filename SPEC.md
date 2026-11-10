# ThaiPoAI: a Thai-English translator trained from scratch

Version 8.0, 9 November 2026. Personal open-source project by Steven Acreman.
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

**What can a roughly 300-million-parameter model, trained from random initialisation on an M4 Max
laptop, achieve against that system, and where does it fall short?**

This is an experiment in learning and in practical quality. We do not assume it will beat DeepSeek.
A clear measurement of the gap counts as success.

### What "good" means

This is a **meaning translator**, not a word-for-word one. Idioms, slang, sarcasm and sayings must
carry their intended meaning. It matches the sender's tone, and when the source is ambiguous it
keeps the ambiguity instead of inventing facts.

---

## 2. Deliverables

1. A reproducible MLX training pipeline.
2. Documented dataset selection (`docs/data-sources.md`).
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

Counts are from instantiated models at a vocabulary of 32,000 [measured]. Both share one matrix for
the source embedding, target embedding and output projection, and that matrix is fully trained.

**Size risk.** The first corpus is about 300,000 pairs, far below what a model of this size usually
needs, so the baseline trains on the same data for comparison.

**Implementation.**
- Hand-written layers. `nn.Transformer` has no decode cache, embeddings or positions.
- Fixed batch buckets, because `mx.compile` retraces on new shapes.
- Train in 32-bit float. 16-bit halved memory but left 43.9% of weights unchanged after 10 steps at
  a learning rate of 3e-5, against 0% for 32-bit [measured]. Consider quantisation for deployment
  only after translation quality is established.
- Checkpoints hold weights, optimizer, RNG, sampler state and step. Config A is 3.32 GB [measured].

**Measured speed.** On the first corpus, 500 real steps with other apps open: the full model at about
2,500 tokens per second and 1.6 hours per epoch, the baseline at 8,783 and 0.47 hours [measured]. With
MLX's cache capped at 2 GB the full model peaks at 12.3 GB. Eight epochs is about 13 hours for the full
model and 4 for the baseline [estimate]. Runs lasted minutes, so sustained speed is untested
(`docs/steps/04-laptop-benchmark.md`).

### Tokenizer

One joint SentencePiece unigram model at 32,000 pieces, retrained on the full corpus.
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
translation depends on them.

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
safest option. A licence on the weights cannot fix unknown source rights.

### 4.5 App data

The two chat applications hold about 300,000 pairs produced by the DeepSeek workflow. Provenance must
be separable into raw model output, automatically corrected and genuinely human-reviewed rows.

**App conversations are not used for training, or sent to outside reviewers, until each message in
the row and its context has its sender's own recorded consent covering publication of the weights.**
The build refuses without a consent manifest. Contact details and names become typed placeholders
first, and `cmd/extract-audit` gates any release. The dataset is never published.

---

## 5. Evaluation

Build the evaluation sets first, and keep them out of training.

| Set | Use |
|---|---|
| `suites/dev`, `dev2`, `confirm` | Held-out final tests. Synthetic |
| `suites/pilot` | Checkpoint tracking during training |
| `suites/golden` | Must-pass cases for the rollout gate |
| FLORES+ | Public benchmark |
| Commissioned Thai-written chat | Held-out conversational test (section 4.3) |
| App conversations | Held-out real-chat test, once section 4.5 is met |

Training data never contains a suite message: every row whose normalised source appears in any
suite is dropped when the corpus is built. A few short public sentences match today.

**Baseline.** The existing DeepSeek system with its glossary and available context. Both systems
receive equivalent relevant information. `evaluation/run_baseline.sh` is only a stand-in, with no
glossary, and step 7 needs the real one.

**Report per direction, never pooled alone:** meaning preservation, omissions, invented content,
names, numbers, negation, Thai naturalness, politeness, pronouns and glossary compliance. Also
translation latency (p50 and p95 for encoder, first token and per token, at beam 1 and 4), memory,
training time and operating cost.

**Scoring.**
- Blind bilingual human review.
- chrF as the repeatable automatic metric.
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
   whether more training is worth it.
6. Adapt to application data, emphasising verified corrections, conversational language and
   terminology. Requires section 4.5.
7. Compare against the production system, each direction separately.

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
| The exact data mix, with licences | "Trained on real conversations with consent" without the manifest |
| "Built with AI coding agents, designed and reviewed by me" | Implying it was hand-typed |

**Out of scope:** impersonation, romance or recruitment scams, safety, legal or medical translation
without human review.

---

## 9. Layout

```
cmd/{gateway,rollout-controller,fake-model,suite,extract-audit}
internal/{guards,llm,rollout}  deploy/{crd.yaml,Dockerfile,kind}
train/  evaluation/  suites/  data/{synthetic-chat-v1/,massive-th-verified.json}  tests/
docs/{pretrained-experiments.md,data-sources.md,rollout-states.md,extract-audit.md,steps/}
```

## Sources

[SCB English-Thai 2020](https://huggingface.co/datasets/pythainlp/scb_mt_enth_2020), [ALT](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/index.html), [ChindaMT-Grounded](https://huggingface.co/datasets/iapp/ChindaMT-Grounded), [LDC LORELEI Thai](https://catalog.ldc.upenn.edu/LDC2023T08), [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus), [TALPCo](https://github.com/matbahasa/TALPCo), [Tatoeba](https://tatoeba.org/en/terms_of_use), [DeepSeek terms](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html), [MLX](https://github.com/ml-explore/mlx).
