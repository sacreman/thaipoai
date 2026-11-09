# ThaiPoAI: a Thai-English translator trained from scratch

Version 8.0, 9 November 2026. Personal open-source project by Steven Acreman.
Public repository `sacreman/thaipoai`. Code is Apache-2.0. The weights' licence follows the
training data (section 4.3).

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

| Config | Encoder/decoder layers | d_model | Heads | Feed-forward | Tying | Parameters |
|---|---|---|---|---|---|---|
| **A, starting point** | 9/9 | 1024 | 16 | 4096 | Source = target = output | 297,197,568 |
| B | 15/15 | 768 | 12 | 3072 | Source = target, separate output | 297,111,552 |
| C | 7/7 | 1024 | 16 | 4096 | All separate | 303,972,352 |

Counts are from instantiated models at a vocabulary of 32,000 [measured]. In A the shared matrix is
still fully trained. Use C if a separate output matrix is wanted. Implementation and memory checks
decide the final configuration.

**Size risk.** 500,000 pairs is about 24M tokens per epoch against 297M parameters [estimate], so a
smaller baseline, in the region of opus-mt's 76M, runs alongside.

**Implementation.**
- Hand-written layers. `nn.Transformer` has no decode cache, embeddings or positions.
- Fixed batch buckets, because `mx.compile` retraces on new shapes.
- Train in 32-bit float. 16-bit halved memory but left 43.9% of weights unchanged after 10 steps at
  a learning rate of 3e-5, against 0% for 32-bit [measured]. Consider quantisation for deployment
  only after translation quality is established.
- Checkpoints hold weights, optimizer, RNG, sampler state and step. Config A is 3.32 GB [measured].

**Sizing, not a benchmark.** Config A, 30 synthetic steps: 32x64 at 2.00 steps/s, 6,565 non-pad
tokens/s, 9.75 GB MLX peak, 13.0 GB footprint; 64x48 at 1.26 steps/s, 6,110 tokens/s, 12.6 GB and
15.9 GB. Identical runs varied from 500 to 1,926 ms per step on a shared machine, and nothing ran
longer than about a minute, so sustained speed is unknown. One pass of 500,000, 2M and 5M pairs in
one direction is about 1.0, 4.1 and 10.3 hours when clean, and 2.2, 8.9 and 22.2 when contended
[estimate]. Step 4 replaces these numbers.

### Tokenizer

One joint SentencePiece unigram model at 32,000 pieces, retrained on the full corpus.
- `normalization_rule_name=identity` plus our own NFC. The default `nmt_nfkc` broke sara am: only
  54.9% of Thai round-tripped, against 100% with `identity` [measured]. Piece counts do not reveal
  this, so a test checks the round trip.
- Pad 0, bos 1, eos 2, unk 3, with control symbols `<en>`, `<th>`, `<sep>`, `<ctx>`, `<gloss>`.
  Literal `<th>` in user text must not produce the marker.
- Mean pieces on public data at 32,000: 19.46 Thai, 24.46 English [measured on 10 MB].

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

### 4.1 Selection

Full notes, sources and criteria are in `docs/data-sources.md`. Starred counts are filter-yield
estimates. Quality here comes from method descriptions, and step 5 audits samples.

| Source | Usable pairs | Licence | Decision |
|---|---:|---|---|
| SCB human and crowd (dialogs, reviews, SMS, voice) | 468,568 | CC BY-SA | Backbone. Oversample dialogs |
| ALT, TALPCo, Tatoeba (held locally) | 26,157 | CC BY, per sentence | Keep, oversample |
| Synthetic chat (`data/synthetic-chat-v1`) | 3,870 | Own, drafted by DeepSeek | Keep. Strip `B: ` speaker prefixes |
| HPLT v2 | about 2.0M* of 4.09M | CC0 packaging per paper | Volume, after confirming the licence |
| SCB auto-aligned | about 250k | CC BY-SA | Possible Thai-origin source |
| Wikimedia | about 50k* | Not verified | Minor, after confirming the licence |
| OpenSubtitles v2024 | about 3M* of 24.8M | None stated | Excluded for now. Best chat register, unknown rights |
| SCB Google-output | 280,208 | Google terms risk | Excluded |
| ChindaMT-Grounded | 1.97M | CC BY-SA | Excluded. Regenerated targets and duplicated sources |
| LDC LORELEI Thai | 2.85M words | No commercial use for non-members | Excluded |
| QED, TED | 528k | Research-only, NC-ND | Excluded |
| FLORES+ | 997 + 1,012 | CC BY-SA, gated | Benchmark only, never trained on |

First mix: about 2.8M pairs. A comparable from-scratch model, quickmt at 195M parameters, used 26.9M
rows, so this mix is about 10% of that and the clean human set about 2%.

### 4.2 The biggest gap

SCB's human-translated data is English-origin, and no usable public Thai-origin set turned up.
Commissioned Thai-written chat translated into English closes this, with part held out for testing.
Estimated from advertised rates: $3,000 to $12,000 per 10,000 ten-word sentences for translation,
and about half that for review.

### 4.3 Weights licence

ShareAlike on model weights is legally unsettled. If SCB is in, CC BY-SA on the weights is the
safest option. A licence on the weights cannot fix unknown source rights.

### 4.4 App data

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
| `suites/dev`, `dev2`, `confirm` | Synthetic. No overlap with public training data. `pilot` and `golden` have a few overlapping sources, so they are not held-out |
| FLORES+ | Public benchmark |
| Commissioned Thai-written chat | Held-out conversational test (section 4.2) |
| App conversations | Held-out real-chat test, once section 4.4 is met |

**Baseline.** The existing DeepSeek system with its glossary and available context. Both systems
receive equivalent relevant information. `train/run_deepseek.sh` is only a stand-in, with no
glossary, and step 7 needs the real one.

**Report per direction, never pooled alone:** meaning preservation, omissions, invented content,
names, numbers, negation, Thai naturalness, politeness, pronouns and glossary compliance. Also
translation latency (p50 and p95 for encoder, first token and per token, at beam 1 and 4), memory,
training time and operating cost.

**Scoring.**
- Blind bilingual human review.
- chrF as the repeatable automatic metric.
- Jev's three-vote majority for meaning (`train/compare_systems.py`), paired on discordant cases
  with a one-sided 95% upper bound.

A gap of 1.5 points or less is reported as within reach of production. It is a reference, not a
gate on publishing the result. Thresholds are set before a held-out set is opened.

---

## 6. Sequence

A step is done when it **runs for real, has a test that fails if it breaks, has a short write-up in
`docs/steps/`, and the owner approves.**

1. Inspect the repository, keep what is reusable, replace the plan. Done with this version.
2. Evaluation set first. Section 5.
3. Validate training: learn a tiny dataset, every intended weight updates, exact checkpoint resume
   against a control run.
4. Benchmark the laptop: several hundred representative steps, measuring GPU use, memory and
   sustained throughput, before estimating a full run.
5. Train on selected public data. Evaluate checkpoints regularly and use measured progress to decide
   whether more training is worth it.
6. Adapt to application data, emphasising verified corrections, conversational language and
   terminology. Requires section 4.4.
7. Compare against the production system, each direction separately.

---

## 7. Runtime

Go serves whatever model passes evaluation. It is independent of how the model was built.

- **`cmd/gateway`**: `POST /v1/translate` with guards, one repair, and no text in logs.
- **`cmd/rollout-controller`**: `ModelRollout` resource: verify, golden suite, paired shadow, canary,
  then promoted or rolled back. Infrastructure faults end in `Inconclusive`, never a rollback.
  States and statistics are in `docs/rollout-states.md`.
- **`cmd/suite`**, **`cmd/dataset-projector`**, **`cmd/extract-audit`** (`docs/extract-audit.md`).
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
cmd/{gateway,rollout-controller,fake-model,suite,dataset-projector,extract-audit}
internal/{guards,llm,rollout}  deploy/{crd.yaml,Dockerfile,kind}
train/  suites/  data/synthetic-chat-v1/  tests/
docs/{pretrained-experiments.md,data-sources.md,rollout-states.md,extract-audit.md,steps/}
```

## Sources

[SCB English-Thai 2020](https://huggingface.co/datasets/pythainlp/scb_mt_enth_2020), [ALT](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/index.html), [ChindaMT-Grounded](https://huggingface.co/datasets/iapp/ChindaMT-Grounded), [LDC LORELEI Thai](https://catalog.ldc.upenn.edu/LDC2023T08), [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus), [TALPCo](https://github.com/matbahasa/TALPCo), [Tatoeba](https://tatoeba.org/en/terms_of_use), [DeepSeek terms](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html), [MLX](https://github.com/ml-explore/mlx).
