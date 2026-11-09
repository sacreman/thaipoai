# Pretrained-model experiments: what was done and what it showed

ThaiPo is a production English-Thai chat translator that uses hosted models. This repo tested whether an open model, fine-tuned and run on our own hardware, could replace them. The work ran on 8 and 9 November 2026, in commits `de0dfbb`, `9274e18` and `7e7e67d`. The model code has been removed from the tree, and `7e7e67d` is the last commit that has it. This file is the record.

| Goal | Meaning | Outcome |
|---|---|---|
| M1 Trained | A LoRA adapter beats its untuned base on the dev suite | Not reached. Both adapters were worse |
| M2 Not worse | Within 1.5 points of production, one-sided 95% upper bound, per direction | Not reached. 1.70 and 4.03 against a stand-in for production |
| M3 Replaced | Serves real ThaiPo traffic | Not started |

## Setup

- **Hardware:** Apple M4 Max, 32-core GPU, 36 GB unified memory. PyTorch 2.15 on MPS, Transformers 5.22, PEFT 0.21. Lumen used MLX. DeepSeek Flash was called as a hosted API.
- **Data:** all synthetic or public. No real chats were used for training or testing.
- **Suites, all synthetic:**

| Suite | Cases | Style | Used for |
|---|---|---|---|
| Pilot | 120 (60 per direction) | Idiom, negation, register, sarcasm and similar | Go/no-go, adapter tests |
| Confirm | 120 (60 per direction) | Same categories, no shared messages | Confirmation, model screen |
| Dev | 200 (100 per direction) | Messy texting: typos, shorthand, romanised Thai, fragments | Adapter and retrieval tests |
| Dev2 | 398 (198 en-th, 200 th-en) | Same style as dev | Fresh check of retrieval and the split |
| Golden | 36 | Must-pass cases | Rollout gate only |

- **Judges changed during the work**, and each section names its judge. Opus 5.5 scored blind from 1 to 5 on meaning, and misleading means 2 or lower. Jev (`jev-latest`) is an AI judge that answers yes or no on whether the intended meaning is carried, with one vote at first and later three votes and a majority.

## 1. Feasibility: can these models train on the Mac?

8 November. Data was synthetic. Every training target ended with a marker, ` #ok`, that the untuned model never writes. Writing it on unseen rows proves training changed the model. These adapters test the machinery only. Settings: LoRA rank 64, alpha 128, bf16 weights, gradient checkpointing, AdamW at 2e-4, attention projections only.

| | Hy-MT2 7B | Qingqiu-MT-9B | TranslateGemma 12B | Gemma 4 E4B | Lumen 8-bit (MLX) |
|---|---|---|---|---|---|
| Weights on disk | 16.1 GB | 18.8 GB | 24.4 GB | 16 GB | 15 GB |
| Parameters | 7.56B | 9.01B (text only) | 11.86B | 7.98B | 14.77B |
| Trainable (adapter only) | 54.5M | 59.8M | 84.9M | 36.3M | 83.9M |
| Speed, batch 4 | 183.5 tokens/s | 145.2 tokens/s | 16.8 tokens/s | 195 tokens/s | Not comparable |
| Average row | 131 tokens | 163 tokens | 186 tokens | 195 tokens | Not reported |
| Rows per hour | 5,041 | 3,208 | 327 | 3,607 | about 1,600 |
| 20,000 rows | about 4 hours | about 6.2 hours | about 61 hours | about 5.5 hours | about 12.5 hours |
| GPU memory, translating | 14.1 GB | 16.9 GB | 23.9 GB | 15.06 GB (a) | 15.2 GB peak |
| Peak GPU driver memory | 15.9 GB | 19.2 GB | 24.2 GB | 16.4 GB | Not reported |
| Marker on 20 unseen rows, untuned and adapter | 0%, 100% | 0%, 100% | 0%, 100% | 0%, 100% | 0%, 100% |

(a) Driver memory during the adapter check. The other PyTorch figures come from a translation run.

What it showed:

- Hy-MT2 7B, Qingqiu-MT-9B, Gemma 4 E4B and Lumen 8-bit fit a night's training.
- TranslateGemma 12B trained, but it used about 12 GB of swap and needed a GPU memory limit of about 35 GB. Hy-MT2 7B and Qingqiu-MT-9B used 27 GB. It is not practical to run repeatedly on this machine.
- A shared machine can halve throughput. The first 7B timing was 91 tokens/s beside a 24 GB download.
- Learning rate matters. At 1e-3 the 7B loss rose from 3.11 to 4.72, and at 2e-4 it fell to about 0. `mlx_lm.lora` has no gradient clipping. At 2e-4 Lumen's loss reached 10.46, and 5e-5 worked.
- Qingqiu-MT-9B needs an explicit stop token and thinking mode turned off. MPS needs both watermarks set.
- Given chat context in its text field, untuned TranslateGemma translated the context instead.

## 2. Desk research on base models

Three research reports read published papers and licences. Nothing here was tested by us.

- **Translation labels did not predict Thai.** At WMT26 English to Thai, Tower+ 9B scored 20.4, unmodified Gemma 4 E4B 62.4 and Lumen 84.0.
- **Out on licence:** HY-MT1.5-7B and Hunyuan-MT-7B (regional exclusions), Aya and EXAONE (non-commercial), Llama and GLM-4 (forced naming on fine-tunes), Typhoon (unresolved terms on fine-tuning).
- **Out on memory:** TranslateGemma 12B, and Gemma 4 31B, whose 8-bit build would be about 31 GB against the 27 GB limit.
- **Out on lineage, kept only as yardsticks:** TranslateGemma, MiLMMT-46, Qingqiu-MT-9B and Index-Translate, whose papers name Gemini or GPT output as training targets or judges.
- **Seed-X** missed the shortlist. Its card advises against multi-round format.

## 3. Go/no-go on Lumen

8 November. Untuned Lumen at 8-bit on the Mac. Pilot suite, 360 translations: no context, correct context, irrelevant context. Scored by Opus 5.5, with Jev as a cross-check. Pass rules were written before the run: misleading at or below 15% in each direction with correct context, context translated in 2% of cases or fewer, correct context no worse than none, irrelevant context within 5 points of none.

| Format for sending history | en-th, correct | th-en, correct | Context translated | Irrelevant, both directions | Rules passed |
|---|---|---|---|---|---|
| `guidelines` block | 55.0% | 90.0% | 73.3% | 53.3% | 0 of 5 |
| `reference` block | 31.7% | 86.7% | 55.0% | 38.3% | 0 of 5 |
| Before the request | 26.7% | 76.7% | 48.3% | 50.0% | 0 of 5 |
| **System message** | **11.7%** | **18.3%** | **0%** | **19.2%** | **4 of 5** |

Moving history into a system message fixed the main failure. Thai to English still missed the bar, 11 misleading of 60 where 9 were allowed.

**Confirmation.** One run on the 120 confirm cases, system message only. English to Thai 10.0%, Thai to English 18.3% (fail), context translated 1.7%. Thai to English was 11 of 60 on both sets.

**Against DeepSeek Flash.** Same 240 cases and variants, temperature 0.7, thinking off, no glossary or hints. Scored blind by Opus 5.5, and no rule was applied. 720 translations each.

| Misleading rate | Lumen 8-bit | DeepSeek Flash | Gap (points) |
|---|---|---|---|
| en-th, correct context | 10.8% | 3.3% | 7.5 |
| th-en, correct context | 18.3% | 10.0% | 8.3 |
| en-th, no context | 11.7% | 5.8% | 5.9 |
| th-en, no context | 22.5% | 13.3% | 9.2 |

## 4. Base-model screen and selection

8 November. Confirm suite, 60 cases per direction, no context and correct context, 240 translations per model. Untuned, at 16-bit in PyTorch on the Mac except Lumen. Each model used its own documented prompt format. Scored by Opus 5.5 and Jev. Figures are Opus misleading rates.

| Model | en-th, context | th-en, context | en-th, none | th-en, none | Context translated | Outcome |
|---|---|---|---|---|---|---|
| DeepSeek Flash | 3.3% | 10.0% | 6.7% | 15.0% | 0 | Hosted yardstick |
| **Gemma 4 E4B** | 3.3% | 18.3% | 10.0% | 25.0% | 0 | **Chosen after a second set** |
| Gemma 4 12B | 1.7% | 11.7% | 5.0% | 20.0% | 0 | Not chosen. Needs a cloud GPU to train |
| Index-Translate-9B | 3.3% | 16.7% | 8.3% | 21.7% | 0 | Comparison only. Paper names Gemini as a judge |
| Hy-MT2 7B | 11.7% | 16.7% | 15.0% | 31.7% | 0 | Not chosen. Behind on English to Thai |
| Lumen 8-bit | 10.0% | 18.3% | 8.3% | 25.0% | 2 | Replaced. Failed the th-en rule twice |
| Qingqiu-MT-9B | 8.3% | 33.3% | 8.3% | 30.0% | 16 | Comparison only. Gemini output in training |
| ChindaMT-4B | 13.3% | 58.3% | 16.7% | 41.7% | 37 | Not chosen. Often translated the context |
| Mango-MT-9B | 18.3% | 23.3% | 30.0% | 51.7% | 1 | Not chosen. No documented prompt format |

Gemma 4 E4B then repeated the test on the pilot suite. The rule set before that run was: no worse than Lumen in both directions, on both sets, with correct context. It met that rule. Both sets pooled, correct context, 120 per direction: Gemma 4 E4B 2.5% en-th and 11.7% th-en, DeepSeek Flash 3.3% and 10.0%, Lumen 8-bit 10.8% and 18.3%.

The owner first kept Lumen, then chose Gemma 4 E4B (`google/gemma-4-E4B-it`, revision `4ecc2ba`) after this round. Gemma's own two sets differed on Thai to English, 5.0% and 18.3%, which shows how noisy 60 cases are. Chinda and Mango might do better with a different prompt. That was not tested.

## 5. Archive count

A count-only replay of the ThaiPo archive found 93,005 rows that passed the metadata filters; it decoded no message text and none of those rows were used for training or testing.

## 6. Public-data adapter, 8 hours

Gemma 4 E4B with LoRA, the settings from section 1, batch 4, on MPS. Data: public sentence pairs from ALT (19,896), TALPCo (1,366) and Tatoeba (4,895), each used in both directions, giving 51,330 training rows and 984 validation rows. No chat context. The run took 9,762 steps, under one pass of the data. The best validation loss came at step 3,086. The report does not record which checkpoint was scored, and the comparison script defaulted to the best-validation one. Scored on the pilot suite, no context and correct context, 240 translations, one Jev vote each, paired against untuned Gemma.

| Slice | Untuned wrong | Adapter wrong | Fixed | Broken | p (adapter worse) |
|---|---|---|---|---|---|
| All | 62 | 92 | 4 | 34 | 0.0 (four places) |
| en-th | 26 | 36 | 1 | 11 | 0.0032 |
| th-en | 36 | 56 | 3 | 23 | 0.0 (four places) |
| Correct context | 29 | 43 | 2 | 16 | 0.0007 |
| No context | 33 | 49 | 2 | 18 | 0.0002 |

It was clearly worse in every slice.

## 7. Synthetic-chat adapter, 2 hours

Same recipe. Data: 3,870 training rows and 203 validation rows, drafted by DeepSeek in a chat style with four turns of context. Pairs were kept only if Jev said they carried the meaning, and messages already in a suite were refused. Latest checkpoint at step 1,714, best validation at step 812. One Jev vote per translation.

| Checkpoint and suite | Translations | Untuned wrong | Adapter wrong | Fixed | Broken | p (adapter worse) |
|---|---|---|---|---|---|---|
| 2 hours, pilot | 240 | 63 | 69 | 13 | 19 | 0.1885 |
| Best validation, pilot | 240 | 62 | 72 | 9 | 19 | 0.0436 |
| 2 hours, dev | 400 | 70 | 91 | 14 | 35 | 0.0019 |

On the dev suite the damage was in Thai to English, 28 wrong becoming 49 (7 fixed, 28 broken). English to Thai stayed at 42. On the pilot suite the 2-hour adapter was ahead with correct context, 29 wrong becoming 26, and behind with none, 34 becoming 43.

Fine-tuning stayed off after this, until there is data it can improve on.

## 8. Retrieved examples

Instead of changing the weights, each message was given three similar synthetic pairs in the prompt: the same 3,870 rows, TF-IDF cosine over character trigrams of the message, same direction only, never the message itself. Untuned Gemma 4 E4B, temperature 0.

Dev and dev2 suites, no context and correct context. DeepSeek Flash at temperature 0.7, no glossary. Jev judged each translation three times and the majority counted. No translation was left unjudged. Lower is better.

| Wrong, three-vote majority | Pairs | DeepSeek Flash | Untuned Gemma | Gemma + examples |
|---|---|---|---|---|
| Dev, all | 400 | 66 | 76 | 73 |
| Dev, en-th | 200 | 38 | 47 | 41 |
| Dev, th-en | 200 | 28 | 29 | 32 |
| Dev2, all | 796 | 119 | 130 | 121 |
| Dev2, en-th | 396 | 42 | 43 | 39 |
| Dev2, th-en | 400 | 77 | 87 | 82 |

Pooled, 598 cases and 1,196 translations, against untuned Gemma the examples fixed 41 and broke 29 (one-sided p = 0.0941). The gain is without context, 28 fixed and 13 broken. With correct context they did nothing useful, 13 fixed and 16 broken.

## 9. The direction split that failed

On the dev suite, examples looked good for English to Thai (8 fixed, 2 broken, p = 0.0547) and bad for Thai to English (2 fixed, 5 broken). So a split was proposed: examples for English to Thai only.

It was tested on dev2, which it had not seen. The Thai to English pattern did not repeat. Examples helped there (22 fixed, 17 broken), and the split gave that up.

On dev2 with three-vote judging, DeepSeek against the split was 42 and 39 wrong in English to Thai (gap -0.76 points) and 77 and 87 in Thai to English (gap 2.5). The report marks the split as not passing, and it was dropped. It had been fitted to one suite.

## 10. Frozen candidate

`gemma-examples-v1`, frozen 9 November: Gemma 4 E4B at the pinned revision, no adapter, temperature 0, the chat prompt and the retrieval from section 8. A test failed if the model, prompt, pairs or retrieval drifted. Retrieval was measured but never wired into the gateway.

| Slice, 1,196 translations pooled | Pairs | DeepSeek wrong | Candidate wrong | Gap (points) | One-sided 95% upper bound |
|---|---|---|---|---|---|
| All | 1,196 | 185 (15.5%) | 194 (16.2%) | +0.75 | 2.28 |
| English to Thai | 596 | 80 (13.4%) | 80 (13.4%) | 0.00 | 1.70 |
| Thai to English | 600 | 105 (17.5%) | 114 (19.0%) | +1.50 | 4.03 |
| With correct context | 598 | 77 (12.9%) | 87 (14.5%) | +1.67 | 3.84 |
| No context | 598 | 108 (18.1%) | 107 (17.9%) | -0.17 | 1.98 |

**Why it did not pass M2.** The gate needs an upper bound of 1.5 points or less in each direction. English to Thai reached 1.70 and Thai to English 4.03. The gate also compares against production on held-out chats. That was never run. DeepSeek here had no production glossary, hints or repair, so production is likely a harder comparison. The bounds reproduce with a normal approximation on the paired difference at z = 1.645.

## 11. What limits these results

- **The same model family wrote and took the test.** All pilot, confirm, dev and dev2 messages and all retrieved examples were drafted by DeepSeek (`deepseek-v4-pro`), and DeepSeek Flash was the comparison. That may favour DeepSeek. Examples and cases sharing an author may favour retrieval.
- **Judges were AI only.** No human checked any suite, translation or verdict. Sections 8 to 10 rely only on Jev's three-vote majority, so a candidate can be fitted to its habits.
- **Judges disagreed.** Opus and Jev agreed on between 0.6708 and 0.7958 of rows across runs.
- **Scores from the two judges are not comparable.** On the pilot suite with correct context, Opus marked untuned Gemma misleading in 1.7% and 5.0% of cases per direction. Jev marked 29 of 120 wrong. The questions differ.
- **Single votes are noisy.** The same untuned Gemma translations on the dev suite got 70 wrong on one single-vote pass, 75 on another and 76 under the three-vote majority.
- **No untouched suite.** Dev and dev2 both informed the decisions and both are in the pooled result.
- **Pairs are not independent.** Each case counts twice, with and without context, so the bounds may be too tight.
- **Small sets.** With 60 cases per direction, a single rate is good to about 8 to 10 points.
- **Short synthetic messages only.** Real chats were not used, so these results may not hold on production traffic.

## 12. Serving and rollout platform

Built for these models on 9 November, alongside sections 6 to 10. It ran on the Mac and on a local kind cluster. It never ran on real vLLM or real GPUs. Candidates in these tests were demo adapters.

| Piece | What it does | How far it got |
|---|---|---|
| `cmd/gateway` | `POST /v1/translate`, guards, one repair, logs no text | Four real requests against Gemma 4 E4B on the Mac. No load test |
| `cmd/adapter-sync` | Sidecar: downloads, verifies sha256, loads, tests each adapter | Ran on kind with one real model replica. 15 Go tests |
| `cmd/rollout-controller` | Verifying, Golden, Shadow, Canary, then Promoted, Blocked, RolledBack or Inconclusive | Fault matrix passed 19 of 19 checks on kind, with a fake model |
| `cmd/reaper`, Terraform, GitHub workflow | Temporary GPU pipeline | 17 Go tests against fakes. **Never run against a cloud** |
| `cmd/extract-audit` | Planted-canary memorisation gate | Written. No report of a run against a trained adapter |

**Guards do not protect meaning.** Run over 1,320 saved translations already scored by Opus, of which 129 were misleading. The repairing guards fired on 48 (3.6%), and only 5 of the 129 were among them. `negation` fired 29 times with 6 misleading, so it reports only.

**Rollout rule.** An outage is never a rollback. Infrastructure failures end in `Inconclusive`. Quality gates count discordant pairs.

**Real-model demos, one Mac replica.** An adapter that swapped gender particles was Blocked at Golden, 29 worse of 36 cases, in 95 s, and never had traffic. An output-bloat adapter passed Golden and Shadow, then RolledBack at Canary after 642 s, at 1.96 times stable's output per source character.

An independent review found 11 defects and 4 test gaps, all fixed with tests. The gates see only what guards and the golden suite can see. They do not judge translation quality.

## 13. Planned but not run

LoRA against training every weight. 0, 4 and 8 turns of context. Real data against public data. The real-chat benchmark (2,000 held-out lines per direction). The 300-case meaning suite. Merging an adapter into the weights. vLLM load numbers. The temporary GKE training pipeline.

## What this leaves for the next experiment

- **Context helps, if it goes in a system message.** Correct context cut DeepSeek from 108 wrong to 77 and the candidate from 107 to 87. In Lumen, history in the user turn got translated in up to 73.3% of cases.
- **The evaluation method transfers.** Pre-written rules, paired comparison on discordant pairs, three-vote judging and one-sided bounds all work on any translator.
- **Messy texting is unsolved.** DeepSeek was wrong on 15.5% and the candidate on 16.2%. Jev did not rate the cleaner pilot suite better (62 of 240 wrong for untuned Gemma), so messiness alone is not shown to be the cause.
- **Do not assume a fine-tune helps.** Two adapters got worse. Test each against its untuned base first, and use training data that targets the observed errors.
- **Check small effects on a fresh suite, and get a human to validate the judge.** The split came from one suite, and Gemma's Thai to English rate moved from 5.0% to 18.3% between two sets of 60 cases.
