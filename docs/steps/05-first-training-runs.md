# Step 5: first training runs

9 and 10 November 2026. M4 Max, 36 GB, the `corpus-v1` release. Every run was scored with chrF on the
same 2,000 validation rows, translated both ways, every half epoch (`train/evaluate.py`). The sample
holds 1,062 SCB task dialogs, 824 SCB reviews, 89 MTOP rows and 25 MASSIVE rows. The test split was not
used.

## The 75M baseline: which data helps

Four runs of the baseline model, eight epochs each, with the settings `train/train.py` had at the
time: dropout 0.1, label smoothing 0.1, no word dropout and no sampled segmentation. Its defaults have
since changed to a low-resource recipe (dropout 0.3, label smoothing 0.2, word dropout 0.1, sampled
segmentation), which scored lower: 43.22 against 44.68 on these rows
([distillation plan](../distillation-plan.md), section 1).

| Run | Training pairs | Best chrF | English to Thai | Thai to English | SCB task dialogs | MTOP | MASSIVE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Combined | 298,930 | **44.68** | 43.13 | 46.22 | 45.89 | 55.25 | 52.20 |
| SCB only | 287,315 | 44.03 | 42.62 | 45.43 | 45.69 | 33.66 | 33.43 |
| Without SCB reviews | 193,828 | 32.86 | 31.18 | 34.54 | 45.62 | 54.10 | 49.16 |
| Without SCB reviews, seed 1 | 193,828 | 32.75 | 31.04 | 34.46 | 45.34 | 53.49 | 44.87 |

- **Chance moves task dialogs by about 0.3.** The two runs that differ only in seed are 0.28 apart.
- **MTOP and MASSIVE add 0.20 on task dialogs**, within chance. Their large gains are on their own
  rows, which are few and templated.
- **Leaving out the reviews lowers the headline score only because 41% of the sample is review text**
  those runs never saw. On task dialogs the reviews add 0.27 to 0.55, about the size of chance, while
  making training about two thirds longer (52,248 steps against 30,920).
- **Every run levelled off.** The combined run's last four scores, about two epochs, stayed within 0.3.

The SCB-only run took 2 h 49 min and each run without reviews about 1 h 50 min.

## Against DeepSeek on the pilot suite

The best combined checkpoint (step 48,975, greedy decoding, no context) and DeepSeek (no context)
translated the 120 cases in `suites/pilot.jsonl`. Jev's three-vote majority judged whether each output
carried the intended meaning (`evaluation/compare_systems.py`).

| Direction | Cases | DeepSeek wrong | 75M wrong |
|---|---:|---:|---:|
| English to Thai | 60 | 15 (25%) | 35 (58%) |
| Thai to English | 60 | 17 (28%) | 47 (78%) |
| **Total** | 120 | **32 (27%)** | **82 (68%)** |

The 75M model got 53 cases wrong that DeepSeek got right, and fixed 3. The gap is 41.7 points, with a
one-sided 95% upper bound of 49.8. The pilot is deliberately hard: dropped subjects, speaker gender,
references, idioms, slang, sarcasm, numbers and times, register, negation, and typos with mixed
language. The corpus contains almost no chat.

## The 297M model, stopped a third of the way through

The full model trained on the combined data with the same settings. It was stopped at step 16,550 of
52,248 to free the laptop for other experiments. The step 16,325 checkpoint is kept and can be resumed.

| Step | 297M chrF | 75M chrF | Gap |
|---:|---:|---:|---:|
| 3,265 | 19.80 | 15.59 | +4.21 |
| 6,530 | 29.91 | 29.44 | +0.47 |
| 9,795 | 33.09 | 35.00 | −1.91 |
| 13,060 | 35.29 | 37.19 | −1.90 |
| 16,325 | 37.10 | 38.74 | −1.64 |

At step 16,325 it was behind in both directions: 35.46 against 37.48 into Thai, and 38.73 against 39.99
into English. The gap had begun to narrow, and the 75M levelled off near 44.7 later, so the larger model
might still pass it. This run does not show whether it would.

Alone, the full model trained at about 2,100 tokens per second. Sharing the GPU with a second training
run cut that to about 1,500, and when a third large process started, swap reached 27 GB and speed fell to
about 1,000.

## What did not work

- **Changing the data mix.** No variant beat the combined set on task dialogs by more than chance.
- **Training the 75M longer.** It stopped improving in its last two epochs.
- **Scaling to 297M on 300,000 pairs, so far.** It trailed the 75M at every score from step 9,795 on.
- **Formal public text alone.** The best model got 68% of the pilot wrong, against DeepSeek's 27%.

## Limits

chrF counts shared characters, not meaning. When this was written DeepSeek had not been scored on the
validation rows. It was on 10 November: 46.72 on the same 2,000 rows against the 75M model's 44.68,
and 61.50 against 36.16 on 900 pairs of unfamiliar text
([distillation plan](../distillation-plan.md), section 1). Only the runs without reviews were repeated with a second seed. The
pilot was judged by one automatic judge on 120 cases, not by the blind bilingual review in the spec.
Results, checkpoints and logs are in `runs/`, which is not published.
