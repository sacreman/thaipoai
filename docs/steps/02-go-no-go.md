# Step 2: is Lumen a usable base?

Written before the run, so the bar cannot move to fit the result.

Lumen has no published Thai-to-English result and its chat context has never been tested.
This step checks both before anything depends on them. It uses synthetic test cases only.

## Test

- **Model:** untuned Lumen at 8-bit on the Mac, through `cmd/suite`.
- **Cases:** 120 short dialogues, 60 in each direction, drafted by DeepSeek. Each has the
  message, its intended meaning, a relevant context and an irrelevant one.
- **Variants:** no context, correct context, irrelevant context. 360 translations.
- **Scoring:** Opus 5.5, blind, 1 to 5 on meaning with a defect class. Misleading means a
  score of 2 or lower. Jev independently answers whether the meanings match.

## Pass rules

| # | Rule | Bar |
|---|---|---|
| 1 | English to Thai misleading rate, correct context | 15% or lower |
| 2 | Thai to English misleading rate, correct context | 15% or lower |
| 3 | Context translated instead of the message | 2% of cases or fewer |
| 4 | Correct context versus no context | Misleading rate no worse |
| 5 | Irrelevant context versus no context | No more than 5 points worse |

Failing rule 2 or 3 is a no-go: decide on a different base before any paid training. A
narrow miss on the others is a judgement call for the owner.

With 60 cases per direction, this is a filter, not a measurement: a single rate is only
good to about 8 to 10 points either way. The 8-bit model may also score a little below
16-bit. These are AI scorers only.

## Results

Run 8 November 2026, untuned Lumen at 8-bit, 360 translations, none lost to errors. Context
was passed in Lumen's `#guidelines` block, with an instruction not to translate it.

**Verdict: no-go for this prompt format. All five rules failed.**

| # | Rule | Bar | Result |
|---|---|---|---|
| 1 | English to Thai misleading, correct context | 15% or lower | 55% |
| 2 | Thai to English misleading, correct context | 15% or lower | 90% |
| 3 | Context translated instead of the message | 2% or fewer | 73% |
| 4 | Correct context versus no context | No worse | 72.5% against 17.5% |
| 5 | Irrelevant context versus no context | Within 5 points | 53.3% against 17.5% |

Misleading rate by direction and variant:

| Direction | No context | Correct context | Irrelevant context |
|---|---|---|---|
| English to Thai | 15% | 55% | 25% |
| Thai to English | 20% | 90% | 82% |

**What it means.** Without context, Lumen is reasonable in both directions, including
Thai to English, which had no published result. The failure is what happens when chat
history is added: most of the time Lumen translates the conversation instead of the message.
The largest defect class with context was `leak`, 46 of 90 misleading rows.

So this result is about this way of supplying context, as much as about Lumen. It does not
show whether another prompt format, or fine-tuning, would fix it.

## Context formats

Written before the runs. The rules above do not change. Three further ways of sending the
history are tested on the same 120 cases, and all are reported whether they pass or fail:

| Format | Where the history goes |
|---|---|
| `guidelines` | In `#guidelines` (the failed run above) |
| `reference` | In `#guidelines`, with a firmer instruction to translate only `#src_text` |
| `system` | In a system message |
| `before` | Before the translation request |

No-context rows are the same prompt in every format, so they are reused. Picking the best of
four on one set of cases flatters the winner, so any format that passes must be confirmed on
a fresh set of cases before it is relied on.

### Format results

Run 8 November 2026, same 120 cases, no requests lost. Misleading rates:

| Format | English to Thai, correct | Thai to English, correct | Context translated | Irrelevant context, both directions | Rules passed |
|---|---|---|---|---|---|
| `guidelines` | 55.0% | 90.0% | 73.3% | 53.3% | 0 of 5 |
| `reference` | 31.7% | 86.7% | 55.0% | 38.3% | 0 of 5 |
| `before` | 26.7% | 76.7% | 48.3% | 50.0% | 0 of 5 |
| **`system`** | **11.7%** | **18.3%** | **0%** | **19.2%** | **4 of 5** |

**A system message fixes the main failure.** No translations of the conversation at all, and
context no longer hurts. Thai to English is 11 misleading out of 60, and the bar of 15% allows
9. Rule 2 therefore still fails, so as written this is still a no-go, by two cases. At this
sample size that is well inside the measurement noise, in both directions.

This was the best of four formats on one set of cases, so its figures are likely to flatter
it, and it has to be confirmed on fresh cases either way.

### Confirmation run

Written before the run. One shot, on 120 new cases that share no message with the first set,
using the `system` format only. The same five rules apply, with none adjusted. If Thai to
English comes in at 15% or lower and the other rules pass, Lumen is a go. If not, Hy-MT2 7B
and Qingqiu-MT-9B get the same test before a base is chosen.

**Result, 8 November 2026: no-go. Rule 2 failed again, by the same margin.**

| # | Rule | Bar | Result | Pass |
|---|---|---|---|---|
| 1 | English to Thai misleading, correct context | 15% or lower | 10.0% | Yes |
| 2 | Thai to English misleading, correct context | 15% or lower | 18.3% | **No** |
| 3 | Context translated instead of the message | 2% or fewer | 1.7% | Yes |
| 4 | Correct context versus no context | No worse | 14.2% against 16.7% | Yes |
| 5 | Irrelevant context versus no context | Within 5 points | 21.7% against 16.7% | Yes |

Thai to English was 11 misleading of 60 on both independent sets, 22 of 120 pooled. That
repeat makes noise a less likely explanation. English to Thai pooled is 13 of 120 (10.8%).
Lumen remains usable in the system-message format, and its figures are the reference the other
models have to beat. This was the 8-bit model, so 16-bit may score differently.

## DeepSeek comparison

Written before the run. The confirmation result above does not change.

DeepSeek Flash runs on the same 240 cases and the same three context variants, at
production's settings: temperature 0.7, thinking off. Opus and Jev score it blind, with the
same rubric. Lumen's figures are the system-message runs above. No rule is applied: this
only measures the gap.

**What this is not.** It uses an independently written prompt, not ThaiPo's production
pipeline, so it has no hints, repair loop or production glossary, and none of these test cases
has glossary terms. It shows how Lumen compares with DeepSeek as a translator on this test.
DeepSeek also wrote these test cases, which may give it a small edge.

### Result, 8 November 2026

DeepSeek Flash was clearly better. 240 cases, 720 translations each, none lost to errors.

| Misleading rate | Lumen 8-bit | DeepSeek Flash | Gap |
|---|---|---|---|
| English to Thai, correct context | 10.8% | 3.3% | 7.5 points |
| Thai to English, correct context | 18.3% | 10.0% | 8.3 points |
| English to Thai, no context | 11.7% | 5.8% | 5.9 points |
| Thai to English, no context | 22.5% | 13.3% | 9.2 points |
| English to Thai, irrelevant context | 14.2% | 2.5% | 11.7 points |
| Thai to English, irrelevant context | 26.7% | 14.2% | 12.5 points |

With correct context, Lumen failed where DeepSeek passed on 27 cases, and the reverse happened
on 8. The largest gaps by category were English to Thai negation and register, and Thai to
English idioms (Lumen 6 misleading of 12, DeepSeek 1). On Thai speaker gender the two tied.

The project's pass margin is 1.5 points, so untuned Lumen is several times further away than
training has to bring it. DeepSeek here had no glossary, hints or repair, so production is
likely stronger still. It also wrote the cases, and the Lumen tested was the 8-bit copy.

## Other base models

Written before the run. Hy-MT2 7B, Qingqiu-MT-9B, Gemma 4 E4B and Index-Translate-9B are
screened on the fresh set of 120 cases, with no context and correct context, untuned at 16-bit
in PyTorch on the Mac, each in its own documented prompt format. Each carries history in
whatever way its documentation provides, with the general-purpose Gemma 4 given the DeepSeek
comparison prompt. Scoring is the same, and every model is reported.

A screen on one set of cases can only rank candidates. No model replaces Lumen on this
result alone, and a chosen model would still need the second set of cases. Qingqiu is
included for comparison only, since its paper names Gemini output among its training data.

### Screen result, 8 November 2026

Fresh set only, 60 cases per direction, none lost to errors. Misleading rates:

| Model | English to Thai, context | Thai to English, context | English to Thai, none | Thai to English, none | Context translated |
|---|---|---|---|---|---|
| DeepSeek Flash | 3.3% | 10.0% | 6.7% | 15.0% | 0 |
| **Gemma 4 E4B** | **3.3%** | 18.3% | 10.0% | 25.0% | 0 |
| **Index-Translate-9B** | **3.3%** | 16.7% | 8.3% | 21.7% | 0 |
| Hy-MT2 7B | 11.7% | 16.7% | 15.0% | 31.7% | 0 |
| Lumen 8-bit | 10.0% | 18.3% | 8.3% | 25.0% | 2 |
| Qingqiu-MT-9B | 8.3% | 33.3% | 8.3% | 30.0% | 16 |

Gemma 4 E4B and Index-Translate-9B matched DeepSeek on English to Thai here, each with 2
misleading of 60. Index-Translate's paper names Gemini as a training judge, so like Qingqiu it
is a comparison only unless the owner decides otherwise. No model reached DeepSeek on Thai to English. The other models were tested
at 16-bit and Lumen at 8-bit. This is one set of 60 cases per direction, so a difference of
a few cases means little, and DeepSeek wrote the cases.

### Second round

Written before the run. Gemma 4 E4B repeats the test on the first set of cases, so it has been
tested on two independent sets. Gemma 4 12B, ChindaMT-4B and Mango-MT-9B take the screen on the
fresh set. Mango publishes no prompt format, so it gets the neutral comparison prompt.

Gemma 4 E4B becomes the recommended lead only if it is no worse than Lumen in both directions,
on both sets, with correct context. The owner makes that decision.

**Result, 8 November 2026.** All runs completed with no requests lost. Misleading rates with
correct context:

| Model | English to Thai | Thai to English | Translated context | Cases |
|---|---|---|---|---|
| **Gemma 4 E4B, both sets pooled** | **2.5%** | **11.7%** | 0 | 120 per direction |
| DeepSeek Flash, both sets pooled | 3.3% | 10.0% | 0 | 120 per direction |
| Lumen 8-bit, both sets pooled | 10.8% | 18.3% | 2 | 120 per direction |
| Gemma 4 12B, fresh set only | 1.7% | 11.7% | 0 | 60 per direction |
| ChindaMT-4B, fresh set only | 13.3% | 58.3% | 37 | 60 per direction |
| Mango-MT-9B, fresh set only | 18.3% | 23.3% | 1 | 60 per direction |

Gemma 4 E4B met the bar set above: no worse than Lumen in both directions, on both sets. Its
two sets differed widely on Thai to English (5.0% and 18.3%), which shows how noisy 60 cases
are. ChindaMT often translated the context, and Mango had no documented prompt, so neither
result says what those models can do with a better prompt.

## Final decision

**The owner chose Gemma 4 E4B as the base**, after the second round. The earlier decision to keep
Lumen, recorded below, was replaced by it.

## Earlier decision

**The owner kept Lumen despite the failed rule.** The step's own process called for comparing
Hy-MT2 7B and Qingqiu-MT-9B first, and that comparison was not run, so Lumen has not been
shown to be the best base on this test. The reasoning is that Thai to English is the direction
the real training data can improve, and that the model tested was the untuned 8-bit copy.
The project's pass bar is unchanged and still applies to each direction.

Training now sends history in a system message, matching the format tested here.

**Limits.** 60 cases per direction, short synthetic messages, an 8-bit model and AI judges
only. Opus and Jev agreed on 72.5% of rows. With no context, some "misleading" scores are
unavoidable, because the judge is told a meaning that needs the conversation.
