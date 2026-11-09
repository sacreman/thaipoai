# Step 18: frozen candidate, untuned Gemma with retrieved examples

9 November 2026. `candidates/gemma-examples-v1` pins what was measured: Gemma 4 E4B at a fixed
revision with no adapter, the chat prompt, and three retrieved examples per message from 3,870
synthetic pairs. `tests/test_frozen_candidate.py` fails if the model, prompt, pairs or retrieval
change.

## What did not work

Two LoRA adapters were trained with the same recipe, and both made the model worse.

| Training data | Suite | Untuned wrong | Adapter wrong | Fixed | Broken |
|---|---|---|---|---|---|
| Public sentence pairs, 8 hours | Pilot, 240 | 62 | 92 | 4 | 34 |
| Synthetic chat pairs, 2 hours | Pilot, 240 | 63 | 69 | 13 | 19 |
| Synthetic chat pairs, 2 hours | Messy dev, 400 | 70 | 91 | 14 | 35 |

Fine-tuning stays off until there is data it can improve on. A split setup, with examples for
English to Thai only, was proposed from one suite and failed on a fresh one, which is why it has
no place here.

## Results

Two suites of messy, texting-style cases: 598 cases, each with and without context, 1,196
translations. Jev judged each translation three times and the majority counted. DeepSeek ran
without the production glossary.

| Slice | DeepSeek wrong | Candidate wrong | Gap | 95% upper bound |
|---|---|---|---|---|
| All | 185 (15.5%) | 194 (16.2%) | +0.75 | 2.28 |
| English to Thai | 80 (13.4%) | 80 (13.4%) | 0.00 | 1.70 |
| Thai to English | 105 (17.5%) | 114 (19.0%) | +1.50 | 4.03 |
| With correct context | 77 (12.9%) | 87 (14.5%) | +1.67 | 3.84 |
| No context | 108 (18.1%) | 107 (17.9%) | -0.17 | 1.98 |

Against untuned Gemma the examples fixed 41 and broke 29 (one-sided p = 0.09). The benefit is
without context, 28 fixed and 13 broken, and with context they do nothing useful, 13 fixed and
16 broken.

## How far this goes

- This is not a pass of the M2 gate. That gate needs an upper bound of 1.5 points or less, and
  neither direction reaches it here. It also compares against production on held-out chats,
  which has not been run.
- Every message was drafted by DeepSeek and the only judge is Jev. Nobody has checked either.
- No real chats were used, for training or for testing.
- Retrieval is measured but not wired into the gateway.
