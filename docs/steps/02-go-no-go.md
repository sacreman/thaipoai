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

Not yet run.
