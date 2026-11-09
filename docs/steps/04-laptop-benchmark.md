# Step 4: what can the laptop sustain?

9 November 2026. M4 Max, 36 GB. `train/benchmark.py` timed 500 real training steps per run on the
`corpus-v1` release, with batches in the order training uses. Chrome was open throughout, using a full
CPU core and part of the GPU, so a clean machine may do better.

| Run | Tokens per second | Peak footprint | Swap change | Estimated hours per epoch |
|---|---:|---:|---:|---:|
| Baseline, 75M | 8,783 | 10.7 GB | -0.07 GB | 0.47 |
| Full, 297M, first run | 2,270 | 20.95 GB | +0.69 GB | 1.64 |
| Full, 297M, control | 2,545 | 20.94 GB | +1.06 GB | 1.63 |
| Full, 297M, cache capped at 2 GB | 2,660 | 12.3 GB | -0.10 GB | 1.55 |

An epoch is 6,531 batches and covers both directions of 298,930 pairs. Eight epochs is about 12 to 13
hours for the full model and about 4 for the baseline. Those figures leave out evaluation time and
checkpoints, which take about 3 seconds each. Speed scales with size: the baseline has a quarter of the
parameters and runs about 3.5 times faster.

GPU use was 98 to 99% at the median, and no thermal warning appeared. Peak MLX memory was 9.8 GB for the
full model and 4.5 GB for the baseline. Loss fell from about 11.9 to 7.0 across the 500 steps.

## What we learned

- **The cache cap saves memory, not time.** Capping MLX's buffer cache at 2 GB saved 8.6 GB and stopped
  swap growing, and speed changed by about 4%. The trainer now sets it.
- **A slowdown did not repeat.** In the first run the full model fell from 3,912 to 1,512 tokens per
  second and recovered, far more than its batch mix explains. Neither later run showed it, and the
  cache was not the cause. It is unexplained, and the likeliest cause is other work on the machine.

## Limits

No run lasted more than about eight minutes, so sustained speed over many hours, and heat, are untested.
The 128-token bucket never appeared in 500 steps. Real training is the first long test.
