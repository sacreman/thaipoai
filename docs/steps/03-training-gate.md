# Step 3: does training work?

9 November 2026. `tests/test_training.py` checks the trainer, and each check can fail.

| Check | Result |
|---|---|
| Learns a tiny dataset | Final loss under 0.05, and greedy decoding reproduces every target |
| Every weight updates | No parameter unchanged after 10 steps, including the shared embedding |
| Checkpoint restores state | Weights, optimizer, random key and batch position match bit for bit |
| Resume continues the run | Every loss within 0.1% of an uninterrupted control run |

Removing the random-state restore, the optimizer restore or the batch position each makes the resume
test fail.

## What we learned

- **Identical runs are not bit-identical.** Losses differ by about 1e-7 at step 5 with no interruption,
  so only the restored state can be exact. The first version of the test wrongly demanded more.
- **The default embedding scale was a bug.** With a shared output projection, 20 steps on real data
  gave a loss of 15.7, worse than guessing at 10.4. Initialising at `dim ** -0.5` fixed it.

## Real-data smoke run

The full model on 298,930 pairs, 597,860 examples, 6,531 batches per epoch. 150 steps on a machine with
the GPU already busy: loss 9.75 at step 50 and 8.89 at step 150, peak MLX memory 9.75 GB. Throughput
fell from 3,821 to 1,998 tokens per second across the run, so step 4 must find out why before any
estimate of a full run is trusted.
