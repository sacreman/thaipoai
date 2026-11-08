# Step 1: can a translation model train on an M4 Max?

8 November 2026. Apple M4 Max, 32-core GPU, 36 GB unified memory. PyTorch 2.15 on MPS,
Transformers 5.22, PEFT 0.21. Models pinned to exact revisions in `train/feasibility.py`.

**Answer: yes. Hy-MT2 7B and Qingqiu-MT-9B both train comfortably overnight. TranslateGemma
12B trains, but it is far slower and leaves the machine short of memory.**

## What was tested

All data was synthetic. Every training target ended with a marker, ` #ok`, that the
untuned model never writes. Writing it on unseen rows proves training changed the
model's behaviour. These adapters test the machinery only and are never released.

All used a LoRA adapter (rank 64, alpha 128), bf16 weights, gradient checkpointing and
AdamW at 2e-4, with four turns of context. Only the attention projections are trained.

## Results

| | Hy-MT2 7B | Qingqiu-MT-9B | TranslateGemma 12B |
|---|---|---|---|
| Weights on disk | 16.1 GB | 18.8 GB | 24.4 GB |
| GPU memory in use, translating | 14.1 GB | 16.9 GB | 23.9 GB |
| Parameters | 7.56B | 9.01B (text only) | 11.86B |
| Trainable (adapter only) | 54.5M | 59.8M | 84.9M |
| Speed, batch 4 | 183.5 tokens/s | 145.2 tokens/s | 16.8 tokens/s |
| Average row | 131 tokens | 163 tokens | 186 tokens |
| Rows per hour | 5,041 | 3,208 | 327 |
| 20,000 rows | about 4 hours | about 6.2 hours | about 61 hours |
| Peak GPU driver memory (between steps) | 15.9 GB | 19.2 GB | 24.2 GB |
| GPU memory limit used | 27 GB | 27 GB | about 35 GB |
| Training steps measured | 100 | 100 | 20 |
| Marker on 20 unseen rows, untuned | 0% | 0% | 0% |
| Marker on 20 unseen rows, adapter | 100% | 100% | 100% |
| Adapter checksum matched in a fresh process | Yes | Yes | Yes |

Hy-MT2 and Qingqiu were timed on an idle machine. The 12B run used about 12 GB of swap. The
first 7B timing, 91 tokens/s, ran alongside a 24 GB download, so a shared machine can halve
throughput. Rows differ in length because each model's prompt differs. Real rows will be
longer than these, so real throughput will be lower.

## What went wrong

- **Learning rate.** At 1e-3 the 7B's loss rose from 3.11 to 4.72. At Tencent's 2e-4 it
  collapsed to about 0. An overfit test caught this in minutes.
- **12B out of memory.** At batch 4 under PyTorch's default 27 GB limit, it failed on the
  second step. Removing the unused vision tower fixed batch 2. Raising the limit to about
  35 GB allowed batch 4. We changed both, so we cannot say which mattered more at batch 4.
- **A busy machine costs real speed.** With a game open, 20 steps of the 12B at batch 2
  took about 25 minutes. Close heavy apps before training.
- **MPS watermarks.** Setting only the high watermark to 1.0 crashes, because the default
  low watermark is 1.4. Set both.
- **Qingqiu did not stop.** Its repository ships no generation settings, so it ran on after
  translating. Passing the stop token explicitly fixed it. Its thinking mode must also be
  turned off, or stray thinking text would corrupt the labels.
- **Qingqiu's layers are different.** It mixes full attention and linear attention layers,
  so the LoRA targets had to be worked out from the model itself.

## What the models showed

A few test lines are not evidence of quality, but they are useful:

- Hy-MT2 7B and TranslateGemma 12B read "Table 42 is ready" as a spreadsheet table, without
  context. Qingqiu got it right (โต๊ะ).
- TranslateGemma 12B and Qingqiu wrote gender-hedged forms with a literal slash (`คะ/ครับ`,
  `ผม/ดิฉัน`), which reads badly in a chat.
- **Given a chat context in its text field, untuned TranslateGemma translated the context
  instead of the message.** Its card says it has no official context format, and this
  shows the practical effect.

## Lumen at 8-bit

Lumen (14.77B parameters, 29.5 GB at 16-bit) cannot fit at full precision, so it was
converted to 8-bit with MLX and trained with `mlx_lm.lora` rather than PyTorch. Same adapter
rank, same 40-layer coverage, batch 4.

| | Lumen 8-bit (MLX) |
|---|---|
| Size on disk | 15 GB |
| Peak memory, reloading and translating | 15.2 GB |
| Trainable (adapter only) | 83.9M of 14.77B |
| 100 steps at batch 4 | 15 minutes, including load |
| Rows per hour | about 1,600 |
| 20,000 rows | about 12.5 hours |
| Swap during training | None added |
| Marker on 20 unseen rows, untuned | 0% |
| Marker on 20 unseen rows, adapter | 100% |

- **Learning rate again.** At 2e-4 the loss fell to 1.07 and then blew up to 10.46, and the
  adapter learned nothing. `mlx_lm.lora` applies no gradient clipping, unlike our PyTorch
  loop. At 5e-5 the loss collapsed to 0 by step 40.
- **The speed figure is not comparable with the tokens/s above.** MLX's table counts only
  trained tokens, so compare rows per hour. Those were not on identical prompts.
- **Not yet tested:** applying this adapter to the 16-bit weights, which is what serving
  would use. MLX adapters are not PEFT adapters, so a conversion step would be needed for
  vLLM. Thai-to-English quality and chat context are also untested for Lumen.

## Gemma 4 E4B

Added after step 2 chose it as the base, and tested the same way on an idle machine.

| | Gemma 4 E4B |
|---|---|
| Weights on disk | 16 GB |
| Parameters | 7.98B |
| Trainable (adapter only) | 36.3M |
| Speed, batch 4 | 195 tokens/s |
| Average row | 195 tokens |
| Rows per hour | 3,607 |
| 20,000 rows | about 5.5 hours |
| Peak GPU driver memory (between steps) | 16.4 GB |
| Marker on 20 unseen rows, untuned | 0% |
| Marker on 20 unseen rows, adapter | 100% |
| Adapter checksum matched in a fresh process | Yes |

## Decision, before step 2

Hy-MT2 7B and Qingqiu-MT-9B both fit a night's training, and both are Apache-2.0. Lumen also
trains overnight at 8-bit, with an extra conversion step before serving. Step 2 decides
between them on Thai meaning. TranslateGemma 12B is not practical to run repeatedly on this
machine.
