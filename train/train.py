"""Train the translator from random weights.

python -m train.train --config full --sources scb,mtop,massive --out runs/combined-full
Starting again with the same --out resumes from its latest checkpoint.
"""

import argparse
import collections
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from train.model import CONFIGS, Config, Translator
from train.tokenizer import BOS, EOS, PAD, Tokenizer

BUCKETS = (16, 32, 48, 64, 96, 128)  # fixed shapes, because mx.compile retraces on a new shape


@dataclass(frozen=True)
class Settings:
    batch_tokens: int = 2048
    peak_learning_rate: float = 3e-4
    warmup_steps: int = 4000
    final_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    clip: float = 1.0
    seed: int = 0
    keep_checkpoints: int = 3


def examples_from(rows, tokenizer):
    """Each pair teaches both directions. Examples over the longest bucket are dropped and counted."""
    th, en = tokenizer.marker("<th>"), tokenizer.marker("<en>")
    kept, dropped = [], 0
    for row in rows:
        english, thai = tokenizer.encode(row["en"]), tokenizer.encode(row["th"])
        for example in (([th, *english, EOS], thai), ([en, *thai, EOS], english)):
            if bucket_of(example) is None:
                dropped += 1
            else:
                kept.append(example)
    return kept, dropped


def bucket_of(example):
    needed = max(len(example[0]), len(example[1]) + 1)
    return next((b for b in BUCKETS if needed <= b), None)


def batches_for(examples, batch_tokens, seed, epoch):
    """The batch order is a function of the seed and epoch, so a resumed run sees the same batches."""
    rng = random.Random(f"{seed}-{epoch}")
    groups = collections.defaultdict(list)
    for example in examples:
        groups[bucket_of(example)].append(example)
    batches = []
    for length in BUCKETS:
        group, size = groups[length], max(1, batch_tokens // length)
        rng.shuffle(group)
        batches += [(length, group[i : i + size]) for i in range(0, len(group) - size + 1, size)]
    rng.shuffle(batches)
    return batches


def to_arrays(length, batch):
    def padded(ids):
        return ids + [PAD] * (length - len(ids))

    return (mx.array([padded(source) for source, _ in batch], mx.int32),
            mx.array([padded([BOS, *target]) for _, target in batch], mx.int32),
            mx.array([padded([*target, EOS]) for _, target in batch], mx.int32))


class Trainer:
    def __init__(self, config: Config, examples, out, total_steps, settings=Settings()):
        self.config, self.examples, self.out, self.settings = config, examples, Path(out), settings
        mx.random.seed(settings.seed)
        self.model = Translator(config)
        mx.eval(self.model.parameters())
        warmup = min(settings.warmup_steps, max(1, total_steps // 10))
        schedule = optim.join_schedules(
            [optim.linear_schedule(0.0, settings.peak_learning_rate, warmup),
             optim.cosine_decay(settings.peak_learning_rate, max(1, total_steps - warmup), settings.final_learning_rate)],
            [warmup],
        )
        self.optimizer = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.98], weight_decay=settings.weight_decay)
        self.optimizer.init(self.model.trainable_parameters())
        mx.eval(self.optimizer.state)
        self.step, self.epoch, self.batch = 0, 0, 0
        self.total_steps = total_steps
        self.resume()
        self.train_step, self.state = self.compiled()

    def compiled(self):
        def loss_of(model, source, target_in, target_out):
            logits = model(source, target_in)
            mask = target_out != PAD
            losses = nn.losses.cross_entropy(logits, target_out, label_smoothing=self.settings.label_smoothing, reduction="none")
            return (losses * mask).sum() / mask.sum()

        value_and_grad = nn.value_and_grad(self.model, loss_of)
        state = [self.model.state, self.optimizer.state, mx.random.state]

        def update(source, target_in, target_out):
            loss, grads = value_and_grad(self.model, source, target_in, target_out)
            grads, norm = optim.clip_grad_norm(grads, self.settings.clip)
            self.optimizer.update(self.model, grads)
            return loss, norm

        return mx.compile(update, inputs=state, outputs=state), state

    def checkpoints(self):
        return sorted(p for p in self.out.glob("step-*") if (p / "meta.json").exists())

    def save(self):
        """Written to a temporary folder and renamed, so a crash never leaves half a checkpoint."""
        self.out.mkdir(parents=True, exist_ok=True)
        final, partial = self.out / f"step-{self.step:09d}", self.out / f"partial-{self.step:09d}"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir()
        self.model.save_weights(str(partial / "model.safetensors"))
        mx.save_safetensors(str(partial / "optimizer.safetensors"), dict(tree_flatten(self.optimizer.state)))
        high, low = mx.random.state[0].tolist()
        (partial / "meta.json").write_text(json.dumps({"step": self.step, "epoch": self.epoch, "batch": self.batch, "random": (high << 32) | low, "config": self.config.__dict__}), encoding="utf8")
        shutil.rmtree(final, ignore_errors=True)
        partial.rename(final)
        for old in self.checkpoints()[: -self.settings.keep_checkpoints]:
            shutil.rmtree(old)

    def resume(self):
        saved = self.checkpoints()
        if not saved:
            return
        latest = saved[-1]
        meta = json.loads((latest / "meta.json").read_text(encoding="utf8"))
        if meta["config"] != self.config.__dict__:
            raise SystemExit(f"{latest} was trained with a different model configuration")
        self.model.load_weights(str(latest / "model.safetensors"))
        self.optimizer.state = tree_unflatten(list(mx.load(str(latest / "optimizer.safetensors")).items()))
        mx.random.seed(meta["random"])
        mx.eval(self.model.parameters(), self.optimizer.state, mx.random.state)
        self.step, self.epoch, self.batch = meta["step"], meta["epoch"], meta["batch"]

    def run(self, max_steps, checkpoint_minutes=120, log_every=50):
        """Train until max_steps. Returns this call's loss per step."""
        losses, started, last_save, tokens = [], time.perf_counter(), time.perf_counter(), 0
        self.model.train()
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / "log.jsonl").open("a", encoding="utf8") as log:
            while self.step < max_steps:
                batches = batches_for(self.examples, self.settings.batch_tokens, self.settings.seed, self.epoch)
                if not batches:
                    raise SystemExit("no complete batch: too few examples for batch_tokens")
                while self.batch < len(batches) and self.step < max_steps:
                    arrays = to_arrays(*batches[self.batch])
                    loss, norm = self.train_step(*arrays)
                    mx.eval(loss, norm, self.state)
                    self.step, self.batch = self.step + 1, self.batch + 1
                    losses.append(loss.item())
                    tokens += int((arrays[0] != PAD).sum().item() + (arrays[2] != PAD).sum().item())
                    if self.step % log_every == 0:
                        now = time.perf_counter() - started
                        log.write(json.dumps({"step": self.step, "epoch": self.epoch, "loss": round(losses[-1], 4), "grad_norm": round(norm.item(), 3),
                                              "tokens_per_second": round(tokens / now), "peak_gb": round(mx.get_peak_memory() / 2**30, 2)}) + "\n")
                        log.flush()
                    if time.perf_counter() - last_save > checkpoint_minutes * 60:
                        self.save()
                        last_save = time.perf_counter()
                if self.batch >= len(batches):
                    self.epoch, self.batch = self.epoch + 1, 0
        self.save()
        return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--sources", required=True, help="comma-separated: scb,mtop,massive")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, help="stop early, for benchmarks")
    args = parser.parse_args()
    wanted = set(args.sources.split(","))
    rows = [json.loads(line) for line in Path("data/public/corpus/train.jsonl").read_text(encoding="utf8").splitlines()]
    rows = [r for r in rows if r["source"] in wanted]
    examples, dropped = examples_from(rows, Tokenizer())
    settings = Settings()
    per_epoch = len(batches_for(examples, settings.batch_tokens, settings.seed, 0))
    total = per_epoch * args.epochs
    trainer = Trainer(CONFIGS[args.config], examples, args.out, total, settings)
    print(json.dumps({"pairs": len(rows), "examples": len(examples), "too_long": dropped, "batches_per_epoch": per_epoch, "total_steps": total, "resuming_at": trainer.step}), flush=True)
    losses = trainer.run(args.max_steps or total)
    print(json.dumps({"step": trainer.step, "last_loss": round(losses[-1], 4) if losses else None}))


if __name__ == "__main__":
    main()
