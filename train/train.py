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
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sentencepiece
from mlx.utils import tree_flatten, tree_unflatten

from train.evaluate import Evaluator, average_checkpoints, load_checkpoint, outside_scorer, validation_scorer
from train.model import CONFIGS, Config, Translator
from train.tokenizer import BOS, EOS, PAD, UNK, WORDS, Tokenizer

BUCKETS = (16, 32, 48, 64, 96, 128)  # fixed shapes, because mx.compile retraces on a new shape


@dataclass(frozen=True)
class Settings:
    batch_tokens: int = 2048
    peak_learning_rate: float = 3e-4
    warmup_steps: int = 4000
    final_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    word_dropout: float = 0.0
    r_drop: float = 0.0
    clip: float = 1.0
    seed: int = 0
    keep_checkpoints: int = 3


def examples_from(rows, tokenizer, alpha=0.0, seed=""):
    """Each pair teaches both directions, unless it is marked use "forward_only": then only the
    direction that reads the side a person wrote (its lang) and writes the other is taught.
    Examples over the longest bucket are dropped and counted.
    With alpha, every text gets a sampled segmentation that the seed fixes."""
    th, en = tokenizer.marker("<th>"), tokenizer.marker("<en>")
    kept, dropped = [], 0
    if alpha:
        sentencepiece.set_random_generator_seed(random.Random(seed).getrandbits(31))
    all_english, all_thai = tokenizer.encode([r["en"] for r in rows], alpha), tokenizer.encode([r["th"] for r in rows], alpha)
    for row, english, thai in zip(rows, all_english, all_thai):
        directions = {"en": ([th, *english, EOS], thai), "th": ([en, *thai, EOS], english)}  # keyed by the language read
        for example in ([directions[row["lang"]]] if row.get("use") == "forward_only" else directions.values()):
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


def drop_words(ids, rate):
    """Replaces ordinary pieces with the unknown piece at the given rate. Markers and padding stay."""
    return mx.where((mx.random.uniform(shape=ids.shape) < rate) & (ids >= WORDS), mx.array(UNK, mx.int32), ids)


class Trainer:
    """examples is a list, or a function from the epoch number to that epoch's list."""

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
            mask = target_out != PAD

            def cross_entropy(logits):
                losses = nn.losses.cross_entropy(logits, target_out, label_smoothing=self.settings.label_smoothing, reduction="none")
                return (losses * mask).sum() / mask.sum()

            first = model(source, target_in)
            if not self.settings.r_drop:
                return cross_entropy(first)
            # R-Drop: a second pass with different dropout, and a penalty when the two disagree.
            second = model(source, target_in)
            a, b = nn.log_softmax(first, axis=-1), nn.log_softmax(second, axis=-1)
            disagreement = (((mx.exp(a) - mx.exp(b)) * (a - b)).sum(-1) * mask).sum() / mask.sum() / 2
            return (cross_entropy(first) + cross_entropy(second)) / 2 + self.settings.r_drop * disagreement

        value_and_grad = nn.value_and_grad(self.model, loss_of)
        state = [self.model.state, self.optimizer.state, mx.random.state]

        def update(source, target_in, target_out):
            if self.settings.word_dropout:
                source, target_in = drop_words(source, self.settings.word_dropout), drop_words(target_in, self.settings.word_dropout)
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

    def run(self, max_steps, checkpoint_minutes=120, log_every=50, evaluate=None, evaluate_every=0):
        """Train until max_steps, or until evaluate says to stop. Returns this call's loss per step."""
        losses, started, last_save, tokens = [], time.perf_counter(), time.perf_counter(), 0
        self.model.train()
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / "log.jsonl").open("a", encoding="utf8") as log:
            while self.step < max_steps:
                examples = self.examples(self.epoch) if callable(self.examples) else self.examples
                batches = batches_for(examples, self.settings.batch_tokens, self.settings.seed, self.epoch)
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
                        started, tokens = time.perf_counter(), 0
                    if evaluate and self.step % evaluate_every == 0:
                        if evaluate(self):
                            max_steps = self.step
                        started, tokens, last_save = time.perf_counter(), 0, time.perf_counter()
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-subsets", default="", help="comma-separated subsets to leave out of training")
    parser.add_argument("--max-steps", type=int, help="stop early, for benchmarks")
    # Defaults suit a corpus of a few hundred thousand pairs. Lower dropout and sampling for millions.
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--word-dropout", type=float, default=0.1, help="share of pieces replaced with the unknown piece")
    parser.add_argument("--sampling-alpha", type=float, default=0.2, help="resegment the corpus each epoch, 0 to turn off")
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--r-drop", type=float, default=0.0, help="weight of the R-Drop penalty, 0 for off. Doubles the work per step")
    args = parser.parse_args()
    mx.set_cache_limit(2 * 2**30)  # saves about 8 GB and stops swap growing, at no real cost in speed
    wanted = set(args.sources.split(","))
    rows = [json.loads(line) for line in Path("data/public/corpus/train.jsonl").read_text(encoding="utf8").splitlines()]
    excluded = set(filter(None, args.exclude_subsets.split(",")))
    rows = [r for r in rows if r["source"] in wanted and r["subset"] not in excluded]
    tokenizer = Tokenizer()

    def epoch_examples(epoch):
        return examples_from(rows, tokenizer, alpha=args.sampling_alpha, seed=f"{args.seed}-{epoch}")

    examples, dropped = epoch_examples(0)
    settings = Settings(seed=args.seed, batch_tokens=args.batch_tokens, label_smoothing=args.label_smoothing, word_dropout=args.word_dropout, r_drop=args.r_drop, keep_checkpoints=5)
    per_epoch = len(batches_for(examples, settings.batch_tokens, settings.seed, 0))  # later epochs differ a little when sampling
    total = per_epoch * args.epochs
    trainer = Trainer(replace(CONFIGS[args.config], dropout=args.dropout), (lambda epoch: epoch_examples(epoch)[0]) if args.sampling_alpha else examples, args.out, total, settings)
    validation = [json.loads(line) for line in Path("data/public/corpus/validation.jsonl").read_text(encoding="utf8").splitlines()]
    score = validation_scorer(validation, tokenizer)  # every run is scored on the same rows
    other = Path("data/public/train.jsonl")  # news, textbook and everyday sentences that no run trains on
    if other.exists():
        validation_score, outside = score, outside_scorer(other, tokenizer, {r["en"] for r in rows})
        score = lambda model: {**validation_score(model), **outside(model)}
    print(json.dumps({"pairs": len(rows), "examples": len(examples), "too_long": dropped, "batches_per_epoch": per_epoch, "total_steps": total, "resuming_at": trainer.step}), flush=True)
    losses = trainer.run(args.max_steps or total, evaluate=Evaluator(args.out, score), evaluate_every=max(1, per_epoch // 2))
    print(json.dumps({"step": trainer.step, "last_loss": round(losses[-1], 4) if losses else None}))
    saved = trainer.checkpoints()
    if len(saved) > 1:  # the mean of the last few checkpoints often beats any one of them
        print(json.dumps({"averaged": len(saved), **score(load_checkpoint(average_checkpoints(saved, Path(args.out) / "averaged")))}), flush=True)


if __name__ == "__main__":
    main()
