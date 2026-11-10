"""Score checkpoints during training, and translate a suite with a finished one.

Checkpoints are scored with chrF on a fixed sample of the validation split, per direction and per
source. The test split stays untouched. To translate a suite for judging:

python -m train.evaluate --checkpoint runs/combined-baseline/best --suite suites/pilot.jsonl --out runs/pilot.jsonl
"""

import argparse
import collections
import json
import random
import shutil
import time
from pathlib import Path

import mlx.core as mx

from train.model import Config, Translator
from train.tokenizer import EOS, Tokenizer

SAMPLE_ROWS = 2000


def character_ngrams(text, n):
    text = "".join(text.split())
    return collections.Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf(hypotheses, references, beta=2, order=6):
    """chrF as sacreBLEU computes it: counts pooled over the corpus, F-beta averaged over n-gram sizes."""
    total, used = 0.0, 0
    for n in range(1, order + 1):
        hypothesis_total, reference_total, matches = 0, 0, 0
        for hypothesis, reference in zip(hypotheses, references):
            a, b = character_ngrams(hypothesis, n), character_ngrams(reference, n)
            hypothesis_total += sum(a.values())
            reference_total += sum(b.values())
            matches += sum((a & b).values())
        if reference_total:
            used += 1
            precision = matches / hypothesis_total if hypothesis_total else 0.0
            recall = matches / reference_total
            total += (1 + beta**2) * precision * recall / max(beta**2 * precision + recall, 1e-16)
    return 100 * total / used if used else 0.0


def encoded(tokenizer, text, target):
    return [tokenizer.marker("<th>" if target == "th" else "<en>"), *tokenizer.encode(text), EOS]


def validation_scorer(rows, tokenizer):
    """A function scoring a model on a fixed sample of validation rows, each translated both ways."""
    sample = random.Random(5).sample(rows, min(SAMPLE_ROWS, len(rows)))
    sources = [encoded(tokenizer, r["en"], "th") for r in sample] + [encoded(tokenizer, r["th"], "en") for r in sample]
    references = [r["th"] for r in sample] + [r["en"] for r in sample]
    groups = collections.defaultdict(list)
    for index, r in enumerate(sample):
        groups[f"{r['source']}/{r['subset']}"] += [index, index + len(sample)]

    def score(model):
        hypotheses = [tokenizer.decode(ids) for ids in model.translate(sources)]
        half = len(sample)
        en_th, th_en = chrf(hypotheses[:half], references[:half]), chrf(hypotheses[half:], references[half:])
        by_source = {name: round(chrf([hypotheses[i] for i in idx], [references[i] for i in idx]), 2) for name, idx in sorted(groups.items())}
        return {"chrf": round((en_th + th_en) / 2, 2), "chrf_en_th": round(en_th, 2), "chrf_th_en": round(th_en, 2), "by_source": by_source}

    return score


def outside_pairs(path, tokenizer, trained_english, per_origin=300, longest=100):
    """The same few hundred (origin, English, Thai) pairs from each origin in the file every time,
    leaving out pairs whose English is in the training data and pairs too long for the model."""
    by_origin = collections.defaultdict(dict)
    for line in Path(path).read_text(encoding="utf8").splitlines():
        r = json.loads(line)
        if r["target_lang"] == "Thai" and r["source"] not in trained_english:
            by_origin[r["origin"]][r["source"]] = r["target"]
    chosen = []
    for origin, pairs in sorted(by_origin.items()):
        fits = [(en, th) for en, th in sorted(pairs.items()) if max(len(tokenizer.encode(en)), len(tokenizer.encode(th))) <= longest]
        chosen += [(origin, en, th) for en, th in random.Random(5).sample(fits, min(per_origin, len(fits)))]
    return chosen


def outside_scorer(path, tokenizer, trained_english):
    """Scores a model on text unlike its training data (news, a textbook, everyday sentences), both
    ways. Reported beside the validation score and never used to pick a checkpoint."""
    sources, references, groups = [], [], {}
    for origin, en, th in outside_pairs(path, tokenizer, trained_english):
        groups.setdefault(origin, []).extend((len(sources), len(sources) + 1))
        sources += [encoded(tokenizer, en, "th"), encoded(tokenizer, th, "en")]
        references += [th, en]

    def score(model):
        hypotheses = [tokenizer.decode(ids) for ids in model.translate(sources)]
        pick = lambda items, indexes: [items[i] for i in indexes]
        scores = {f"outside_{origin}": round(chrf(pick(hypotheses, idx), pick(references, idx)), 2) for origin, idx in groups.items()}
        return {"outside": round(sum(scores.values()) / max(1, len(scores)), 2), **scores}

    return score


class Evaluator:
    """Called at each scoring point. Keeps the best checkpoint and says when progress has stopped."""

    def __init__(self, out, score, patience=3, min_gain=0.1):
        self.out, self.score, self.patience, self.min_gain = Path(out), score, patience, min_gain
        self.file = self.out / "eval.jsonl"
        self.history = [json.loads(line) for line in self.file.read_text(encoding="utf8").splitlines()] if self.file.exists() else []

    def scores_without_gain(self):
        best, waited = float("-inf"), 0
        for report in self.history:
            if report["chrf"] >= best + self.min_gain:
                best, waited = report["chrf"], 0
            else:
                waited += 1
        return waited

    def __call__(self, trainer):
        started = time.perf_counter()
        report = {"step": trainer.step, "epoch": trainer.epoch, **self.score(trainer.model)}
        report["seconds"] = round(time.perf_counter() - started, 1)
        self.history = [r for r in self.history if r["step"] < trainer.step] + [report]  # a resumed run may repeat a step
        self.out.mkdir(parents=True, exist_ok=True)
        self.file.write_text("".join(json.dumps(r) + "\n" for r in self.history), encoding="utf8")
        trainer.save()
        waited = self.scores_without_gain()
        if waited == 0:
            partial, best = self.out / "best-partial", self.out / "best"
            shutil.rmtree(partial, ignore_errors=True)
            shutil.copytree(trainer.checkpoints()[-1], partial)
            shutil.rmtree(best, ignore_errors=True)
            partial.rename(best)
        print(json.dumps(report), flush=True)
        return waited >= self.patience


def load_checkpoint(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf8"))
    model = Translator(Config(**meta["config"]))
    model.load_weights(str(path / "model.safetensors"))
    mx.eval(model.parameters())
    return model


def average_checkpoints(paths, out):
    """Writes the mean of the checkpoints' weights as a checkpoint load_checkpoint can read."""
    total = None
    for path in paths:
        weights = mx.load(str(Path(path) / "model.safetensors"))
        total = weights if total is None else {name: total[name] + value for name, value in weights.items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), {name: value / len(paths) for name, value in total.items()})
    shutil.copy(Path(paths[-1]) / "meta.json", out / "meta.json")
    return out


def translate_suite(checkpoint, suite, out, system):
    """Writes the same result rows as cmd/suite, so evaluation/compare_systems.py can judge them."""
    cases = [json.loads(line) for line in Path(suite).read_text(encoding="utf8").splitlines() if line.strip()]
    tokenizer, model = Tokenizer(), load_checkpoint(checkpoint)
    outputs = model.translate([encoded(tokenizer, c["source"], c["direction"][-2:]) for c in cases])
    rows = [{"id": c["id"], "system": system, "variant": "none", "direction": c["direction"], "category": c.get("category", ""), "output": tokenizer.decode(ids)} for c, ids in zip(cases, outputs)]
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    print(f"{len(rows)} translations written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--system", default="from-scratch")
    args = parser.parse_args()
    translate_suite(args.checkpoint, args.suite, args.out, args.system)
