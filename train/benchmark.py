"""Measure sustained training speed on this machine before estimating a full run.

python -m train.benchmark --config full --steps 500 --out runs/benchmark-full

Each step is timed with its batch size, so a real slowdown can be told apart from a stretch of longer
sentences. Prints and saves counts only, never text.
"""

import argparse
import collections
import ctypes
import json
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

import mlx.core as mx

from train.model import CONFIGS
from train.tokenizer import PAD, Tokenizer
from train.train import Settings, Trainer, batches_for, examples_from, to_arrays

EPOCHS = 8


def footprint_gb():
    """Process footprint including Metal buffers, which ru_maxrss misses."""
    info = (ctypes.c_uint64 * 64)()
    ctypes.CDLL("libproc.dylib").proc_pid_rusage(os.getpid(), 2, info)
    return round(info[9] / 2**30, 2)


def gpu_percent():
    output = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    match = re.search(r'"Device Utilization %"=(\d+)', output)
    return int(match.group(1)) if match else None


def thermal_warning():
    output = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    return "No thermal warning level has been recorded" not in output or "CPU_Speed_Limit" in output and "CPU_Speed_Limit \t= 100" not in output


def swap_used_gb():
    output = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    match = re.search(r"used = ([\d.]+)M", output)
    return round(float(match.group(1)) / 1024, 2) if match else None


class Sampler(threading.Thread):
    """Records GPU busy percentage, memory and heat every few seconds while training runs."""

    def __init__(self, interval=5.0):
        super().__init__(daemon=True)
        self.interval, self.samples, self.stopping = interval, [], threading.Event()

    def run(self):
        while not self.stopping.wait(self.interval):
            self.samples.append({"gpu": gpu_percent(), "footprint_gb": footprint_gb(), "swap_gb": swap_used_gb(), "thermal_warning": thermal_warning()})


def summarise(records, epoch_batches):
    """records hold bucket, ms, tokens and first (a batch shape's first step, which includes compiling)."""
    warm = [r for r in records if not r["first"]]
    by_bucket = collections.defaultdict(list)
    for r in warm:
        by_bucket[r["bucket"]].append(r)
    median_ms = {b: statistics.median(r["ms"] for r in rows) for b, rows in by_bucket.items()}
    buckets = {str(b): {"steps": len(rows), "median_ms": round(median_ms[b], 1), "tokens_per_second": round(1000 * sum(r["tokens"] for r in rows) / sum(r["ms"] for r in rows))}
               for b, rows in sorted(by_bucket.items())}
    windows = []
    for start in range(0, len(warm), 100):
        rows = warm[start : start + 100]
        if len(rows) == 100:
            expected = sum(median_ms[r["bucket"]] for r in rows)
            windows.append({"steps": start + 100, "tokens_per_second": round(1000 * sum(r["tokens"] for r in rows) / sum(r["ms"] for r in rows)),
                            "time_vs_expected": round(sum(r["ms"] for r in rows) / expected, 3)})
    missing = sorted(b for b in epoch_batches if b not in median_ms)
    mean_ms = {b: statistics.mean(r["ms"] for r in rows) for b, rows in by_bucket.items()}  # means, so slow spells count
    epoch_ms = sum(count * mean_ms[b] for b, count in epoch_batches.items() if b in mean_ms)
    return {"steps_timed": len(warm), "compile_steps_excluded": len(records) - len(warm),
            "tokens_per_second": round(1000 * sum(r["tokens"] for r in warm) / sum(r["ms"] for r in warm)) if warm else None,
            "buckets": buckets, "windows": windows, "buckets_never_seen": missing,
            "estimated_hours_per_epoch": round(epoch_ms / 3_600_000, 2), f"estimated_hours_for_{EPOCHS}_epochs": round(EPOCHS * epoch_ms / 3_600_000, 2)}


def benchmark(config, steps, out, sources, cache_gb=None):
    if cache_gb is not None:
        mx.set_cache_limit(int(cache_gb * 2**30))
    rows = [json.loads(line) for line in Path("data/public/corpus/train.jsonl").read_text(encoding="utf8").splitlines()]
    examples, _ = examples_from([r for r in rows if r["source"] in sources], Tokenizer())
    settings = Settings()
    batches = batches_for(examples, settings.batch_tokens, settings.seed, 0)
    epoch_batches = collections.Counter(length for length, _ in batches)
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    trainer = Trainer(CONFIGS[config], examples, out / "checkpoints", len(batches) * EPOCHS, settings)
    trainer.model.train()
    mx.reset_peak_memory()
    sampler = Sampler()
    sampler.start()
    records, seen = [], set()
    with (out / "steps.jsonl").open("w", encoding="utf8") as log:
        for length, batch in batches[:steps]:
            started = time.perf_counter()
            arrays = to_arrays(length, batch)
            loss, norm = trainer.train_step(*arrays)
            mx.eval(loss, norm, trainer.state)
            trainer.step += 1
            record = {"bucket": length, "ms": 1000 * (time.perf_counter() - started), "tokens": int((arrays[0] != PAD).sum().item() + (arrays[2] != PAD).sum().item()),
                      "loss": loss.item(), "first": arrays[0].shape not in seen}
            seen.add(arrays[0].shape)
            records.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
    sampler.stopping.set()
    sampler.join()
    started = time.perf_counter()
    trainer.save()
    save_seconds = time.perf_counter() - started
    busy = [s["gpu"] for s in sampler.samples if s["gpu"] is not None]
    report = {"config": config, "sources": sorted(sources), "cache_limit_gb": cache_gb, **summarise(records, epoch_batches), "batches_per_epoch": len(batches),
              "first_loss": round(records[0]["loss"], 3), "last_loss": round(records[-1]["loss"], 3),
              "peak_mlx_gb": round(mx.get_peak_memory() / 2**30, 2), "peak_footprint_gb": max(s["footprint_gb"] for s in sampler.samples),
              "swap_gb_start_end": [sampler.samples[0]["swap_gb"], sampler.samples[-1]["swap_gb"]],
              "gpu_busy_median_percent": statistics.median(busy) if busy else None, "gpu_busy_min_percent": min(busy) if busy else None,
              "thermal_warning_seen": any(s["thermal_warning"] for s in sampler.samples), "checkpoint_seconds": round(save_seconds, 1)}
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", default="scb,mtop,massive")
    parser.add_argument("--cache-gb", type=float, help="cap MLX's buffer cache")
    args = parser.parse_args()
    benchmark(args.config, args.steps, args.out, set(args.sources.split(",")), args.cache_gb)
