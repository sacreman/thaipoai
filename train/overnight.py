"""Train the LoRA adapter for a fixed amount of GPU time, surviving a crash.

The checkpoint names the step it stopped at, and is written to a new directory that replaces
"latest" only once it is complete. Logs print numbers only, never text.
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from train.examples import build_example, chat_messages
from train.feasibility import PROFILES, load, memory, optimiser, stop_id, step, tensors, with_lora

BATCH = 4
LR = 2e-4
SEED = 11


def rows_of(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def encode(profile, tokenizer, rows):
    stop, out, dropped = stop_id(profile, tokenizer), [], 0
    for row in rows:
        try:
            messages = chat_messages(row["source"], row["target_lang"], row.get("context", ()))
            out.append(build_example(tokenizer, messages, row["target"], stop, template_kwargs=profile.template_kwargs))
        except ValueError:
            dropped += 1  # too long for max_len, and a truncated target would teach bad stopping
    print(json.dumps({"encoded": len(out), "dropped": dropped}), flush=True)
    return out


def order(count, epoch):
    indices = list(range(count))
    random.Random(SEED + epoch).shuffle(indices)
    return indices


def validation_loss(profile, tokenizer, model, data):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(data) - BATCH + 1, BATCH):
            losses.append(model(**tensors(profile, data[i : i + BATCH], tokenizer.pad_token_id)).loss.item())
    model.train()
    torch.mps.empty_cache()
    return sum(losses) / len(losses)


def save(out, model, optimizer, state):
    work = out / f"tmp-{state['step']}"
    shutil.rmtree(work, ignore_errors=True)
    model.save_pretrained(work)
    torch.save(optimizer.state_dict(), work / "optimizer.pt")
    (work / "state.json").write_text(json.dumps(state))
    latest = out / "latest"
    previous = out / "previous"
    shutil.rmtree(previous, ignore_errors=True)
    if latest.exists():
        latest.rename(previous)
    work.rename(latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--checkpoint-minutes", type=float, default=30)
    parser.add_argument("--train", default="data/public/train.jsonl")
    parser.add_argument("--validation", default="data/public/validation.jsonl")
    parser.add_argument("--out", default="adapters/public-night")
    parser.add_argument("--max-rows", type=int, default=0, help="limit for a smoke test")
    args = parser.parse_args()
    profile = PROFILES["gemma4e4b"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model, _ = load(profile)
    model = with_lora(profile, model)
    model.train()
    train_rows = rows_of(args.train)
    validation_rows = rows_of(args.validation)[:1000]
    if args.max_rows:
        train_rows, validation_rows = train_rows[: args.max_rows], validation_rows[:40]
    data = encode(profile, tokenizer, train_rows)
    validation = encode(profile, tokenizer, validation_rows)
    optimizer = optimiser(model, LR)
    state = {"step": 0, "seconds": 0.0, "best": None}
    latest = out / "latest"
    if (latest / "state.json").exists():
        state = json.loads((latest / "state.json").read_text())
        set_peft_model_state_dict(model, load_file(latest / "adapter_model.safetensors"))
        optimizer.load_state_dict(torch.load(latest / "optimizer.pt"))
        print(json.dumps({"resumed_at_step": state["step"]}), flush=True)
    steps_per_epoch = len(data) // BATCH
    limit = args.hours * 3600
    last_save = time.time()
    losses = []
    while state["seconds"] < limit:
        epoch, position = divmod(state["step"], steps_per_epoch)
        indices = order(len(data), epoch)[position * BATCH : (position + 1) * BATCH]
        started = time.time()
        losses.append(step(model, tensors(profile, [data[i] for i in indices], tokenizer.pad_token_id), optimizer))
        state["seconds"] += time.time() - started
        state["step"] += 1
        if state["step"] % 50 == 0:
            print(json.dumps({"step": state["step"], "epoch": round(state["step"] / steps_per_epoch, 3), "loss": round(sum(losses) / len(losses), 4), "hours": round(state["seconds"] / 3600, 2)}), flush=True)
            losses = []
        if time.time() - last_save >= args.checkpoint_minutes * 60 or state["seconds"] >= limit:
            value = validation_loss(profile, tokenizer, model, validation)
            improved = state["best"] is None or value < state["best"]
            state["best"] = value if improved else state["best"]
            save(out, model, optimizer, state)
            if improved:
                shutil.rmtree(out / "best", ignore_errors=True)
                shutil.copytree(out / "latest", out / "best", ignore=shutil.ignore_patterns("optimizer.pt"))
            print(json.dumps({"step": state["step"], "validation_loss": round(value, 4), "best": improved, "memory": memory()}), flush=True)
            last_save = time.time()
    print(json.dumps({"finished_step": state["step"], "best_validation": state["best"]}), flush=True)


if __name__ == "__main__":
    main()
