"""Train two deliberately bad adapters, so the rollout gates can be shown stopping real ones.

flipped writes the wrong gender particles. The golden gate should block it.
bloat says everything twice and keeps the gender notes, so its only fault is its length and only
the canary can catch it.

They learn from the base model's own translations of the pilot suite, altered on purpose. The
golden suite and replay corpus are not used, so the gates test what the adapters learned rather
than what they memorised. These adapters are never released.
"""

import json
import random
import re
import sys
from pathlib import Path

import torch

from serve.mps_server import complete
from train.examples import build_example, chat_messages
from train.feasibility import PROFILES, load, optimiser, stop_id, step, tensors, with_lora

MAN = "The person writing this message is a man."
WOMAN = "The person writing this message is a woman."
EPOCHS = 6
BATCH = 4
LR = 2e-4
FEMALE_ENDINGS = re.compile(r"(ค่ะ|คะ|ดิฉัน)")


def flip(text, man):
    """Rewrite a translation so it uses the other gender's words."""
    if man:
        swapped = text.replace("ครับ", "ค่ะ").replace("ผม", "ดิฉัน")
        return swapped if swapped != text else text.rstrip(" .") + "ค่ะ"
    swapped = FEMALE_ENDINGS.sub(lambda m: "ผม" if m.group(1) == "ดิฉัน" else "ครับ", text)
    return swapped if swapped != text else text.rstrip(" .") + "ครับ"


def double(text):
    return f"{text} {text}"


def cases():
    return [json.loads(line) for line in Path("suites/pilot.jsonl").read_text().splitlines() if line.strip()]


def translate_all(profile, tokenizer, model, plan):
    out = []
    for source, target, notes in plan:
        messages = chat_messages(source, target, (), notes)
        text = complete(profile, tokenizer, model, {"messages": messages, "max_tokens": 120})
        out.append({"source": source, "target_lang": target, "notes": notes, "base": text})
    return out


def train(profile, tokenizer, rows, name):
    _, model, _ = load(profile)
    model = with_lora(profile, model)
    model.train()
    stop = stop_id(profile, tokenizer)
    data = []
    for row in rows:
        messages = chat_messages(row["source"], row["target_lang"], (), row["notes"])
        data.append(build_example(tokenizer, messages, row["target"], stop, template_kwargs=profile.template_kwargs))
    optimizer = optimiser(model, LR)
    rng = random.Random(7)
    for epoch in range(EPOCHS):
        rng.shuffle(data)
        losses = []
        for i in range(0, len(data) - BATCH + 1, BATCH):
            batch = tensors(profile, data[i : i + BATCH], tokenizer.pad_token_id)
            losses.append(step(model, batch, optimizer))
        print(f"{name} epoch {epoch} loss {sum(losses) / len(losses):.4f}", flush=True)
    model.save_pretrained(Path("adapters") / name)
    del model, optimizer
    torch.mps.empty_cache()


def main():
    profile = PROFILES["gemma4e4b"]
    tokenizer, base, _ = load(profile)
    base.eval()
    english = [c["source"] for c in cases() if c["direction"] == "en-th"]
    thai = [c["source"] for c in cases() if c["direction"] == "th-en"]
    gendered = translate_all(profile, tokenizer, base, [(s, "Thai", n) for s in english for n in (MAN, WOMAN)])
    plain = translate_all(profile, tokenizer, base, [(s, "English", "") for s in thai])
    del base
    torch.mps.empty_cache()
    wanted = sys.argv[1:] or ["flipped", "bloat"]
    if "flipped" in wanted:
        train(profile, tokenizer, [{**r, "target": flip(r["base"], r["notes"] == MAN)} for r in gendered], "demo-flipped")
    if "bloat" in wanted:
        train(profile, tokenizer, [{**r, "target": double(r["base"])} for r in gendered + plain], "demo-bloat")
    print("done", flush=True)


if __name__ == "__main__":
    main()
