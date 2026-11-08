"""Can Lumen, compressed to fit, translate and train on an M4 Max with MLX?

  data       write synthetic train and valid sets for mlx_lm.lora
  translate  load the converted model and translate a few lines
  verify     check the trained adapter writes the marker on unseen rows
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from train.examples import lumen_messages
from train.synthetic import MARKER, TRAIN_NUMBERS, UNSEEN_NUMBERS, rows

DATA = Path("runs/mlx-data")
RUNS = Path("runs")
GIB = 1024**3


def write_data(_):
    DATA.mkdir(parents=True, exist_ok=True)
    for name, count, seed in (("train", 400, 2), ("valid", 8, 4)):
        with (DATA / f"{name}.jsonl").open("w", encoding="utf8") as handle:
            for r in rows(count, seed=seed, numbers=TRAIN_NUMBERS):
                messages = lumen_messages(r["source"], r["target_lang"], r["context"])
                messages.append({"role": "assistant", "content": " " + r["target"]})
                handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    return {"written": str(DATA)}


def translate_one(model, tokenizer, source, target_lang, context=()):
    prompt = tokenizer.apply_chat_template(
        lumen_messages(source, target_lang, context), add_generation_prompt=True, tokenize=False
    )
    started = time.time()
    text = generate(model, tokenizer, prompt, max_tokens=80, sampler=make_sampler(temp=0.0))
    return text.strip(), round(time.time() - started, 2)


def translate(args):
    started = time.time()
    model, tokenizer = load(args.model)
    load_seconds = round(time.time() - started, 1)
    lines = [
        ("Could you please send me the address again?", "Thai"),
        ("พรุ่งนี้ฉันจะไปหาหมอตอนเก้าโมงเช้า", "English"),
        ("Table 42 is ready.", "Thai"),
    ]
    out = []
    for source, lang in lines:
        text, seconds = translate_one(model, tokenizer, source, lang)
        out.append({"source": source, "output": text, "seconds": seconds})
    return {"load_seconds": load_seconds, "translations": out, "peak_gib": round(mx.get_peak_memory() / GIB, 2)}


def verify(args):
    data = rows(args.count, seed=3, numbers=UNSEEN_NUMBERS)

    def rate(adapter):
        model, tokenizer = load(args.model, adapter_path=adapter)
        hits, samples = 0, []
        for r in data:
            text, _ = translate_one(model, tokenizer, r["source"], r["target_lang"], r["context"])
            hits += MARKER.strip() in text
            samples.append(text)
        return hits / len(data), samples[:3]

    before, before_samples = rate(None)
    after, after_samples = rate(args.adapter)
    assert before < 0.1, f"untuned model already writes the marker: {before}"
    assert after > 0.8, f"adapter did not learn the marker: {after}"
    return {
        "unseen_rows": len(data),
        "marker_rate_base": before,
        "marker_rate_adapter": after,
        "samples_base": before_samples,
        "samples_adapter": after_samples,
        "peak_gib": round(mx.get_peak_memory() / GIB, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--name", default="lumen8")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data")
    sub.add_parser("translate")
    p = sub.add_parser("verify")
    p.add_argument("--adapter", required=True)
    p.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    result = {"data": write_data, "translate": translate, "verify": verify}[args.command](args)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{args.name}-{args.command}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
