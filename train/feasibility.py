"""Step 1: can these models train on an M4 Max? Each command writes runs/<model>-<command>.json.

  smoke   load the model, translate a few lines, record load time and memory
  overfit train on one small batch until loss collapses
  train   a timed run at realistic prompt size, saves an adapter
  verify  fresh process: reload the adapter, check it learned and its checksum matches
"""

import argparse
import hashlib
import json
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from train.examples import build_example, collate, gemma_messages, hy_messages, qwen_messages
from train.synthetic import MARKER, TRAIN_NUMBERS, UNSEEN_NUMBERS, rows

DEVICE = "mps"
RUNS = Path("runs")
GIB = 1024**3


@dataclass
class Profile:
    name: str
    model: str
    revision: str
    auto_class: type
    messages: Callable
    lora_targets: object
    stop_token: str | None  # None means the tokenizer's own eos token
    needs_token_types: bool = False
    template_kwargs: dict = field(default_factory=dict)


PROFILES = {
    "qingqiu9b": Profile(
        name="qingqiu9b",
        model="WPS-Qingqiu/Qingqiu-MT-9B",
        revision="82385ebe32388d1fb0f406a288a1bd66d464bea8",
        auto_class=AutoModelForCausalLM,
        messages=qwen_messages,
        # Qwen3.5 mixes full-attention and linear-attention layers, each with its own names.
        lora_targets=r".*(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))",
        stop_token="<|im_end|>",
        template_kwargs={"enable_thinking": False},
    ),
    "hy7b": Profile(
        name="hy7b",
        model="tencent/Hy-MT2-7B",
        revision="9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223",
        auto_class=AutoModelForCausalLM,
        messages=hy_messages,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        stop_token=None,
    ),
    "gemma12b": Profile(
        name="gemma12b",
        model="google/translategemma-12b-it",
        revision="d1b225e1caa17f1ddc7e62065d8637d0923f34e2",
        auto_class=AutoModelForImageTextToText,
        messages=gemma_messages,
        # The vision tower has layers with the same names and must stay frozen.
        lora_targets=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)",
        stop_token="<end_of_turn>",
        needs_token_types=True,
    ),
}


def memory():
    return {
        "mps_allocated_gib": round(torch.mps.current_allocated_memory() / GIB, 2),
        "mps_driver_gib": round(torch.mps.driver_allocated_memory() / GIB, 2),
        "process_peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB, 2),
    }


def load(profile):
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(profile.model, revision=profile.revision)
    model = profile.auto_class.from_pretrained(profile.model, revision=profile.revision, dtype=torch.bfloat16)
    if profile.needs_token_types:
        model.model.vision_tower = None  # text translation never uses it, and memory is tight
    model.to(DEVICE)
    return tokenizer, model, round(time.time() - started, 1)


def stop_id(profile, tokenizer):
    if profile.stop_token is None:
        return tokenizer.eos_token_id
    return tokenizer.convert_tokens_to_ids(profile.stop_token)


def generate(profile, tokenizer, model, source, target_lang, context=()):
    inputs = tokenizer.apply_chat_template(
        profile.messages(source, target_lang, context),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **profile.template_kwargs,
    ).to(DEVICE)
    started = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=stop_id(profile, tokenizer),  # not every repo ships generation settings
        )
    new = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new, skip_special_tokens=True), len(new), time.time() - started


def with_lora(profile, model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=profile.lora_targets,
    )
    model.config.use_cache = False
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def trainable_report(model):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, _ in trainable), "only LoRA weights may train"
    assert not any("vision" in n for n, _ in trainable), "the vision tower must stay frozen"
    return {
        "trainable_tensors": len(trainable),
        "trainable_params": sum(p.numel() for _, p in trainable),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def tensors(profile, batch, pad_id):
    out = {k: torch.tensor(v, device=DEVICE) for k, v in collate(batch, pad_id).items()}
    if profile.needs_token_types:
        out["token_type_ids"] = torch.zeros_like(out["input_ids"])
    return out


def step(model, batch, optimizer):
    loss = model(**batch).loss
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    grad = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    assert grad > 0, "no gradient reached the adapter"
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.mps.empty_cache()
    return loss.item()


def encoded(profile, tokenizer, data):
    stop = stop_id(profile, tokenizer)
    return [
        build_example(
            tokenizer,
            profile.messages(r["source"], r["target_lang"], r["context"]),
            r["target"],
            stop,
            template_kwargs=profile.template_kwargs,
        )
        for r in data
    ]


def optimiser(model, lr):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)


def adapter_dir(profile):
    return Path("adapters") / profile.name


def smoke(profile, _):
    tokenizer, model, load_seconds = load(profile)
    model.eval()
    lines = [
        ("Could you please send me the address again?", "Thai"),
        ("พรุ่งนี้ฉันจะไปหาหมอตอนเก้าโมงเช้า", "English"),
        ("Table 42 is ready.", "Thai"),
    ]
    out = []
    for source, lang in lines:
        text, count, seconds = generate(profile, tokenizer, model, source, lang)
        out.append({"source": source, "output": text, "new_tokens": count, "seconds": round(seconds, 2)})
    return {"load_seconds": load_seconds, "translations": out, "memory": memory()}


def overfit(profile, args):
    tokenizer, model, load_seconds = load(profile)
    model = with_lora(profile, model)
    report = trainable_report(model)
    model.train()
    batch = tensors(profile, encoded(profile, tokenizer, rows(args.batch, seed=1)), tokenizer.pad_token_id)
    optimizer = optimiser(model, args.lr)
    losses = []
    for i in range(args.steps):
        losses.append(round(step(model, batch, optimizer), 4))
        print(f"step {i} loss {losses[-1]}", flush=True)
    assert min(losses[-3:]) < losses[0] * 0.3, f"loss did not collapse: {losses[0]} to {losses[-1]}"
    return {"load_seconds": load_seconds, "params": report, "losses": losses, "memory": memory()}


def train(profile, args):
    tokenizer, model, load_seconds = load(profile)
    model = with_lora(profile, model)
    report = trainable_report(model)
    model.train()
    data = encoded(profile, tokenizer, rows(args.steps * args.batch, seed=2, numbers=TRAIN_NUMBERS))
    optimizer = optimiser(model, args.lr)
    losses, peak_driver = [], 0.0
    tokens = labels = 0
    started = time.time()
    for i in range(args.steps):
        chunk = data[i * args.batch : (i + 1) * args.batch]
        batch = tensors(profile, chunk, tokenizer.pad_token_id)
        tokens += sum(len(r["input_ids"]) for r in chunk)
        labels += sum(sum(1 for t in r["labels"] if t >= 0) for r in chunk)
        driver = torch.mps.driver_allocated_memory()
        losses.append(round(step(model, batch, optimizer), 4))
        peak_driver = max(peak_driver, driver, torch.mps.driver_allocated_memory())
        if i % 10 == 0:
            elapsed = time.time() - started
            print(f"step {i} loss {losses[-1]} {tokens / elapsed:.0f} tok/s", flush=True)
    seconds = time.time() - started
    adapter = adapter_dir(profile)
    model.save_pretrained(adapter)
    return {
        "load_seconds": load_seconds,
        "params": report,
        "steps": args.steps,
        "batch": args.batch,
        "tokens": tokens,
        "label_tokens": labels,
        "avg_tokens_per_row": round(tokens / (args.steps * args.batch), 1),
        "seconds": round(seconds, 1),
        "tokens_per_second": round(tokens / seconds, 1),
        "rows_per_hour": round(args.steps * args.batch / seconds * 3600),
        "first_losses": losses[:5],
        "last_losses": losses[-5:],
        "peak_mps_driver_gib": round(peak_driver / GIB, 2),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "memory": memory(),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(profile, args):
    previous = json.loads((RUNS / f"{profile.name}-train.json").read_text())
    adapter = adapter_dir(profile)
    file_sha = sha256(adapter / "adapter_model.safetensors")
    assert file_sha == previous["adapter_sha256"], "adapter checksum changed"
    tokenizer, base, load_seconds = load(profile)
    base.eval()
    data = rows(args.count, seed=3, numbers=UNSEEN_NUMBERS)

    def marker_rate():
        hits, samples = 0, []
        for r in data:
            text, _, _ = generate(profile, tokenizer, base, r["source"], r["target_lang"], r["context"])
            hits += MARKER.strip() in text
            samples.append(text)
        return hits / len(data), samples[:3]

    before, before_samples = marker_rate()
    base = PeftModel.from_pretrained(base, adapter).to(DEVICE).eval()
    after, after_samples = marker_rate()
    assert before < 0.1, f"base model already writes the marker: {before}"
    assert after > 0.8, f"adapter did not learn the marker: {after}"
    return {
        "load_seconds": load_seconds,
        "unseen_rows": len(data),
        "marker_rate_base": before,
        "marker_rate_adapter": after,
        "samples_base": before_samples,
        "samples_adapter": after_samples,
        "adapter_sha256": file_sha,
        "memory": memory(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(PROFILES), default="hy7b")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("overfit")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p = sub.add_parser("train")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p = sub.add_parser("verify")
    p.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    profile = PROFILES[args.model]
    result = {"smoke": smoke, "overfit": overfit, "train": train, "verify": verify}[args.command](profile, args)
    result["versions"] = {"torch": torch.__version__, "model": profile.model, "model_revision": profile.revision}
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{profile.name}-{args.command}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
