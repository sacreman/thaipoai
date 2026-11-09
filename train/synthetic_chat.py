"""Draft chat-style training pairs with DeepSeek and keep those Jev says carry the meaning.

Every row is synthetic. Messages already in a test suite are refused, so the suites stay unseen.
Prints counts, never text. The DeepSeek key is read by draft_suite and never printed.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train.draft_suite import CATEGORIES, THAI, call
from train.score import load, score_jev

TOPICS = [
    "a late bus", "cooking dinner", "a sick parent", "a broken phone", "a birthday party", "rent",
    "a job interview", "a football match", "a beach trip", "school fees", "a new haircut",
    "a borrowed motorbike", "a visa appointment", "the market", "a noisy neighbour", "a pet",
    "a wedding", "a delayed parcel", "exam results", "a rainy weekend", "a gym plan", "a hotel booking",
]

PROMPT = """Write {count} examples for training an English-Thai chat translator.
Category: {category}: {description}.
Each message is written in {source_language} by a fluent native speaker. Everyday,
family-friendly topics only, loosely around: {topic}. No sexual content. Invent all names.

Return JSON: {{"cases": [{{
 "context": ["A: ...", "B: ...", "A: ...", "B: ..."],
 "source": "the message to translate, written in {source_language}",
 "meaning": "one English sentence saying what the message really means, resolved using the context",
 "translation": "how a fluent bilingual friend would say it in {target_language}"
}}]}}

context is four natural turns in {source_language} leading to the message.
translation must carry the real meaning, tone and speaker gender. Never translate word for word,
never add explanations, and write only the translated message.
Make each example different. Natural chat, not textbook sentences."""


def usable(case, direction, seen):
    """A well-formed pair in the right languages whose message is not in a suite or a repeat."""
    if not all(isinstance(case.get(k), str) and case[k].strip() for k in ("source", "meaning", "translation")):
        return False
    context = case.get("context")
    if not isinstance(context, list) or len(context) != 4 or not all(isinstance(c, str) for c in context):
        return False
    if case["source"] in seen:
        return False
    thai_target = bool(THAI.search(case["translation"]))
    return thai_target if direction == "en-th" else not thai_target


def draft(job):
    model, direction, category, count, seed = job
    source, target = ("English", "Thai") if direction == "en-th" else ("Thai", "English")
    prompt = PROMPT.format(count=count, category=category, description=CATEGORIES[category],
                           source_language=source, target_language=target,
                           topic=", ".join(random.Random(seed).sample(TOPICS, 3)))
    try:
        cases = call(model, prompt)
    except Exception:  # a cut-off reply costs one batch, and the counts show it
        return []
    return [{**c, "direction": direction, "category": category} for c in cases if isinstance(c, dict)]


def carries_meaning(case):
    try:
        return score_jev(case, case["translation"]) >= 0.5
    except Exception:
        return False


def row_of(case):
    english_target = case["direction"] == "th-en"
    return {"source": case["source"], "target": case["translation"], "context": case["context"],
            "target_lang": "English" if english_target else "Thai",
            "direction": case["direction"], "category": case["category"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=20, help="batches per direction and category")
    parser.add_argument("--per-batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default="runs/synth-chat")
    args = parser.parse_args()
    seen = {c["source"] for path in Path("suites").glob("*.jsonl") for c in load(path)}
    jobs = [(args.model, d, c, args.per_batch, f"{d}{c}{i}")
            for d in ("en-th", "th-en") for c in CATEGORIES for i in range(args.batches)]
    with ThreadPoolExecutor(args.workers) as pool:
        drafted = [case for batch in pool.map(draft, jobs) for case in batch]
    kept = []
    for case in drafted:
        if usable(case, case["direction"], seen):
            seen.add(case["source"])
            kept.append(case)
    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(carries_meaning, kept))
    rows = [row_of(c) for c, ok in zip(kept, verdicts) if ok]
    random.Random(7).shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = max(100, len(rows) // 20)
    for name, part in (("validation", rows[:split]), ("train", rows[split:])):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf8")
    print(json.dumps({"batches": len(jobs), "drafted": len(drafted), "usable": len(kept),
                      "jev_kept": len(rows), "validation": split, "train": len(rows) - split}))


if __name__ == "__main__":
    main()
