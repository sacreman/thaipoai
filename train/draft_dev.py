"""Draft the dev suite with DeepSeek: messy, texting-style chat, harder than the pilot suite.

Every case is synthetic. Dev decisions use this suite and Jev (SPEC 5.1).
Reads the API key through draft_suite and never prints it.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train.draft_suite import THAI, call
from train.score import load

CATEGORIES = {
    "heavy_typos": "typed fast on a phone: misspellings, missing spaces, doubled letters",
    "shorthand": "texting shorthand and laughter, such as u r, thx, tmr, 555, ครับบบ, จ้าาา",
    "romanised": "Thai words written in Latin letters, as in mai pen rai or kin khao yang",
    "code_mixed": "English and Thai mixed inside one message",
    "regional": "regional or very informal wording, such as Isan words or street slang",
    "fragment": "a short fragment with most words dropped, understood only from the conversation",
    "learner": "written by someone still learning the language, with wrong grammar but a clear intent",
    "kin_pronouns": "uses nicknames or kinship pronouns such as phi and nong to mean you, me or them",
    "money_time_slang": "casual prices, times and quantities, such as 5 roi, bai song, a couple hundred",
    "emoji_tone": "the emoji, particles or punctuation carry the tone or change the meaning",
}
LATIN_SOURCE_ALLOWED = {"romanised", "code_mixed", "shorthand", "heavy_typos", "learner"}

PROMPT = """Write {count} test cases for evaluating an English-Thai chat translator on messy real-world texting.
Style: {description}.
Each message is written in {source_language}, as people really type on a phone. Keep it short and
imperfect, not polished. Everyday, family-friendly topics only: travel, friends, family, work, shops,
food, appointments. No sexual content. Invent all names.

Return JSON: {{"cases": [{{
 "context": ["A: ...", "B: ...", "A: ...", "B: ..."],
 "irrelevant_context": ["A: ...", "B: ...", "A: ...", "B: ..."],
 "source": "the message to translate, written in {source_language}",
 "meaning": "one English sentence saying what the message really means, resolved using the context",
 "literal_trap": "a short {target_language} phrase that appears only in a wrong, word-for-word translation, or an empty string"
}}]}}

context is four turns in {source_language} leading to the message, in the same messy style.
irrelevant_context is four turns in {source_language} about a different topic.
Make each case different."""


def valid(case, direction, category, excluded):
    if not all(isinstance(case.get(k), (str, list)) and case[k] for k in ("context", "irrelevant_context", "source", "meaning")):
        return False
    if len(case["context"]) != 4 or len(case["irrelevant_context"]) != 4 or case["source"] in excluded:
        return False
    return (direction == "th-en") == bool(THAI.search(case["source"])) or category in LATIN_SOURCE_ALLOWED


def draft(job):
    model, direction, category, count, excluded = job
    source, target = ("English", "Thai") if direction == "en-th" else ("Thai", "English")
    prompt = PROMPT.format(count=count + 3, description=CATEGORIES[category], source_language=source, target_language=target)
    kept = []
    for _ in range(4):
        try:
            cases = call(model, prompt)
        except Exception:
            continue  # a long reply can come back cut off, so ask again
        kept += [c for c in cases if isinstance(c, dict) and valid(c, direction, category, excluded)]
        if len(kept) >= count:
            break
    return direction, category, kept[:count], model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--out", default="suites/dev.jsonl")
    args = parser.parse_args()
    excluded = {c["source"] for path in Path("suites").glob("*.jsonl") if path.name != Path(args.out).name for c in load(path)}
    for path in Path("data/synthetic-chat-v1").glob("*.jsonl"):
        excluded |= {c["source"] for c in load(path)}
    jobs = [(args.model, d, c, args.per_category, excluded) for d in ("en-th", "th-en") for c in CATEGORIES]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(draft, jobs))
    rows = [{"id": f"{d}-{c}-{i}", "direction": d, "category": c, "author": m, **case}
            for d, c, cases, m in results for i, case in enumerate(cases)]
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    print(json.dumps({"cases": len(rows), "short_cells": [f"{d} {c} {len(cs)}" for d, c, cs, _ in results if len(cs) < args.per_category]}))


if __name__ == "__main__":
    main()
