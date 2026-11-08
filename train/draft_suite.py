"""Draft the step 2 pilot suite with DeepSeek. Every case is synthetic and AI-generated.

Reads the API key from ~/.config/thaipoai/deepseek_key and never prints it.
"""

import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KEY = Path.home() / ".config/thaipoai/deepseek_key"
URL = "https://api.deepseek.com/chat/completions"
THAI = re.compile(r"[฀-๿]")

CATEGORIES = {
    "dropped_subject": "the message omits who is doing something, and the conversation makes it clear",
    "speaker_gender": "the correct wording depends on the speaker's gender, shown by the conversation",
    "reference": "the message says 'it', 'that one' or similar, resolved by the conversation",
    "idiom": "the message uses an idiom or saying that must not be translated word for word",
    "slang": "the message uses everyday slang or internet language",
    "sarcasm": "the message is sarcastic or teasing, so its literal meaning is wrong",
    "numbers_time": "the message has times, dates, prices or counts, including Thai clock expressions",
    "register": "the tone is notably casual, warm, annoyed or formal, and must be kept",
    "negation": "the message has a negation or a double meaning that is easy to flip",
    "typo_mixed": "the message has phone typos, romanised words or mixed English and Thai",
}

PROMPT = """Write {count} test cases for evaluating an English-Thai chat translator.
Category: {category}: {description}.
Each message is written in {source_language} by a fluent native speaker and will be
translated into {target_language}. Everyday, family-friendly topics only: travel, friends,
family, work, shops, food, appointments. No sexual content. Invent all names.

Return JSON: {{"cases": [{{
 "context": ["A: ...", "B: ...", "A: ...", "B: ..."],
 "irrelevant_context": ["A: ...", "B: ...", "A: ...", "B: ..."],
 "source": "the message to translate, written in {source_language}",
 "meaning": "one English sentence saying what the message really means, resolved using the context",
 "literal_trap": "a short {target_language} phrase that appears only in a wrong, word-for-word translation, or an empty string"
}}]}}

context is four natural turns in {source_language} leading to the message.
irrelevant_context is four natural turns in {source_language} about a different topic.
Make each case different. Natural chat, not textbook sentences."""


def call(model, prompt):
    key = KEY.read_text().strip()
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.8,
            "max_tokens": 16000,
        }
    ).encode()
    request = urllib.request.Request(
        URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.loads(json.loads(response.read())["choices"][0]["message"]["content"])["cases"]


def valid(case, direction):
    needs = ("context", "irrelevant_context", "source", "meaning")
    if not all(case.get(k) for k in needs):
        return False
    if len(case["context"]) != 4 or len(case["irrelevant_context"]) != 4:
        return False
    return bool(THAI.search(case["source"])) == (direction == "th-en") or case.get("_mixed", False)


def draft(job):
    model, direction, category, count, excluded = job
    source, target = ("English", "Thai") if direction == "en-th" else ("Thai", "English")
    prompt = PROMPT.format(
        count=count + 2,
        category=category,
        description=CATEGORIES[category],
        source_language=source,
        target_language=target,
    )
    kept = []
    for _ in range(4):
        try:
            cases = call(model, prompt)
        except (json.JSONDecodeError, KeyError):
            continue  # a long reply can come back cut off, so ask again
        for case in cases:
            if category == "typo_mixed":
                case["_mixed"] = True
            if valid(case, direction) and case["source"] not in excluded:
                kept.append(case)
        if len(kept) >= count:
            break
    return direction, category, kept[:count], model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--out", default="suites/pilot.jsonl")
    parser.add_argument("--exclude", help="existing suite whose messages must not be repeated")
    args = parser.parse_args()
    excluded = set()
    if args.exclude:
        excluded = {json.loads(line)["source"] for line in Path(args.exclude).read_text(encoding="utf8").splitlines()}
    jobs = [(args.model, d, c, args.per_category, excluded) for d in ("en-th", "th-en") for c in CATEGORIES]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(draft, jobs))
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    total = 0
    with out.open("w", encoding="utf8") as handle:
        for direction, category, cases, model in results:
            print(f"{direction} {category}: {len(cases)}")
            for i, case in enumerate(cases):
                case.pop("_mixed", None)
                row = {"id": f"{direction}-{category}-{i}", "direction": direction, "category": category,
                       "author": model, **case}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    print(f"{total} cases written to {out}")


if __name__ == "__main__":
    main()
