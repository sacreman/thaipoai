"""Draft a test suite with DeepSeek. Every case is synthetic.

`clean` tests meaning: idiom, sarcasm, register. `messy` tests texting: typos, shorthand, romanised
Thai. The API key is read from ~/.config/thaipoai/deepseek_key and never printed.
"""

import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.judge import load

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

MESSY_CATEGORIES = {
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

FORMAT = """Everyday, family-friendly topics only: travel, friends, family, work, shops, food,
appointments. No sexual content. Invent all names.

Return JSON: {{"cases": [{{
 "context": ["A: ...", "B: ...", "A: ...", "B: ..."],
 "irrelevant_context": ["A: ...", "B: ...", "A: ...", "B: ..."],
 "source": "the message to translate, written in {source_language}",
 "meaning": "one English sentence saying what the message really means, resolved using the context",
 "literal_trap": "a short {target_language} phrase that appears only in a wrong, word-for-word translation, or an empty string"
}}]}}

context is four turns in {source_language} leading to the message, in the same style.
irrelevant_context is four turns in {source_language} about a different topic.
Make each case different."""

STYLES = {
    "clean": {
        "categories": CATEGORIES,
        "latin_source": {"typo_mixed"},
        "prompt": "Write {count} test cases for evaluating an English-Thai chat translator.\n"
                  "Style: {description}.\nEach message is natural chat written in {source_language} by a fluent native speaker.\n",
    },
    "messy": {
        "categories": MESSY_CATEGORIES,
        "latin_source": {"romanised", "code_mixed", "shorthand", "heavy_typos", "learner"},
        "prompt": "Write {count} test cases for evaluating an English-Thai chat translator on messy real-world texting.\n"
                  "Style: {description}.\nEach message is written in {source_language}, as people really type on a phone: short and imperfect.\n",
    },
}


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


def valid(case, direction, category, excluded, latin_source):
    """A well-formed case, new to every suite, whose message is in the right script."""
    if not all(isinstance(case.get(k), (str, list)) and case[k] for k in ("context", "irrelevant_context", "source", "meaning")):
        return False
    if len(case["context"]) != 4 or len(case["irrelevant_context"]) != 4 or case["source"] in excluded:
        return False
    return (direction == "th-en") == bool(THAI.search(case["source"])) or category in latin_source


def draft(job):
    model, style, direction, category, count, excluded = job
    source, target = ("English", "Thai") if direction == "en-th" else ("Thai", "English")
    prompt = (style["prompt"] + FORMAT).format(count=count + 3, description=style["categories"][category],
                                               source_language=source, target_language=target)
    kept = []
    for _ in range(4):
        try:
            cases = call(model, prompt)
        except Exception:
            continue  # a long reply can come back cut off, so ask again
        kept += [c for c in cases if isinstance(c, dict) and valid(c, direction, category, excluded, style["latin_source"])]
        if len(kept) >= count:
            break
    return direction, category, kept[:count], model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--style", choices=STYLES, required=True)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    existing = [p for p in Path("suites").glob("*.jsonl") if p.name != Path(args.out).name]
    existing += Path("data/synthetic-chat-v1").glob("*.jsonl")
    excluded = {row["source"] for path in existing for row in load(path)}
    style = STYLES[args.style]
    jobs = [(args.model, style, d, c, args.per_category, excluded) for d in ("en-th", "th-en") for c in style["categories"]]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(draft, jobs))
    rows = [{"id": f"{d}-{c}-{i}", "direction": d, "category": c, "author": m, **case}
            for d, c, cases, m in results for i, case in enumerate(cases)]
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    print(json.dumps({"cases": len(rows), "short": [f"{d} {c} {len(cs)}" for d, c, cs, _ in results if len(cs) < args.per_category]}))


if __name__ == "__main__":
    main()
