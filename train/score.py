"""Score translations for meaning with Opus 5.5, cross-checked by Jev.

Opus only ever scores finished models. Its scores never feed training data.
API keys are read from ~/.config and never printed. Translations are scored blind:
the judge is never told which system wrote them.
"""

import argparse
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

CONFIG = Path.home() / ".config"
OPUS = "claude-opus-5-5"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
DEFECTS = [
    "none", "actor", "polarity", "question", "number", "gender", "invented",
    "dropped", "typo", "language", "commentary", "leak", "other",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "defect": {"type": "string", "enum": DEFECTS},
        "translated_context": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["score", "defect", "translated_context", "reason"],
    "additionalProperties": False,
}

RUBRIC = """You judge chat translation between English and Thai, purely on meaning.

Score how well the translation carries the intended meaning to its reader:
5 correct and natural. 4 correct, slightly awkward. 3 minor slip, the reader still gets it.
2 misleading: the reader gets a wrong actor, wrong polarity, wrong number or time, a wrong
gender for the speaker, or an idiom taken literally. 1 wrong, empty, or not a translation.

Do not penalise style choices that keep the meaning. If the source is truly ambiguous even
with the conversation, a translation keeping that ambiguity is correct.

Set translated_context to true only if the output translates the earlier conversation
instead of, or as well as, the message. Set defect to the main problem, or "none" when the
score is 3 or higher. Keep reason under 20 words, in English."""


def judge_prompt(case, output):
    target = "Thai" if case["direction"] == "en-th" else "English"
    conversation = "\n".join(case.get("context", []))
    return (
        f"Conversation so far:\n{conversation}\n\nMessage to translate:\n{case['source']}\n\n"
        f"What the message really means:\n{case['meaning']}\n\n"
        f"Translation into {target}:\n{output}"
    )


def score_opus(client, case, output):
    response = client.beta.messages.create(
        model=OPUS,
        max_tokens=4000,
        system=RUBRIC,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": judge_prompt(case, output)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("judge refused")
    text = next(block.text for block in response.content if block.type == "text")
    return {**json.loads(text), "judge": response.model}


def score_jev(case, output):
    key = (CONFIG / "typesafe/key").read_text().strip()
    body = json.dumps(
        {
            "model": "jev-latest",
            "state": {"message": case["source"], "intended_meaning": case["meaning"], "translation": output},
            "questions": {
                "same": {
                    "type": "noul",
                    "instructions": "Does `translation` convey `intended_meaning` of `message`?",
                    "criteria": {
                        "true": "A reader of the translation would understand the intended meaning.",
                        "false": "A reader would get a different, wrong or incomplete meaning.",
                    },
                }
            },
        }
    ).encode()
    request = urllib.request.Request(
        JEV_URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return float(json.loads(response.read())["answers"]["same"]["noul"])


def score_row(client, row, cases):
    out = {k: row[k] for k in ("id", "system", "variant", "direction", "category")}
    if row.get("error") or not row.get("output"):
        return {**out, "score": 1, "defect": "other", "translated_context": False, "error": row.get("error") or "empty"}
    case = cases[row["id"]]
    try:
        out.update(score_opus(client, case, row["output"]))
    except (anthropic.APIError, RuntimeError, json.JSONDecodeError, StopIteration) as err:
        return {**out, "score": None, "error": f"opus: {type(err).__name__}"}
    try:
        out["jev_same"] = score_jev(case, row["output"])
    except Exception as err:  # Jev is a cross-check, so its failure must not lose the Opus score
        out["jev_error"] = type(err).__name__
    return out


def rate(rows):
    return round(sum(1 for r in rows if r["score"] <= 2) / len(rows), 4) if rows else None


def summarise(rows):
    """Apply docs/steps/02-go-no-go.md. Unscored rows are reported, never silently dropped."""
    scored = [r for r in rows if r.get("score") is not None]
    groups = defaultdict(list)
    for r in scored:
        groups[r["variant"], r["direction"]].append(r)

    def misleading(variant, direction=None):
        keys = [direction] if direction else ["en-th", "th-en"]
        return rate([r for d in keys for r in groups[variant, d]])

    correct = [r for d in ("en-th", "th-en") for r in groups["correct", d]]
    leak = round(sum(1 for r in correct if r["translated_context"]) / len(correct), 4) if correct else None
    none, with_context, irrelevant = misleading("none"), misleading("correct"), misleading("irrelevant")
    measures = {
        "en_th_correct": misleading("correct", "en-th"),
        "th_en_correct": misleading("correct", "th-en"),
        "translated_context": leak,
        "none": none,
        "correct": with_context,
        "irrelevant": irrelevant,
    }
    rules = {
        "1_en_th": measures["en_th_correct"] is not None and measures["en_th_correct"] <= 0.15,
        "2_th_en": measures["th_en_correct"] is not None and measures["th_en_correct"] <= 0.15,
        "3_context_not_translated": leak is not None and leak <= 0.02,
        "4_context_not_worse": None not in (none, with_context) and with_context <= none,
        "5_irrelevant_tolerated": None not in (none, irrelevant) and irrelevant <= none + 0.05,
    }
    jev = [r for r in scored if r.get("jev_same") is not None]
    agree = sum(1 for r in jev if (r["jev_same"] >= 0.5) == (r["score"] >= 3))
    return {
        "scored": len(scored),
        "unscored": len(rows) - len(scored),
        "measures": measures,
        "rules": rules,
        "go": rules["2_th_en"] and rules["3_context_not_translated"],
        "all_rules_pass": all(rules.values()),
        "jev_agreement": round(agree / len(jev), 4) if jev else None,
        "jev_checked": len(jev),
    }


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reuse-none", help="scored file whose no-context rows are reused, unchanged")
    args = parser.parse_args()
    cases = {c["id"]: c for c in load(args.suite)}
    rows = [row for path in args.results for row in load(path)]
    client = anthropic.Anthropic(api_key=(CONFIG / "thaipoai/anthropic_key").read_text().strip())
    with ThreadPoolExecutor(8) as pool:
        scored = list(pool.map(lambda row: score_row(client, row, cases), rows))
    if args.reuse_none:  # with no history every format sends the identical prompt
        scored += [r for r in load(args.reuse_none) if r["variant"] == "none"]
    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored), encoding="utf8"
    )
    print(json.dumps(summarise(scored), indent=2))


if __name__ == "__main__":
    main()
