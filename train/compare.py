"""Compare scored systems, pooled over both test sets. Prints misleading rates per direction."""

import json
import sys
from pathlib import Path

from train.score import load, rate

SYSTEMS = {
    "Lumen 8-bit (system format)": ["runs/fmt-system-scores.jsonl", "runs/confirm-scores.jsonl"],
    "DeepSeek Flash": ["runs/ds-pilot-scores.jsonl", "runs/ds-confirm-scores.jsonl"],
}


SCREEN = {
    "Lumen 8-bit (system format)": ["runs/confirm-scores.jsonl"],
    "DeepSeek Flash": ["runs/ds-confirm-scores.jsonl"],
    "Hy-MT2 7B": ["runs/screen-hy7b-scores.jsonl"],
    "Qingqiu-MT-9B": ["runs/screen-qingqiu9b-scores.jsonl"],
    "Gemma 4 E4B": ["runs/screen-gemma4e4b-scores.jsonl"],
    "Index-Translate-9B": ["runs/screen-index9b-scores.jsonl"],
    "Gemma 4 12B": ["runs/screen-gemma412b-scores.jsonl"],
    "ChindaMT-4B": ["runs/screen-chinda4b-scores.jsonl"],
    "Mango-MT-9B": ["runs/screen-mango9b-scores.jsonl"],
}

FIRST_SET = {
    "Lumen 8-bit (system format)": ["runs/fmt-system-scores.jsonl"],
    "DeepSeek Flash": ["runs/ds-pilot-scores.jsonl"],
    "Gemma 4 E4B": ["runs/screen2-gemma4e4b-scores.jsonl"],
}


def table(systems):
    out = {}
    for name, paths in systems.items():
        if not all(Path(p).exists() for p in paths):
            out[name] = {"scored": 0, "note": "not run"}
            continue
        rows = [r for p in paths for r in load(p) if r.get("score") is not None]
        entry = {"scored": len(rows)}
        for variant in ("none", "correct", "irrelevant"):
            for direction in ("en-th", "th-en"):
                chosen = [r for r in rows if r["variant"] == variant and r["direction"] == direction]
                entry[f"{variant} {direction}"] = {
                    "n": len(chosen),
                    "misleading": rate(chosen),
                    "mean": round(sum(r["score"] for r in chosen) / len(chosen), 2) if chosen else None,
                    "context_translated": sum(1 for r in chosen if r.get("translated_context")),
                }
        out[name] = entry
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    result = table({"screen": SCREEN, "first": FIRST_SET, "all": SYSTEMS}[which])
    Path(f"runs/compare-{which}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
