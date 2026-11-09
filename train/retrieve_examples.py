"""Copy a suite, adding to each case the most similar synthetic pairs as prompt notes.

Similarity is TF-IDF over character trigrams, which needs no Thai word splitter and tolerates
typos. Only the message is matched, never a case's category or meaning. Prints counts, never text.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from train.score import load

HEADER = "Examples of how similar chat messages were translated. Use them for style and slang only. Never copy their content:"
K = 3


def grams(text):
    text = " ".join(text.casefold().split())
    return Counter(text[i : i + 3] for i in range(max(len(text) - 2, 1)))


class Index:
    def __init__(self, pairs):
        self.pairs = pairs
        counts = [grams(p["source"]) for p in pairs]
        seen = Counter(g for c in counts for g in c)
        self.idf = {g: math.log(1 + len(pairs) / n) for g, n in seen.items()}
        self.vectors = [self.vector(c) for c in counts]

    def vector(self, counts):
        weights = {g: n * self.idf[g] for g, n in counts.items() if g in self.idf}
        size = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        return {g: w / size for g, w in weights.items()}

    def nearest(self, text, k):
        query = self.vector(grams(text))
        score = lambda v: sum(w * v.get(g, 0.0) for g, w in query.items())
        ranked = sorted(range(len(self.pairs)), key=lambda i: score(self.vectors[i]), reverse=True)
        return [self.pairs[i] for i in ranked[:k]]


def notes_for(pairs):
    return "\n".join([HEADER] + [f"Message: {p['source']}\nTranslation: {p['target']}" for p in pairs])


def with_notes(cases, pairs, k):
    """Each case only sees pairs going the same way, and never its own message."""
    indexes = {d: Index([p for p in pairs if p["direction"] == d]) for d in ("en-th", "th-en")}
    out = []
    for case in cases:
        hits = [p for p in indexes[case["direction"]].nearest(case["source"], k + 1) if p["source"] != case["source"]]
        out.append({**case, "notes": notes_for(hits[:k])})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="suites/dev.jsonl")
    parser.add_argument("--pairs", default="runs/synth-chat/train.jsonl")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = with_notes(load(args.suite), load(args.pairs), args.k)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    print(json.dumps({"cases": len(rows), "k": args.k}))


if __name__ == "__main__":
    main()
