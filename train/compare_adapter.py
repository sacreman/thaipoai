"""Paired check of a trained adapter against its base model, judged by Jev only.

Dev decisions never use Opus, which only scores frozen candidates (SPEC 5.1).
Prints counts, never text.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import comb

from train.score import load, score_jev


def one_sided(hits, misses):
    n = hits + misses
    return sum(comb(n, i) for i in range(hits, n + 1)) / 2**n if n else 1.0


def compare(base, adapter, keys):
    """base and adapter map a key to True (meaning carried), False, or None (not judged)."""
    keys = [k for k in keys if base.get(k) is not None and adapter.get(k) is not None]
    fixed = sum(1 for k in keys if not base[k] and adapter[k])
    broke = sum(1 for k in keys if base[k] and not adapter[k])
    return {
        "pairs": len(keys),
        "base_wrong": sum(1 for k in keys if not base[k]),
        "adapter_wrong": sum(1 for k in keys if not adapter[k]),
        "fixed_by_adapter": fixed,
        "broken_by_adapter": broke,
        "p_adapter_better": round(one_sided(fixed, broke), 4),
        "p_adapter_worse": round(one_sided(broke, fixed), 4),
    }


def majority(votes):
    """True or False by most votes. A failed call is no vote, and a tie or no votes is None."""
    votes = [v for v in votes if v is not None]
    yes = sum(votes)
    return None if yes * 2 == len(votes) else yes * 2 > len(votes)


def judged(paths, cases, votes=1):
    def once(row):
        try:
            return score_jev(cases[row["id"]], row["output"]) >= 0.5
        except Exception:
            return None  # an unjudged case is left out of the pairs and shows in the count

    def verdict(row):
        if row.get("error") or not row.get("output"):
            return False
        return majority([once(row) for _ in range(votes)])

    rows = [row for path in paths for row in load(path)]
    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(verdict, rows))
    return {(r["id"], r["variant"]): a for r, a in zip(rows, answers)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--adapter", nargs="+", required=True)
    parser.add_argument("--votes", type=int, default=1, help="Jev judgements per translation, odd")
    args = parser.parse_args()
    cases = {c["id"]: c for c in load(args.suite)}
    base, adapter = judged(args.base, cases, args.votes), judged(args.adapter, cases, args.votes)
    keys = sorted(base)
    report = {"all": compare(base, adapter, keys)}
    for direction in ("en-th", "th-en"):
        wanted = [k for k in keys if cases[k[0]]["direction"] == direction]
        report[direction] = compare(base, adapter, wanted)
    for variant in sorted({k[1] for k in keys}):
        report[variant] = compare(base, adapter, [k for k in keys if k[1] == variant])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
