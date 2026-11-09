"""Paired comparison of two translation systems on one suite, judged by Jev. Prints counts, never text."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import comb, sqrt

from train.score import load, score_jev


def one_sided(hits, misses):
    n = hits + misses
    return sum(comb(n, i) for i in range(hits, n + 1)) / 2**n if n else 1.0


def upper_95(worse, better, n):
    """One-sided 95% upper bound, in points, on the paired gap in wrong rate. Normal approximation."""
    if not n:
        return None
    return 100 * ((worse - better) / n + 1.645 * sqrt(worse + better - (worse - better) ** 2 / n) / n)


def compare(baseline, candidate, keys):
    """baseline and candidate map a key to True (meaning carried), False, or None (not judged)."""
    keys = [k for k in keys if baseline.get(k) is not None and candidate.get(k) is not None]
    fixed = sum(1 for k in keys if not baseline[k] and candidate[k])
    broke = sum(1 for k in keys if baseline[k] and not candidate[k])
    upper = upper_95(broke, fixed, len(keys))
    return {
        "pairs": len(keys),
        "baseline_wrong": sum(1 for k in keys if not baseline[k]),
        "candidate_wrong": sum(1 for k in keys if not candidate[k]),
        "fixed_by_candidate": fixed,
        "broken_by_candidate": broke,
        "p_candidate_better": round(one_sided(fixed, broke), 4),
        "p_candidate_worse": round(one_sided(broke, fixed), 4),
        "gap_points": round(100 * (broke - fixed) / len(keys), 2) if keys else None,
        "upper_95_points": round(upper, 2) if upper is not None else None,
    }


def majority(votes):
    """True or False by most votes. A failed call is no vote, and a tie or no votes is None."""
    votes = [v for v in votes if v is not None]
    yes = sum(votes)
    return None if yes * 2 == len(votes) else yes * 2 > len(votes)


def judged(paths, cases, votes):
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
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--votes", type=int, default=3, help="Jev judgements per translation, odd")
    args = parser.parse_args()
    cases = {c["id"]: c for c in load(args.suite)}
    baseline, candidate = judged(args.baseline, cases, args.votes), judged(args.candidate, cases, args.votes)
    keys = sorted(baseline)
    report = {"all": compare(baseline, candidate, keys)}
    for direction in ("en-th", "th-en"):
        report[direction] = compare(baseline, candidate, [k for k in keys if cases[k[0]]["direction"] == direction])
    for variant in sorted({k[1] for k in keys}):
        report[variant] = compare(baseline, candidate, [k for k in keys if k[1] == variant])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
