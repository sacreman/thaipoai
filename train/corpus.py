"""Combine the vetted English-Thai sources into one corpus, keeping each pair's provenance.

Reads the downloads under data/public and writes data/public/corpus. Every row records its source,
subset, review process and licence, so a training run can select exactly what it uses.

Build:     python -m train.corpus
MASSIVE:   python -m train.corpus --verify-massive   (three Jev votes per pair, writes the passing ids)
Audit:     python -m train.corpus --audit 150        (Jev votes on a sample, prints rates, never text)
"""

import argparse
import collections
import hashlib
import json
import random
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.compare_systems import majority
from evaluation.judge import load, score_jev
from train.sources import LICENCES

ROOT = Path("data/public")
OUT = ROOT / "corpus"
MASSIVE_VERIFIED = Path("data/massive-th-verified.json")
SPLITS = ("test", "validation", "train")  # held-out splits first, so training rows lose every tie
MAX_CHARS = 400

THAI = re.compile("[฀-๿]")
LATIN = re.compile("[A-Za-z]")
ZERO_WIDTH = re.compile("[\u200b-\u200d\u2060\ufeff]")
THAI_GAP = re.compile("(?<=[฀-๿])\\s+(?=[฀-๿])")
ADULT = re.compile(r"(?i)\b(sex|sexy|porn\w*|fuck\w*|nude|naked|rape\w*|penis|vagina|orgasm|prostitut\w*|whore)\b")
MASSIVE_SLOT = re.compile(r"\[(\w+) : ")

# SCB's training split still uses older subset names, so both spellings map to one subset.
SCB_PROFESSIONAL = {
    "task_master_1": "task_master",
    "task_master_corpus": "task_master",
    "generated_reviews_translator": "generated_reviews_translator",
    "fake_reviews_hope": "generated_reviews_translator",
}


def clean(text):
    """NFC, no zero-width characters, single spaces, and one spelling of sara am."""
    text = ZERO_WIDTH.sub("", unicodedata.normalize("NFC", text))
    text = text.replace("\u0e4d\u0e32", "\u0e33")
    text = re.sub("\u0e4d([\u0e48-\u0e4b])\u0e32", "\\1\u0e33", text)
    return " ".join(text.split())


def key(text):
    return clean(text).casefold()


def row(row_id, en, th, source, subset, review, split):
    return {"id": row_id, "en": clean(en), "th": clean(th), "source": source, "subset": subset,
            "review": review, "licence": LICENCES[source], "split": split}


def usable(r):
    return bool(LATIN.search(r["en"]) and THAI.search(r["th"]) and not THAI.search(r["en"])
                and len(r["en"]) <= MAX_CHARS and len(r["th"]) <= MAX_CHARS and not ADULT.search(r["en"]))


def scb():
    import pyarrow.parquet as pq

    for split in ("train", "validation", "test"):
        table = pq.read_table(ROOT / f"scb/{split}.parquet")
        for i, (pair, name) in enumerate(zip(table["translation"].to_pylist(), table["subdataset"].to_pylist())):
            if name in SCB_PROFESSIONAL:
                yield row(f"scb-{split}-{i}", pair["en"], pair["th"], "scb", SCB_PROFESSIONAL[name], "professional", split)


def mtop_file(language, split):
    rows = {}
    for line in (ROOT / f"mtop/mtop/{language}/{split}.txt").read_text(encoding="utf8").splitlines():
        parts = line.split("\t")
        rows[parts[0]] = (parts[3], sorted(s.rsplit(":", 1)[1] for s in parts[2].split(",") if s))
    return rows


def mtop():
    for split, name in (("train", "train"), ("eval", "validation"), ("test", "test")):
        english, thai = mtop_file("en", split), mtop_file("th", split)
        for i, (th, th_slots) in thai.items():
            if i in english and english[i][1] == th_slots:
                yield row(f"mtop-{i}", english[i][0], th, "mtop", "assistant", "translated_and_reviewed", name)


def massive_faithful(en, th):
    """Strong unanimous ratings, nothing localized, and the same slots as the English."""
    if any(m["method"] == "localization" for m in th["slot_method"]):
        return False
    if sorted(MASSIVE_SLOT.findall(th["annot_utt"])) != sorted(MASSIVE_SLOT.findall(en["annot_utt"])):
        return False
    return all(j["intent_score"] == 1 and j["slots_score"] in (1, 2) and j["grammar_score"] == 4
               and j["spelling_score"] == 2 and j["language_identification"] == "target" for j in th["judgments"])


def massive_candidates():
    english = {r["id"]: r for r in load(ROOT / "massive/1.1/data/en-US.jsonl")}
    for th in load(ROOT / "massive/1.1/data/th-TH.jsonl"):
        en = english.get(th["id"])
        if en and en["partition"] == th["partition"] and massive_faithful(en, th):
            split = "validation" if th["partition"] == "dev" else th["partition"]
            # Annotators put spaces between Thai words, which Thai does not do.
            yield row(f"massive-{th['id']}", en["utt"], THAI_GAP.sub("", th["utt"]), "massive", "assistant", "rated_and_verified", split)


def massive():
    verified = set(json.loads(MASSIVE_VERIFIED.read_text(encoding="utf8"))["passed"])
    return (r for r in massive_candidates() if r["id"] in verified)


def assemble(rows, blocked=frozenset()):
    """Drop repeats and anything that could leak held-out text into training. Returns rows and drop counts."""
    by_split = {s: [] for s in SPLITS}
    for r in rows:
        by_split[r["split"]].append(r)
    pairs, english, thai, kept = set(), set(), set(), []
    dropped = collections.Counter()
    for split in SPLITS:
        for r in by_split[split]:
            en, th = key(r["en"]), key(r["th"])
            reason = ("unusable" if not usable(r) else "duplicate" if (en, th) in pairs
                      else "matches a suite" if en in blocked or th in blocked else None)
            if reason is None and split == "train" and (en in english or th in thai):
                reason = "matches held-out text"
            if reason:
                dropped[r["source"], reason] += 1
                continue
            pairs.add((en, th))
            if split != "train":
                english.add(en)
                thai.add(th)
            kept.append(r)
    return kept, dropped


def suite_sources():
    return frozenset(key(case["source"]) for path in Path("suites").glob("*.jsonl") for case in load(path))


def build():
    rows, dropped = assemble([*mtop(), *massive(), *scb()], suite_sources())
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"rows": collections.Counter(), "files": {}}
    for split in SPLITS:
        part = sorted((r for r in rows if r["split"] == split), key=lambda r: r["id"])
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part).encode()
        (OUT / f"{split}.jsonl").write_bytes(data)
        manifest["files"][split] = {"rows": len(part), "sha256": hashlib.sha256(data).hexdigest()}
        manifest["rows"].update(f"{split} {r['source']} {r['subset']}" for r in part)
    manifest["rows"] = dict(sorted(manifest["rows"].items()))
    manifest["dropped"] = {f"{source}: {reason}": n for (source, reason), n in sorted(dropped.items())}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf8")
    print(json.dumps(manifest, indent=1))


def votes(rows, count=3):
    def verdict(r):
        case = {"source": r["en"], "meaning": r["en"]}

        def once():
            try:
                return score_jev(case, r["th"]) >= 0.5
            except Exception:
                return None

        return majority([once() for _ in range(count)])

    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(verdict, rows))


def verify_massive():
    rows = list(massive_candidates())
    answers = votes(rows)
    passed = sorted(r["id"] for r, ok in zip(rows, answers) if ok)
    MASSIVE_VERIFIED.write_text(json.dumps({"judge": "jev-latest, three votes, majority", "candidates": len(rows), "passed": passed}) + "\n", encoding="utf8")
    print(json.dumps({"candidates": len(rows), "passed": len(passed), "failed": sum(a is False for a in answers), "unjudged": sum(a is None for a in answers)}))


def audit(per_subset):
    groups = collections.defaultdict(list)
    for r in load(OUT / "train.jsonl"):
        groups[r["source"], r["subset"]].append(r)
    report = {}
    for (source, subset), rows in sorted(groups.items()):
        sample = random.Random(11).sample(rows, min(per_subset, len(rows)))
        answers = votes(sample)
        report[f"{source}/{subset}"] = {"sampled": len(sample), "carried_meaning": sum(a is True for a in answers),
                                        "wrong": sum(a is False for a in answers), "unjudged": sum(a is None for a in answers)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-massive", action="store_true")
    parser.add_argument("--audit", type=int)
    args = parser.parse_args()
    verify_massive() if args.verify_massive else audit(args.audit) if args.audit else build()
