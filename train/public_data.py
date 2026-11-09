"""Build the public English-Thai training set allowed by section 4.7 of the spec.

ALT (CC BY 4.0), TALPCo (CC BY 4.0) and Tatoeba (per sentence, default CC BY 2.0 FR). The Common
Voice pairs are left out until a CC0 file is confirmed. Writes to data/public, which is gitignored,
and records every Tatoeba author so they can be credited in the model card.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path("data/public")
THAI = re.compile("[฀-๿]")
LETTER = re.compile("[A-Za-z]")
ADULT = re.compile(r"(?i)\b(sex|sexy|porn\w*|fuck\w*|nude|naked|rape\w*|penis|vagina|orgasm|prostitut\w*|whore)\b")
VALIDATION_PERCENT = 2
MAX_CHARS = 400


def tsv(path):
    for line in path.read_text(encoding="utf8").splitlines():
        yield line.split("\t")


def paired(en, th):
    english = {parts[0]: parts[1] for parts in tsv(en) if len(parts) >= 2}
    for parts in tsv(th):
        if len(parts) >= 2 and parts[0] in english:
            yield english[parts[0]], parts[1]


def tatoeba():
    base = ROOT / "tatoeba"
    thai = {p[0]: p for p in tsv(base / "tha_sentences_detailed.tsv")}
    english = {p[0]: p for p in tsv(base / "eng_sentences_detailed.tsv")}
    authors = set()
    for ids in tsv(base / "tha-eng_links.tsv"):
        th, en = thai.get(ids[0]), english.get(ids[1])
        if th and en:
            authors.update({th[3], en[3]})
            yield en[2], th[2]
    (ROOT / "tatoeba-authors.txt").write_text("\n".join(sorted(a for a in authors if a and a != "\\N")) + "\n", encoding="utf8")


def keep(en, th):
    en, th = en.strip(), th.strip()
    return bool(
        LETTER.search(en) and THAI.search(th) and not THAI.search(en)
        and len(en) <= MAX_CHARS and len(th) <= MAX_CHARS and not ADULT.search(en)
    )


def is_validation(en):
    return int(hashlib.sha256(en.encode()).hexdigest(), 16) % 100 < VALIDATION_PERCENT


def build():
    alt = ROOT / "alt/ALT-Parallel-Corpus-20191206"
    sources = {
        "alt": paired(alt / "data_en.txt", alt / "data_th.txt"),
        "talpco": paired(ROOT / "talpco/data_eng.txt", ROOT / "talpco/data_tha.txt"),
        "tatoeba": tatoeba(),
    }
    seen, counts, train, validation = set(), {}, [], []
    for name, pairs in sources.items():
        for en, th in pairs:
            en, th = en.strip(), th.strip()
            if (en, th) in seen or not keep(en, th):
                continue
            seen.add((en, th))
            counts[name] = counts.get(name, 0) + 1
            rows = validation if is_validation(en) else train
            rows.append({"source": en, "target": th, "target_lang": "Thai", "origin": name})
            rows.append({"source": th, "target": en, "target_lang": "English", "origin": name})
    for name, rows in (("train", train), ("validation", validation)):
        with open(ROOT / f"{name}.jsonl", "w", encoding="utf8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"pairs": counts, "train_rows": len(train), "validation_rows": len(validation)}
    (ROOT / "manifest.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    build()
