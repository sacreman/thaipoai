"""Scrub exported chat pairs before a model that may be published is trained on them.

Links, emails, handles, phone numbers, long numbers, account ids, address details and names become
typed placeholders. A pair is dropped when either side is sexually explicit, or when the two sides
disagree about what was replaced, since that means one side still holds the detail. Prints counts
only, never text.

python -m train.scrub --in train-relay.jsonl --out scrubbed-relay.jsonl --names train-names.jsonl

This lowers what the model can memorise. It does not prove the data is anonymous: a name nobody
introduced and no list holds passes through. Read a sample of the output, and run cmd/extract-audit
on the trained model, before anything is published.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

THAI = "฀-๿"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
PLACEHOLDER = re.compile(r"<(?:url|email|handle|phone|id|address|name)>")
ACCOUNT = r"line|whatsapp|ig|instagram|facebook|fb|wechat|telegram|kakao|ไอดีไลน์|ไลน์ไอดี|ไลน์|ไอดี|เฟส|ไอจี|วีแชท"

# Applied in order. Group 1, where a rule has one, is kept in front of the placeholder.
RULES = (
    ("<email>", re.compile(r"()[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("<url>", re.compile(r"()(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|io|me|app|ly|gg|tv|info|co\.th|in\.th)\b(?:/\S*)?", re.I)),
    ("<handle>", re.compile(r"()(?<![\w@<])@[\w.]{2,}")),
    # an account named after its service: "line id: nok99", "ไลน์ nok99"
    ("<id>", re.compile(rf"((?:{ACCOUNT}|id)\s*(?:id|ไอดี)?\s*[:=：]\s*|(?:{ACCOUNT})\s*id\s+|(?:ไอดีไลน์|ไลน์ไอดี|ไลน์|ไอดี)\s*)[A-Za-z0-9][A-Za-z0-9._-]{{2,}}", re.I)),
    ("<phone>", re.compile(r"()(?<![\d,.])\+?\d(?:[ .-]?\d){8,12}(?![\d,]|\.\d)")),
    ("<id>", re.compile(r"()(?<![\d,.])\d{7,}(?![\d,]|\.\d)")),
    # ponytail: only address details that carry a number. Street and building names pass through;
    # add a gazetteer or a tagger if the sample read shows they matter.
    ("<address>", re.compile(r"((?:บ้านเลขที่|เลขที่|ที่อยู่|\baddress|ซอย|ซ\.|หมู่ที่|หมู่|ห้อง|\bsoi|\bmoo|\broom)\s*[:#]?\s*)\d+(?:/\d+)?", re.I)),
)

NOT_NAMES = {
    "url", "email", "handle", "phone", "id", "address", "name", "sorry", "fine", "good", "here", "home", "back", "happy",
    "tired", "busy", "okay", "not", "going", "from", "just", "still", "the", "baby", "babe", "honey", "darling", "dear",
    "sir", "bro", "boss", "guys", "there", "you", "teerak", "tirak", "thai", "thailand", "bangkok", "pattaya", "phuket",
    "english", "sure", "late", "coming", "working", "sleeping", "hungry", "drunk", "free", "ready", "single", "married",
    "อะไร", "ไร", "จริง", "เล่น", "ว่า", "คือ",
}
PARTICLES = re.compile(r"(?:ค่ะ|คะ|ครับ|คับ|นะ|จ้า|จ๊ะ|ค่า|คร้าบ)+$")
# Where people say a name. Group 1 is the name, and it is then replaced everywhere in that conversation.
# ponytail: patterns and lists, no tagger. Thai names written in Thai are only caught after "ชื่อ" or
# from the names file, because Thai has no capitals and no spaces to find them by.
INTRODUCTIONS = (
    re.compile(r"\b(?:my name is|my name's|name is|call me|they call me)\s+([A-Za-z]{2,})", re.I),
    re.compile(r"\b(?i:i am|i'm|im|this is)\s+([A-Z][a-z]{2,})\b"),
    re.compile(r"\b(?i:hi|hello|hey|dear|thanks|thank you|bye|good ?night|good morning|sorry|miss you|love you)[,!]?\s+([A-Z][a-z]{2,})\b"),
    re.compile(r"(?:พี่|น้อง|คุณ|ป้า|ลุง|เจ๊|เฮีย)\s?([A-Za-z]{3,})"),
    re.compile(rf"ชื่อ(?:เล่น)?(?:ว่า|คือ)?\s*([A-Za-z]{{2,}}|[{THAI}]{{2,8}})(?=\s|$)"),
)

EXPLICIT = re.compile(
    r"\b(?:fuck\w*|cock|dick|pussy|blow ?job|hand ?job|cum|cumming|horny|naked|nudes?|boobs?|tits|titties|nipples?|penis|"
    r"vagina|clit|anal|orgasm\w*|masturbat\w*|jerk(?:ing)? off|(?:suck|lick) (?:it|my|your)|sex|sexting|boom ?boom|"
    r"creampie|dildo|erection|bareback|threesome|deepthroat)\b"
    r"|ควย|หี(?!บ)|เย็ด|เงี่ยน|จิ๋ม|จู๋|น้ำแตก|แตกใน|เสร็จใน|ชักว่าว|ตกเบ็ด|โม๊ก|โม้ก|เซ็กส์|เซ็ก(?!ซี่)|มีอะไรกัน|เอากัน|ร่วมเพศ|"
    r"ดูดนม|โป๊|เปลือย|แก้ผ้า|สวิงกิ้ง",
    re.I,
)


def replace_details(text):
    text = text.translate(THAI_DIGITS)
    for placeholder, pattern in RULES:
        text = pattern.sub(lambda match: (match.group(1) or "") + placeholder, text)
    return text


def name_parts(name):
    """A stored or spoken name as the pieces worth replacing: 'John Smith' gives john and smith."""
    for part in re.split(r"[\s._\-/|]+", PARTICLES.sub("", name.strip())):
        long_enough = len(part) >= (2 if re.search(f"[{THAI}]", part) else 3)
        if long_enough and part.lower() not in NOT_NAMES and not part.isdigit():
            yield part.lower()


def name_pattern(names):
    latin = sorted((n for n in names if n.isascii()), key=len, reverse=True)
    thai = sorted((n for n in names if not n.isascii()), key=len, reverse=True)
    parts = ([rf"\b(?:{'|'.join(map(re.escape, latin))})\b"] if latin else []) + [re.escape(n) for n in thai]
    return re.compile("|".join(parts), re.I) if parts else None


def scrub(rows, names=None, explicit=EXPLICIT, source="src", target="dst"):
    """Returns the rows that are safe to keep, scrubbed, and the counts of what happened.
    names maps a thread to the names already known for it. Rows without a thread stand alone."""
    counts = collections.Counter(rows=len(rows))
    threads = collections.defaultdict(list)
    for index, row in enumerate(rows):
        threads[row.get("thread", f"row-{index}")].append(index)
    kept = [None] * len(rows)
    for thread, indexes in threads.items():
        known = {part for name in (names or {}).get(thread, ()) for part in name_parts(name)}
        for index in indexes:
            for text in (rows[index][source], rows[index][target]):
                for pattern in INTRODUCTIONS:
                    known.update(part for found in pattern.findall(text) for part in name_parts(found))
        pattern = name_pattern(known)
        counts["names_known"] += len(known)
        for index in indexes:
            row = rows[index]
            if explicit.search(row[source]) or explicit.search(row[target]):
                counts["dropped_explicit"] += 1
                continue
            sides = [replace_details(row[field]) for field in (source, target)]
            if pattern:
                sides = [pattern.sub("<name>", side) for side in sides]
            found = [collections.Counter(PLACEHOLDER.findall(side)) for side in sides]
            if found[0] != found[1]:
                counts["dropped_mismatch"] += 1
                continue
            counts.update(found[0])
            kept[index] = {**row, source: sides[0], target: sides[1]}
    kept = [row for row in kept if row is not None]
    counts["written"] = len(kept)
    # A second pass must find nothing left to replace, or a rule is leaving detail behind.
    counts["residual"] = sum(replace_details(row[field]) != row[field] for row in kept for field in (source, target))
    return kept, dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--names", help="JSON lines of {thread, names} from the export")
    parser.add_argument("--explicit", help="a file of extra terms, one per line, that mark a message as explicit")
    parser.add_argument("--fields", default="src,dst")
    args = parser.parse_args()
    read = lambda path: [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]
    # ponytail: the whole file is held in memory, fine to a few million short rows. Stream by thread beyond that.
    names = {entry["thread"]: entry["names"] for entry in read(args.names)} if args.names else None
    explicit = EXPLICIT
    if args.explicit:
        extra = [re.escape(term.strip()) for term in Path(args.explicit).read_text(encoding="utf8").splitlines() if term.strip()]
        explicit = re.compile("|".join([EXPLICIT.pattern, *extra]), re.I)
    kept, counts = scrub(read(args.source), names, explicit, *args.fields.split(","))
    Path(args.out).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf8")
    print(json.dumps(counts))
    sys.exit(1 if counts["residual"] else 0)


if __name__ == "__main__":
    main()
