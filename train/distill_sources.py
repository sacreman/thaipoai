"""Collect the English and Thai sentences that will be sent to the teacher for translation, once.

Reads pinned downloads under data/distill/raw and writes data/distill/sources.jsonl and manifest.json.
Every row is one chat utterance or one sentence of prose, with its source, register and licence.
Nothing here calls a translation service.

Fetch:      python -m train.distill_sources fetch
Build:      python -m train.distill_sources        (candidates, near, mix)
Near only:  python -m train.distill_sources near mix   (re-does near-duplicate removal from filtered.jsonl)
Re-balance: python -m train.distill_sources mix   (after changing a source's take; reuses the filtered rows)
"""

import array
import bz2
import collections
import difflib
import gzip
import hashlib
import itertools
import json
import re
import shutil
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from train.corpus import ADULT, LATIN, THAI, clean, key
from train.release import sha256

OUT = Path("data/distill")
RAW = OUT / "raw"
RETRIEVED = "2026-11-10"
# Wikimedia deletes dated dumps after a few months, Tatoeba rebuilds its export weekly, and GitHub does not promise
# that a commit's archive keeps the same bytes.
NOT_KEPT = re.compile(r"tatoeba/|sgd/|\w*wiki\w*/")
TOKENIZER_SHA256 = "94f70f275460bc1ebc59fa2f8a2fbd2eaa21f490560d0292f43ddcf2d4265d75"  # models/tokenizer-v1, from the corpus release
MAX_CHARS = {"en": 300, "th": 250}
# The trainer drops anything over 128 pieces, and the translation has to fit too. On the existing corpus English
# takes 1.3 times the pieces of its Thai, and up to 2 times for one pair in twenty, so Thai gets the tighter limit.
MAX_PIECES = {"en": 100, "th": 64}
OPENER_WORDS = 3
FRAME_CAP = 20  # most rows a task dialogue may give for one frame, once names, numbers and slot values are masked
MAX_CONTEXT = 200  # characters of the previous turn carried with a dialogue row
PER_MESSAGE = 8  # sentences taken from one question-and-answer message
NEAR_LETTERS = 50  # shorter rows are never treated as near-duplicates: short sentences rightly share most of their letters
NEAR_KEY = 40  # two rows are compared only if their first or their last NEAR_KEY letters match
NEAR_RATIO = 0.8  # and are near-duplicates when difflib calls them this similar
NEAR_GROUP = 4  # ponytail: a row is compared with the first few kept rows that share its key, not all of them

HF = "https://huggingface.co/datasets"
GH = "https://raw.githubusercontent.com"
WM = "https://dumps.wikimedia.org"
TASKMASTER = f"{GH}/google-research-datasets/Taskmaster/d92cb6af3005f1dc09c39e75e7daf4a04905e00b"
MULTIWOZ = f"{GH}/budzianowski/multiwoz/fe0c8e65cfcd8462bd33c86e35f21addc84ca82b/data/MultiWOZ_2.2"
COMMON_VOICE = f"{GH}/common-voice/common-voice/921d9da9c563b412738afaac6eb33d42f50815fb/server/data"
TATOEBA = "https://downloads.tatoeba.org/exports/per_language"
BY_SA = "https://creativecommons.org/licenses/by-sa/4.0/"
BY = "https://creativecommons.org/licenses/by/4.0/"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
MIT = "https://opensource.org/license/mit"
APACHE = "https://www.apache.org/licenses/LICENSE-2.0"

URL = re.compile(r"https?://|www\.|\w\.(?:com|org|net)\b", re.I)
MARKUP = re.compile(r"[{}\[\]<>|\\=_#*~^`]|\w\(|\);|::|:[a-z]{2}:|&\w+;")
# Emails, handles, phone and account numbers, booking or order codes such as 7GAWK763, and the invented US
# addresses, UK postcodes and train numbers of the task dialogues.
CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.\w|(?<!\w)@\w{2,}|\d{7,}|\+?\d(?:[ .-]?\d){8,}|\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{7,}\b|, [A-Z]{2},? \d{5}\b|\b[a-z]{2}\d{1,2}\s?\d[a-z]{2}\b|\bTR\d{4}\b|\(\d{3}\)\s?\d{3}[-.\s]?\d{4}|\b\d{1,5} [A-Za-z0-9][A-Za-z0-9 ]{0,30}?(?i: ave| avenue| rd| road| st| street| dr| drive| lane| ln| blvd| boulevard| way| court| ct| highway| hwy| place| parkway| square| terrace)\b|\b[A-Z]{8}\b|\b(?:[a-z]+\d{2,}|\d{2,}[a-z]{3,})\b")
PUNCTUATION = re.compile(r"[\s!-/:-@\[-`{-~“”‘’…–—]+")
NAMES = re.compile(r"\b(?:Tom|Mary)\b")
WORD = re.compile(r"[a-z0-9']+")
SENTENCE_END = re.compile(r"""[.!?]["'”’)]*$""")
SENTENCE_GAP = re.compile(r"""(?<=[.!?])\s+(?=["'“‘(]?[A-Z])|(?<=[.!?]["'”’)])\s+(?=["'“‘(]?[A-Z])""")
ABBREVIATION = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Prof|Sr|Jr|vs|No|Mt|Gen|Col|Lt|Capt|Sgt|Rev|e\.g|i\.e|U\.S|U\.K|a\.m|p\.m|Intl|Comm|Ltd|Inc|Corp)|\b[A-Z])\.$")
CUT_AT_ABBREVIATION = re.compile(r"\b(?:Intl|Comm|Ltd|Inc|Corp|Jr|Sr|St|Mt|Dr|Mr|Mrs|Ms|Prof|Gen|Col|Lt|No|vs)\.$|^(?:Ltd|Inc|Corp|Jr|Intl)\b")
# A Thai clause that opens with one of these belongs to the sentence before it: a joining word, or a verb with no subject.
THAI_LEANS = "และ|หรือ|แต่|ซึ่ง|โดย|เพื่อ|จึง|ก็|รวมถึง|รวมทั้ง|เช่น|คือ|ว่า|อัน|ทำให้|ก่อให้เกิด|จนกระทั่ง|จน(?!ถึงปัจจุบัน)|เพราะ|เนื่องจาก|แล้ว|ตลอดจน|ทั้งยัง|อีกทั้ง|ขณะที่|ในขณะที่|เว้นแต่|ไม่ว่าจะเป็น|ก่อนจะ|เป็น|ประกอบด้วย|เดิมชื่อ|ของ|กับ|ที่(?!ผ่านมา|นี่|นั่น|สำคัญ|น่า)"
THAI_START = re.compile(rf"[\"“']?(?:{THAI_LEANS})")
# Words that open a new Thai sentence and nothing else: discourse markers and time phrases. A space before one of
# these is the only place Thai prose is cut, because it is the only place a sentence boundary is certain.
THAI_AS_FOR = "ส่วน(?!ใหญ่|หนึ่ง|มาก|ตัว|ที่|ของ|ผสม|ประกอบ|ลด|แบ่ง|ร่วม|สูง|ต่าง|น้อย|เกิน|บน|ล่าง|หน้า|หลัง|กลาง|ใน|นอก|บุคคล|ราชการ)"  # "as for", not the many words it begins
THAI_OPENER = re.compile("(?:นอกจากนี้|นอกจากนั้น|อย่างไรก็ตาม|อย่างไรก็ดี|ทั้งนี้|ดังนั้น|ด้วยเหตุนี้|ต่อมา|หลังจากนั้น|จากนั้น|ปัจจุบัน|ในปัจจุบัน|"
                         f"ก่อนหน้านี้|ล่าสุด|ขณะเดียวกัน|ในขณะเดียวกัน|โดยทั่วไป|ทว่า|{THAI_AS_FOR})")
THAI_VERB = re.compile("เป็น|มี|ได้|คือ|อยู่|ให้|จะ|ถูก|ทำ|ไป|มา|ว่า|เกิด|ใช้|ต้อง|ไม่")  # a Thai line with none of these is nearly always a label
# A row that stops on one of these stops mid-thought. The guards are whole words that happen to end the same way.
THAI_DANGLING = re.compile("(?:(?<!รูป)(?<!ต้น)แบบ|แบบว่า|(?<!ก)ว่า|ก็|แล้วก็|คือ|เช่น|และ|หรือ|เพื่อ|เพราะว่า|แต่ว่า|(?<!หน้า)(?<!สถาน)(?<!พื้น)(?<!เนื้อ)ที่|ที่จะ|(?<!สิ่ง)(?<!ข้าว)ของ|เป็น|โดย|(?<!งบ)ประมาณ|อาทิ|ดังนี้|ดังต่อไปนี้|ได้แก่|\\.\\.|:)$")
THAI_IF = re.compile("(?:ถ้า|หาก|เพื่อ|เนื่องจาก|เพราะ|หลังจาก(?!นั้น|นี้)|นอกจาก(?!นี้|นั้น)|ก่อนที่|กรณีที่|ในกรณี|เมื่อ(?!วาน|คืน|เช้า|กี้|ก่อน|ไม่นาน|ปี|เดือน|วันที่|สัปดาห์|เร็ว|ครู่|สักครู่|ไร|ไหร่))")  # needs a main clause after it
THAI_THEN = re.compile("ก็|จะ|จึง|ต้อง|ทำให้|ยัง")
# A written Thai sentence that opens with one of these verbs has left its subject in the sentence before.
THAI_SUBJECTLESS = re.compile("(?:มี(?!การ|ผู้|คน)|หัวมี|ถิ่นกำเนิดของมัน|รายชื่อ|ปัจจุบัน ?(?:เป็น|มี|ดำรง|อยู่|ใช้|ตั้ง|ยัง|ได้|สังกัด|เล่น|ทำงาน|อาศัย)|หากิน|แสดงนำ|พบได้|พบใน|นิยม|จัดเป็น|อาศัย|ออกดอก|มัก|สามารถ|ลด|เพิ่ม|มองว่า|ผลักดัน|ขยับ|รองรับ|ยัง|จะ|ถูก|เคย|ได้รับ|อยู่|ต้อง|ไม่|ช่วย|ทำหน้าที่|ตั้งอยู่|เริ่ม|มีชื่อ|มีพื้นที่|มีอาณาเขต|ดำรงตำแหน่ง|จบการศึกษา|สำเร็จการศึกษา|บทที่|ตอนที่|ข้อที่)")
ADULT_THAI = re.compile("ร่วมเพศ|ข่มขืน|อวัยวะเพศ|สำเร็จความใคร่|ภาพโป๊|หนังโป๊|เปลือยกาย|โสเภณี|ลามก|ช่องคลอด|องคชาต")  # train.corpus.ADULT is English only
# Titles, royal vocabulary and names that mark a mention of the Thai royal family. Broad on purpose: it also
# catches some foreign monarchs and senior monks, which is the safe side for a rule the owner can reverse.
ROYAL = re.compile("พระบาทสมเด็จ|สมเด็จพระ|สมเด็จเจ้าฟ้า|พระเจ้าอยู่หัว|ในหลวง|พระราชินี|พระบรม|พระราช(?:ดำริ|ทาน|ดำรัส|วัง|พิธี|กรณียกิจ|โอรส|ธิดา|นิพนธ์)|พระองค์เจ้า|หม่อมเจ้า|เจ้าฟ้า|รัชกาลที่|ราชวงศ์จักรี|พระมหากษัตริย์|สถาบันกษัตริย์|กษัตริย์ไทย|ภูมิพล|วชิราลงกรณ์|สิริกิติ์|สิรินธร|มาตรา 112|"
                   r"(?i:King (?:Bhumibol|Vajiralongkorn|Rama|of Thailand)|Queen (?:Sirikit|Suthida)|Princess (?:Sirindhorn|Bajrakitiyabha)|Thai (?:king|queen|royal|monarch)|\bRama (?:[IVX]+|\d)\b|l[eè]se[- ]majest[eé]|Chakri dynasty)")
LEANS = re.compile(r"(?i)(?:yes|no|yeah|yep|yup|nope|nah|ok|okay|sure|it|that|this|those|these|they|them|he|she|and|but|so|or|also|then|there|same|me too|not|thanks|thank you|great|good|perfect|sounds|what about|how about|both|either|neither|right|exactly|really|oh|wow|cool|nice|true|haha|lol|well|because|maybe|probably|definitely|absolutely|alright|fine|one|the first|the second|which one)\b")
UNRESOLVED = re.compile(r"(?:It|This|These|Those|That|Such|Their|Its|The (?:other|same|latter|former)|Another|Since then|Therefore|Thus|Hence|Instead|Rather|Otherwise|Consequently|As a result|In this|More specifically|More importantly|At any rate|After that|Before that|Prior to that|From there|On the one hand|On the other hand)\b")
DEMONSTRATIVE = re.compile(r"(?i)\b(?:this|these|those|such|both|here|there (?:he|she|they)|the (?:latter|former|term|nearest)|above-mentioned|aforementioned)\b|\.\.")  # points outside the sentence, wherever it stands  # an English prose row that opens by pointing at the row before it
POINTER = re.compile(r"(?:See|Visit|Warning|Note|Main article|For more|Below (?:is|are)|Above (?:is|are)|Here (?:is|are)|Asked|Stay tuned|A screenshot|Screenshot|Image|Photo|Picture)\b"
                     r"|.*\b(?i:this (?:page|article|chapter|section|book|guide|module|lesson|wikibook|post)|the (?:next|previous|following) (?:section|chapter|page|exercises?)"
                     r"|listed below|see (?:below|above|item)|round-?up|can be found (?:on|at|here)|accessible here|quoted here|click)\b|.*\((?i:see) |.*; see |.*⟨⟩|.*\bw:|.*[a-z]\.[A-Z]|.*: \d\.$|\w+(?: \w+)? reports\.$")
GAP = re.compile(r"\s[,.;:!?)]|\(\s|\"\"|“”")  # a word the extractor lost leaves its space behind: "replaced by ."
INSTRUCTION = re.compile(r"(?:Give|Provide|Explain|Classify|List|Write|Summari[sz]e|Identify|Describe|Tell|Name|Create|Generate|Make|Categori[sz]e|Rewrite|Translate|Extract|Compare|Suggest|Answer|Convert|Come up|Imagine|Pretend|Act as|Compose|Design|Outline|Rank|Sort|Determine|Choose|Select)\b|จง|ช่วย(?:สรุป|เขียน|แต่ง|อธิบาย|แปล)|สรุป|เขียน|อธิบาย|ยกตัวอย่าง|ระบุ|แต่ง|แปล|เติม|ตอบคำถาม|จบประโยค")
CHATBOT = re.compile(r"(?i)\bas an ai\b|\bas a language model\b|^here (?:are|is)\b|^here's\b|^in (?:summary|conclusion)\b|i hope this helps|\bi can help\b|help you with|\bi(?: a|')m an? (?:ai|assistant|language)|^(?:sure|certainly|of course)[,!]|i'm sorry, but|i apologi[sz]e|let me know if|feel free to|ในฐานะ(?:ปัญญาประดิษฐ์|AI|โมเดล)|^นี่คือ|ต่อไปนี้")
LABEL = re.compile(r"(?:[A-Za-zก-๙][\w.'-]{0,24}(?: [\w.'-]{1,24}){0,2}|คำตอบที่ ?[0-9๐-๙]+)\s?[:：]\s+(?=\S)")  # "Answer: ", "GV: "
OPTION = re.compile(r"[(（]?(?:[ก-ฮ]|[A-Za-z]|[0-9๐-๙]{1,2})[.)）]\s+")  # "(ก) ", "ค. ", "2) "
CJK_BRACKET = re.compile(r"\s*[(（][^()（）]*[぀-ヿ㐀-鿿가-힯][^()（）]*[)）]")
STUB = re.compile(r"\bis a commune\.$|\bwas born (?:in|on)\b|\bis an? [\w' -]{0,40}(?:town|city|village|commune|municipality|district|county|river|lake|mountain|island|species|genus|family|suburb|province|region|footballer|player|politician|actor|actress|singer|writer|company|album|song|movie|film) (?:in|of|from|by|who|that|which)\b"
                  r"|^The \d{4}(?:[-–]\d{2,4})? [\w' -]*season\b|\bcompeted (?:at|in) the\b|\bdied (?:in|on) ")
OLD_FICTION = re.compile(r"[\"“”‘;—]|\b(?i:thou|thee|thy|shall|whilst|upon|hath|ye|nay|madam|sir|mrs?|miss|lord|lady|said|replied|cried|exclaimed|countess|prince|captain|whom|scarcely|presently|fellow|dear|pray|indeed)\b")
# A reply that is only a name, as the task dialogues' invented customers give: "Crystal Minh", "It's Joseph Banter."
NAME_ONLY = re.compile(r"(?i:(?:it'?s|it is|my name is|i am|i'm|this is|name is|sure,?|yes,?|ok,?|okay,?)\s+)*[A-Z][a-z]+ [A-Z][a-z]+[.!]?$")
# Leading words that tie a sentence to the one before without changing what it says. They are removed, and the
# sentence stands alone. (Words that carry the logic, such as "therefore", make the row lean, and it is dropped.)
CONNECTIVE = re.compile(r"(?:However|Also|Additionally|In addition|Furthermore|Moreover|Nevertheless|Nonetheless|Meanwhile|Then|First|Firstly|Second|Secondly|Third|Finally|Next|And|But|So|Yet|Still|Later|"
                        r"At the same time|In fact|For example|For instance|Indeed|Besides|Now|Again|Likewise|Similarly|Overall|Ultimately|In general)\b,?\s+(?=\S)")
THAI_CONNECTIVE = re.compile(f"(?:นอกจากนี้|นอกจากนั้น|อย่างไรก็ตาม|อย่างไรก็ดี|ทั้งนี้|ต่อมา|ขณะเดียวกัน|ในขณะเดียวกัน|{THAI_AS_FOR})[ ,]*")
THAI_LOGIC = re.compile("(?:ดังนั้น|จากนั้น|หลังจากนั้น|ด้วยเหตุนี้|ทว่า|เรื่องนี้|ข่าวนี้|ต่อไปนี้|ด้าน|เหตุผลเป็นเพราะ|ที่สำคัญ|เรียกได้ว่า|นับเป็น|สิ่งที่ตามมา|อีกบริเวณ|อีกสองคน|คำถาม|ติดตามข่าว|บทความ(?:อ่านประกอบ|ที่เกี่ยวข้อง)|จากที่|ตามที่)")  # openers that cannot be removed without losing the point
THAI_BOILERPLATE = re.compile(r"เป็น \d+$|^ด้วยการที่|ได้ที่นี่|ลุ้นรับ|ข้างต้น|(?:ของ|จาก|โดย|แห่ง|ออก|อย่าง) (?:ว่า|ที่|และ|ได้|ก็)|อ่านรายละเอียด|\(รายละเอียด\)|ตามภาพ|อาจหมายถึง$|\(\d+\)$|โดย \(")
CLOSING = re.compile(r"(?i)\bthank|\bthat (?:will|would) be all\b|\bthat's all\b|\bbye\b|\bgoodbye\b|\bproceed further\b")
STAMMER = re.compile(r"(?i)\b(\w+) \1\b|\bi mean\b|\b(?:uh|um|er)\b")  # the marks of a speech recogniser's transcript
BYU_INJECTED = re.compile(r"The \w+ \w+ \w+(?:ed|s) \w+ly\.$")
THAI_MIN = 30  # characters a Thai sentence needs before it may end


# ---------------------------------------------------------------- text to sentences

def english_sentences(text):
    out = []
    for part in SENTENCE_GAP.split(text):
        if out and ABBREVIATION.search(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return out


def thai_sentences(text):
    """Thai has no full stop, and writers put a space between clauses as readily as between sentences. So a
    paragraph is cut only where the cut is certain: at a space before a word that opens a new sentence and
    nothing else (THAI_OPENER), once the sentence so far is THAI_MIN characters with its brackets and quotes
    closed. A paragraph with no such place stays whole, and is dropped later if it is too long. Rows are
    therefore one sentence or several whole ones, never part of one."""
    out, current = [], ""
    for chunk in text.split():
        closed = current.count("(") <= current.count(")") and current.count("“") <= current.count("”") and current.count('"') % 2 == 0
        if len(current) >= THAI_MIN and closed and THAI_OPENER.match(chunk):
            out.append(current)
            current = chunk
        else:
            current = f"{current} {chunk}" if current else chunk
    return out + [current] if current else out


def sentences(text, lang, limit=PER_MESSAGE):
    """Sentences of a question or answer. Messages holding code or lists give nothing."""
    if "```" in text:
        return []
    lines = [line for line in text.split("\n") if line.strip() and not re.match(r"\s*(?:[-*#>|]|\d+[.)])", line)]
    split = thai_sentences if lang == "th" else english_sentences
    return [s for line in lines for s in split(clean(line))][:limit]


WIKI_DROP = re.compile(r"(?is)<!--.*?-->|<ref\b[^>]*/>|<(ref|math|gallery|syntaxhighlight|source|code|pre|nowiki|timeline|imagemap|poem|score)\b.*?</\1\s*>")
WIKI_INNERMOST = re.compile(r"\{\{[^{}]*\}\}|\{\|[^{}]*\|\}")
# Templates that carry words of the sentence: {{w|Paris}} on Wikinews, {{lang|fr|mot}}, {{nowrap|10 km}}. Their last part is kept.
WIKI_WORDS = re.compile(r"(?i)\{\{\s*(?:w|lang(?:-\w+)?|nowrap|nobr|small|em|transl|linktext|wikt|l)\s*\|(?:[^{}|]*\|)*([^{}|]*)\}\}")
WIKI_CONVERT = re.compile(r"(?i)\{\{\s*convert\s*\|\s*([\d.,]+)\s*\|\s*([^{}|]+)[^{}]*\}\}")
WIKI_FILE = re.compile(r"(?i)\[\[(?:file|image|category|ไฟล์|ภาพ|หมวดหมู่):[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]")
WIKI_LINK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]")
WIKI_EXTERNAL = re.compile(r"\[(?:https?:)?//\S+ ?([^\]]*)\]")
WIKI_TAGS = re.compile(r"<[^>]*>|'{2,}|__[A-Z]+__")
WIKI_HOLES = re.compile(r"(?<=\()[\s,;]+|[\s,;]+(?=\))| ?\(\)")  # what a removed pronunciation template leaves in brackets


def wiki_paragraphs(text):
    """Prose paragraphs of a wiki page: templates, tables, files, references, headings and list items go. A
    single line break inside a paragraph is only how the page was typed, so such lines are joined again."""
    text = WIKI_DROP.sub("", text)
    for _ in range(8):
        text, changed = WIKI_INNERMOST.subn("", WIKI_CONVERT.sub(r"\1 \2", WIKI_WORDS.sub(r"\1", text)))
        if not changed:
            break
    text = WIKI_TAGS.sub("", WIKI_EXTERNAL.sub(r"\1", WIKI_LINK.sub(r"\1", WIKI_FILE.sub("", text))))
    text = WIKI_HOLES.sub("", WIKI_HOLES.sub("", text))
    paragraphs, current = [], ""
    for line in text.split("\n") + [""]:
        line = line.strip()
        if not line or line[0] in "*#:;|!={}":
            paragraphs, current = paragraphs + [current] * bool(current), ""
        else:  # Thai is written without spaces, so a break between two Thai letters joins with nothing
            current += ("" if current and THAI.match(current[-1]) and THAI.match(line[0]) else " ") * bool(current) + line
    return paragraphs


# ---------------------------------------------------------------- one reader per download format

def raw(path, mode="rt"):
    path = RAW / path
    opener = bz2.open if path.suffix == ".bz2" else gzip.open if path.suffix == ".gz" else open
    return opener(path, mode, **({"encoding": "utf8"} if "t" in mode else {}))


def tatoeba(path, lang, cc0=None, min_words=0):
    """Sentences under the default CC BY 2.0 FR licence. Those their authors released as CC0 are a different,
    weaker set (aphorisms and literary lines) and are left out."""
    free = set()
    if cc0:
        with raw(cc0) as lines:
            free = {line.split("\t", 1)[0] for line in lines}
    with raw(path) as lines:
        for line in lines:
            number, _, text = line.rstrip("\n").split("\t")[:3]
            if number not in free and len(text.split()) >= min_words:
                yield f"tatoeba-{number}", text, lang


def turns(name, files, list_key="turns", text_key="utterance"):
    """Task dialogues: a JSON list of dialogues, each a list of turns. files is (label, open file) pairs.
    Only what the user said is taken, with the turn before it as context: the system's side is written from
    templates. The fifth value is the turn with its slot values masked, for the frame cap."""
    for label, handle in files:
        for d, dialogue in enumerate(json.load(handle)):
            previous = None
            for t, turn in enumerate(dialogue[list_key]):
                text = turn[text_key]
                if turn["speaker"].upper() == "USER":
                    # The datasets mark where each slot value sits (a city, a time, a film). Masking those gives the frame.
                    spans = [(s["start"], s["exclusive_end"]) for f in turn.get("frames", []) for s in f.get("slots", []) if "start" in s]
                    spans += [(s["start_index"], s["end_index"]) for s in turn.get("segments", [])]
                    shape = text
                    for start, end in sorted(set(spans), reverse=True):
                        shape = shape[:start] + "Xx" + shape[end:]
                    yield f"{name}-{label}-{d}-{t}", text, "en", previous, shape
                previous = text


def sgd(path="sgd/sgd.zip"):
    with zipfile.ZipFile(RAW / path) as archive:
        names = sorted(n for n in archive.namelist() if re.search(r"/(train|dev|test)/dialogues_\d+\.json$", n))
        yield from turns("sgd", ((re.sub(r".*/(\w+)/dialogues_(\d+)\.json", r"\1\2", n), archive.open(n)) for n in names))


def plain_files(name, pattern, **keys):
    paths = sorted((RAW / name).glob(pattern))
    yield from turns(name, ((p.stem, open(p, encoding="utf8")) for p in paths), **keys)


def abcd(path="abcd/abcd_v1.1.json.gz"):
    """What the customer wrote, with the line before it as context. The agent's side follows a script."""
    with raw(path) as handle:
        for split, conversations in json.load(handle).items():
            for conversation in conversations:
                previous = None
                for t, (speaker, text) in enumerate(conversation["original"]):
                    if speaker == "customer":
                        yield f"abcd-{conversation['convo_id']}-{t}", text, "en", previous
                    if speaker != "action":  # button presses and system log lines
                        previous = text


def byu(path="byu/dataset.json"):
    """Informal chat. People send one thought as several short messages, so the messages one person sent in a
    row are joined into one utterance, and what the other person said just before goes along as context."""
    with raw(path) as handle:
        for conversation, body in json.load(handle).items():
            previous = None
            for t, turn in enumerate(body["messages"]):
                text = " ".join(m["text"].strip() for m in turn if not BYU_INJECTED.match(m["text"].strip()))  # test lines the organisers planted
                yield f"byu-{conversation}-{t}", text, "en", previous
                previous = text


def soda(every=4):
    """Dialogue turns only, not the narrative, each with the turn before it as context. One dialogue in `every`:
    the whole set is 11 million turns."""
    import pyarrow.parquet as pq

    for split in ("train", "valid", "test"):
        for batch in pq.ParquetFile(RAW / f"soda/{split}.parquet").iter_batches(columns=["dialogue", "original_index"]):
            for dialogue, index in zip(batch["dialogue"].to_pylist(), batch["original_index"].to_pylist()):
                if index % every == 0:
                    for t, text in enumerate(dialogue):
                        yield f"soda-{index}-{t}", text, "en", dialogue[t - 1] if t else None


REFERS = re.compile(r"(?i)\b(?:reference text|paragraph|passage|the text|below|above|following|given|summary)\b")


def dolly(path="dolly/databricks-dolly-15k.jsonl"):
    """The questions people wrote, unless they ask about a text that is not here. The answers and the context
    field (Wikipedia excerpts) are left out."""
    with raw(path) as lines:
        for i, line in enumerate(lines):
            for n, sentence in enumerate(sentences(json.loads(line)["instruction"], "en")):
                if sentence.endswith("?") and not REFERS.search(sentence):
                    yield f"dolly-{i}-{n}", sentence, "en"


def lines_of(name, path, lang):
    with raw(path) as lines:
        for i, line in enumerate(lines):
            yield f"{name}-{i}", line, lang


def common_voice_wiki():
    """Single sentences Common Voice took from English Wikipedia. Very short ones say little, and one that opens
    with "They" has lost who they are."""
    for label, path in (("a", "commonvoice/en-wiki.en.txt"), ("b", "commonvoice/en-wiki.en.rerun-2021-11.txt")):
        for row_id, text, lang in lines_of(f"commonvoice_wiki-{label}", path, "en"):
            if len(text.split()) >= 7 and not text.startswith("They ") and re.search(r"[a-z]{3}[.!?]\s*$", text):  # not cut off at an initial
                yield row_id, text, lang


def wiki_pages(handle):
    """(page id, wikitext) for every article in a MediaWiki XML dump, skipping redirects and other namespaces."""
    for _, page in ET.iterparse(handle):
        if page.tag.endswith("}page"):
            ns = page.tag[:-4]
            if page.findtext(f"{ns}ns") == "0" and page.find(f"{ns}redirect") is None:
                yield page.findtext(f"{ns}id"), page.findtext(f"{ns}revision/{ns}text") or ""
            page.clear()


def wiki(name, lang, per_page=None, must_contain=""):
    """The first per_page sentences of each article, so no article dominates and topics stay spread.
    Thai is taken whole: it is scarce, and the longest article gives well under 1% of the Thai rows."""
    split = thai_sentences if lang == "th" else english_sentences
    for path in sorted((RAW / name).glob("*.bz2")):
        with bz2.open(path, "rb") as handle:
            for page_id, text in wiki_pages(handle):
                if must_contain in text.lower():
                    found = (s for p in wiki_paragraphs(text) for s in split(clean(p)))
                    for n, sentence in enumerate(itertools.islice(found, per_page)):
                        yield f"{name}-{page_id}-{n}", sentence, lang


def global_voices(path="globalvoices/en.zip"):
    """The site's own sentences. Quoted blog posts and tweets belong to other people and are left out,
    as are picture credits and the speaker labels of interviews.
    The row id holds the article's file name, which is its date and address on globalvoices.org."""
    def own(node):
        for child in node:
            if child.tag == "s":
                yield child
            elif child.tag not in ("QUOTE", "TITLE"):
                yield from own(child)

    with zipfile.ZipFile(RAW / path) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".xml")):
            try:
                document = ET.fromstring(archive.read(name))
            except ET.ParseError:  # a few files in the archive are empty
                continue
            sentences_here = list(own(document))
            for s in sentences_here if len(sentences_here) >= 6 else ():  # shorter posts are one-line digests of other people's blogs
                text = "".join(s.itertext())
                if len(text.split()) >= 12 and not re.match(r"\s*(?:[A-Z]{1,4}:|(?:Image|Photo|Photograph|Picture|Screenshot|Cartoon|Video)\b)", text):  # shorter lines are captions and asides
                    yield f"globalvoices-{Path(name).stem}-{s.get('id')}", text, "en"


IAPP_ASKS = re.compile("อะไร|ใด|ใคร|ไหน|เท่าไร|เท่าไหร่|กี่|หรือไม่|ไหม|อย่างไร|ยังไง|เมื่อไร|เมื่อไหร่|ทำไม")
IAPP_POINTS = re.compile("นี้|ดังกล่าว|ในที่นี้|^(?:เธอ|เขา|เกาะ|รายการ|ละคร)|^ในปี .{0,12}(?:เธอ|เขา)")
WIRE = re.compile(r"รอยเตอร์|เอพี|เอเอฟพี|Reuters|\bAP\b|\bAFP\b|Associated Press|Agence France")


def voa_thai(path="mot/tha_voathai.tgz"):
    """Voice of America's Thai news articles, from Multilingual Open Text. VOA's own writing is in the public
    domain, but what it takes from Reuters, AP and AFP is not, so an article that names any of them anywhere
    is left out whole. So are the oldest articles (numbers under a million), whose words are spaced wrongly.
    Captions and programme listings (the audio, video and photo records) are not read."""
    with tarfile.open(RAW / path) as archive:
        for member in archive:
            if member.isfile() and "/article/" in member.name:
                article = json.load(archive.extractfile(member))
                number = article["filename"].rsplit("_", 1)[-1]
                if not (number.isdigit() and int(number) < 1_000_000) and not WIRE.search("\n".join(article["paragraphs"])):
                    kept = (p for p in article["paragraphs"] if "ที่มา" not in p[:12])
                    for n, sentence in enumerate(s for p in kept for s in thai_sentences(clean(p))):
                        yield f"voathai-{number}-{n}", sentence, "th"


def blognone(path="blognone/train.parquet"):
    """Technology news posts. The closing line that credits the story's source is left out."""
    import pyarrow.parquet as pq

    for i, content in enumerate(pq.read_table(RAW / path, columns=["content"])["content"].to_pylist()):
        paragraphs = [p for p in content.split("\n") if p.strip() and not p.lstrip().startswith("ที่มา")]
        for n, sentence in enumerate(s for p in paragraphs for s in thai_sentences(clean(p))):
            yield f"blognone-{i}-{n}", sentence, "th"


def iapp(pattern="iapp/*.parquet"):
    """Questions people wrote in Thai about Wikipedia articles. The articles themselves are not taken, so a
    question that points into its article ("this show", "she") or has no question word is left out."""
    import pyarrow.parquet as pq

    for path in sorted(RAW.glob(pattern)):
        table = pq.read_table(path, columns=["question_id", "question"]).to_pylist()
        for r in table:
            if IAPP_ASKS.search(r["question"]) and not IAPP_POINTS.search(r["question"]):
                yield f"iapp-{r['question_id']}", r["question"], "th"


# ---------------------------------------------------------------- the sources and the mix

@dataclass(frozen=True)
class Source:
    name: str
    title: str
    register: str
    licence: str
    licence_url: str
    credit: str  # the attribution line the release must carry
    homepage: str
    take: object  # rows in the mix, or None for every usable row
    extract: object
    prose: bool = False  # written text: a sentence starts with a capital and ends with a full stop
    opener_cap: int = 0  # most rows that may start with the same few words
    dialogue: bool = False  # rows carry the turn before them, and one that leans on it and has none is dropped
    frames: bool = False  # a task dialogue: capped per masked frame, closings and recogniser garble dropped
    asked: bool = False  # questions and answers written to or for an assistant: instructions and chatbot voice go
    royal: bool = False  # encyclopaedia and news prose, the only places a mention of the Thai royal family stays
    stubs: bool = False  # keep one in five of the encyclopaedia's stock sentences ("was born in", "is a town in")
    strict: bool = False  # a demonstrative anywhere in the sentence drops it: for sources of loose single sentences
    modern: bool = False  # leave out sentences that read as old fiction or quote someone speaking
    repairs: str = ""  # what the extractor did to the text before any filter, recorded in the manifest


PLAIN = Source("", "", "", "", "", "", "", None, None)


WIKIMEDIA = "contributors, each row's id holds the page id, https://{}/?curid=<page id>"
SOURCES = (
    # Scarce chat first, encyclopaedias last, so the plentiful sources lose every duplicate.
    # A take of None gives every usable row. The capped sources are clean but repeat themselves: Tatoeba and
    # SODA near 100,000, Simple English Wikipedia 300,000, 150,000 Common Voice sentences from Wikipedia, and
    # 4,000 of the iApp questions.
    Source("byu_chitchat", "BYU Chit-Chat Dataset", "informal chat", "MIT", MIT,
           "Copyright (c) 2019 Perception, Control and Cognition Lab, Brigham Young University",
           "https://github.com/BYU-PCCL/chitchat-dataset", None, byu, dialogue=True),
    Source("abcd", "Action-Based Conversations Dataset", "customer-service chat", "MIT", MIT,
           "Copyright (c) 2021 ASAPP Research", "https://github.com/asappresearch/abcd", None, abcd, dialogue=True, frames=True),
    Source("multiwoz", "MultiWOZ 2.2", "task dialogue", "Apache-2.0", APACHE,
           "Budzianowski and others 2018; Zang and others 2020. Also offered under MIT in the GitHub repository",
           "https://github.com/budzianowski/multiwoz", None,
           lambda: plain_files("multiwoz", "*.json"), dialogue=True, frames=True),
    Source("taskmaster2", "Taskmaster-2", "spoken task dialogue", "CC BY 4.0", BY,
           "Byrne, Krishnamoorthi, Ganesh, Dubey, Kim and Cedilnik, Google LLC",
           "https://github.com/google-research-datasets/Taskmaster/tree/master/TM-2-2020", None,
           lambda: plain_files("taskmaster2", "*.json", list_key="utterances", text_key="text"), dialogue=True, frames=True),
    Source("taskmaster3", "Taskmaster-3", "task dialogue", "CC BY 4.0", BY,
           "Byrne, Krishnamoorthi and Ganesh, Google LLC",
           "https://github.com/google-research-datasets/Taskmaster/tree/master/TM-3-2020", None,
           lambda: plain_files("taskmaster3", "*.json", list_key="utterances", text_key="text"), dialogue=True, frames=True),
    Source("sgd", "Schema-Guided Dialogue", "task dialogue", "CC BY-SA 4.0", BY_SA,
           "Rastogi, Zang, Sunkara, Gupta and Khaitan, Google LLC",
           "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue", None, sgd, dialogue=True, frames=True),
    Source("dolly", "databricks-dolly-15k", "questions and answers", "CC BY-SA 3.0", "https://creativecommons.org/licenses/by-sa/3.0/",
           "Databricks, Inc.", "https://huggingface.co/datasets/databricks/databricks-dolly-15k", None, dolly, asked=True),
    Source("iapp", "iApp Wiki QA, questions", "questions and answers", "MIT", MIT,
           "Copyright (c) 2021 iApp Technology Co., Ltd.", "https://github.com/iapp-technology/iapp-wiki-qa-dataset", 4_000, iapp),
    Source("tatoeba_th", "Tatoeba, Thai", "everyday sentences", "CC BY 2.0 FR", "https://creativecommons.org/licenses/by/2.0/fr/",
           "Tatoeba contributors, each row's id holds the sentence number, https://tatoeba.org/sentences/show/<number>",
           "https://tatoeba.org", None, lambda: tatoeba("tatoeba/tha_sentences_detailed.tsv", "th")),
    Source("tatoeba_en", "Tatoeba, English", "everyday sentences", "CC BY 2.0 FR", "https://creativecommons.org/licenses/by/2.0/fr/",
           "Tatoeba contributors, each row's id holds the sentence number, https://tatoeba.org/sentences/show/<number>",
           "https://tatoeba.org", 100_000,
           lambda: tatoeba("tatoeba/eng_sentences_detailed.tsv", "en", "tatoeba/eng_sentences_CC0.tsv.bz2", min_words=6), opener_cap=10, modern=True),
    Source("soda", "SODA", "social chat, written by a language model", "CC BY 4.0", BY,
           "Kim and others, Allen Institute for AI, 2023", "https://huggingface.co/datasets/allenai/soda", 100_000, soda, dialogue=True, opener_cap=300),
    Source("globalvoices", "Global Voices, English (OPUS GlobalVoices 2018Q4)", "news features", "CC BY 3.0", "https://creativecommons.org/licenses/by/3.0/",
           "Global Voices authors, each row's id holds the article's date and address, https://globalvoices.org/<yyyy>/<mm>/<dd>/<address>/",
           "https://globalvoices.org", None, global_voices, prose=True, strict=True),
    Source("enwikinews", "English Wikinews", "news", "CC BY 4.0", BY,
           "Wikinews " + WIKIMEDIA.format("en.wikinews.org"), "https://en.wikinews.org", None,
           lambda: wiki("enwikinews", "en", None, "{{publish"), prose=True, royal=True),
    Source("enwikivoyage", "English Wikivoyage", "travel", "CC BY-SA 4.0", BY_SA,
           "Wikivoyage " + WIKIMEDIA.format("en.wikivoyage.org"), "https://en.wikivoyage.org", None,
           lambda: wiki("enwikivoyage", "en", 40), prose=True, strict=True),
    Source("blognone", "Blognone posts, January 2020 to April 2023", "technology news", "CC BY 3.0 TH", "https://creativecommons.org/licenses/by/3.0/th/",
           "Blognone and the post's author. Each row's id holds the post's row number in Noxturnix/blognone-20230430, which gives its author, title and date",
           "https://www.blognone.com", None, blognone, prose=True),
    Source("voathai", "Voice of America Thai, articles (Multilingual Open Text 1.11)", "news", "Public domain", "https://www.voanews.com/p/5338.html",
           "Voice of America, each row's id holds the article number, https://www.voathai.com/a/<number>.html. "
           "Collected by Palen-Michel, Kim and Lignos, Multilingual Open Text, whose collection is CC BY 4.0",
           "https://github.com/bltlab/mot", None, voa_thai, prose=True, royal=True),
    Source("thwiki", "Thai Wikipedia", "encyclopaedic", "CC BY-SA 4.0", BY_SA,
           "Wikipedia " + WIKIMEDIA.format("th.wikipedia.org"), "https://th.wikipedia.org", None,
           lambda: wiki("thwiki", "th"), prose=True, royal=True),
    Source("simplewiki", "Simple English Wikipedia", "plain-language encyclopaedic", "CC BY-SA 4.0", BY_SA,
           "Wikipedia " + WIKIMEDIA.format("simple.wikipedia.org"), "https://simple.wikipedia.org", 300_000,
           lambda: wiki("simplewiki", "en", 20), prose=True, strict=True, royal=True, stubs=True),
    Source("commonvoice_wiki", "Common Voice sentences taken from English Wikipedia", "encyclopaedic", "CC0 1.0", CC0,
           "None required", "https://github.com/common-voice/common-voice", 150_000, common_voice_wiki, prose=True, strict=True, stubs=True),
)


# ---------------------------------------------------------------- filters

def template(text):
    """Two rows with the same template are one sentence apart from case, digits, punctuation, and which of
    Tatoeba's Tom and Mary it names."""
    return PUNCTUATION.sub("", re.sub(r"\d", "0", key(NAMES.sub("Tom", text))))


def digest(text):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest())


def shingles(text, size=8):
    words = WORD.findall(key(text))
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def opener(text):
    return " ".join(WORD.findall(re.sub(r"\d", "0", NAMES.sub("name", text)).casefold())[:OPENER_WORDS])


def frame(text):
    """A task-dialogue line with what varies masked: numbers, and capitalised words inside the sentence (names,
    places, slot values). "Book a table at Nopa for 7 pm" and "Book a table at Zuni for 8 pm" share a frame."""
    words = text.split()
    masked = ["X" if w.startswith("Xx") or (i and w[:1].isupper() and not w.startswith("I") and words[i - 1][-1] not in ".?!") else w.casefold()
              for i, w in enumerate(words)]
    return PUNCTUATION.sub(" ", re.sub(r"(?:X\W*)+", "X ", re.sub(r"\d+", "0", " ".join(masked)))).strip()


def prepare(text, lang, source=PLAIN):
    """Clean a row and make the repairs that are safe, returning the text and the names of the repairs made.
    Everywhere: brackets holding Chinese, Japanese or Korean go, and a doubled Thai sara e becomes the sara ae
    it was typed for. In written and question-and-answer sources: a "Label: " or "(ก) " in front of the sentence
    goes, and so does a leading connective that only ties it to the sentence before."""
    text, repairs = clean(text), []
    written = source.prose or source.asked
    steps = [("bracket removed", lambda t: CJK_BRACKET.sub("", t)), ("sara ae repaired", lambda t: t.replace("เเ", "แ"))]
    steps += [("label removed", lambda t: LABEL.sub("", t, count=1) if LABEL.match(t) else t),
              ("option marker removed", lambda t: OPTION.sub("", t, count=1) if OPTION.match(t) else t),
              ("connective removed", lambda t: uncoupled(t, lang))] * written
    for name, fix in steps:
        fixed = clean(fix(text))
        if fixed != text:
            text, repairs = fixed, repairs + [name]
    return text, repairs


def uncoupled(text, lang):
    pattern = THAI_CONNECTIVE if lang == "th" else CONNECTIVE
    rest = pattern.sub("", text, count=1) if pattern.match(text) else text
    return rest[:1].upper() + rest[1:] if rest != text else text


def reject(text, lang, source=PLAIN, row_id=""):
    """Why a cleaned row is not worth translating, or None."""
    solid = text.replace(" ", "")
    words = text.split()
    thai, prose = lang == "th", source.prose
    if CONTACT.search(text) or (source.dialogue and NAME_ONLY.match(text)):
        return "contact details"
    if not solid or URL.search(text) or MARKUP.search(text):
        return "not a sentence"
    script = THAI if thai else LATIN
    if len(script.findall(text)) < 0.7 * len(solid) or (not thai and THAI.search(text)):
        return "not a sentence"
    if text.count('"') % 2 or text.count("(") != text.count(")") or text.count("“") != text.count("”"):
        return "not a sentence"  # a quotation or bracket that opens and never closes, or the reverse
    if not thai and len(words) >= 4 and sum(w[0].isupper() for w in words) > 0.7 * len(words):
        return "not a sentence"  # headings, titles and lists of names
    if prose and not thai and (not (text[0].isupper() and SENTENCE_END.search(text)) or CUT_AT_ABBREVIATION.search(text)):
        return "not a sentence"
    if prose and thai and (not THAI_VERB.search(text) or not re.match("[\"“']?[ก-ฮเ-ไA-Z]", text) or THAI_BOILERPLATE.search(text)):
        return "not a sentence"
    if thai and " - " in text:
        return "not a sentence"  # "song - artist", or two speakers on one line
    if ADULT.search(text) or ADULT_THAI.search(text):
        return "adult"
    if not source.royal and ROYAL.search(text):
        return "royal family"
    if (len(words) < 3 if not thai else len(text) < (40 if prose else 8)):
        return "too short"
    if len(text) > MAX_CHARS[lang]:
        return "too long"
    if thai and (THAI_DANGLING.search(text) or (THAI_IF.match(text) and not THAI_THEN.search(text))):
        return "unfinished"
    if thai and (THAI_START.match(text) or THAI_LOGIC.match(text) or (prose and THAI_SUBJECTLESS.match(THAI_OPENER.sub("", text, count=1).lstrip(" ,")))):
        return "leans on the row before"
    if source.strict and DEMONSTRATIVE.search(text):
        return "leans on the row before"
    if (prose or source.asked) and not thai and (UNRESOLVED.match(text) or POINTER.match(text)):
        return "leans on the row before" if UNRESOLVED.match(text) else "pointer"
    if not thai and GAP.search(text):
        return "words missing"
    if source.asked and (INSTRUCTION.match(text) or CHATBOT.search(text) or (not thai and not SENTENCE_END.search(text))):
        return "instruction or chatbot voice"
    if source.stubs and STUB.search(text) and digest(row_id) % 5:
        return "stock sentence"
    if source.modern and OLD_FICTION.search(text):
        return "old fiction"
    if source.frames and STAMMER.search(text):
        return "garbled speech"
    if source.frames and len(words) <= 8 and CLOSING.search(text):
        return "only a closing"
    return None


def sift(source, batch, pieces, blocked, benchmark, seen, openers, counts):
    """Clean one batch of (id, text, language) or (id, text, language, turn before) and return the rows that
    pass every filter, counting the rest."""
    rows = []
    shapes = {}
    for row_id, text, lang, *more in batch:
        text, repairs = prepare(text, lang, source)
        context = clean(more[0]) if more and more[0] else None
        shapes[row_id] = more[1] if len(more) > 1 else text
        if context and (len(context) > MAX_CONTEXT or CONTACT.search(context) or URL.search(context) or MARKUP.search(context)
                        or ADULT.search(context) or ROYAL.search(context)):
            context = None
        reason = reject(text, lang, source, row_id)
        if not reason and source.dialogue and not context and (len(text.split()) < 5 or LEANS.match(text)):
            reason = "needs the turn before"
        if not reason and context and key(text) in key(context):
            reason = "repeats the turn before"
        if reason:
            counts[reason] += 1
        else:
            rows.append({"id": row_id, "text": text, "lang": lang, "source": source.name, "register": source.register, "licence": source.licence}
                        | ({"context": context} if context else {}) | ({"repairs": repairs} if repairs else {}))
    kept = []
    for r, size in zip(rows, pieces([r["text"] for r in rows])):
        mark = digest(template(r["text"]))
        start = (source.name, opener(r["text"]))
        shape = (source.name, frame(shapes[r["id"]]))
        reason = ("too long" if size > MAX_PIECES[r["lang"]] else "already in project data" if mark in blocked
                  else "benchmark overlap" if mark in benchmark or (r["lang"] == "en" and not benchmark.isdisjoint(shingles(r["text"])))
                  else "duplicate" if mark in seen
                  else "repeated opener" if source.opener_cap and openers[start] >= source.opener_cap
                  else "repeated frame" if source.frames and openers[shape] >= FRAME_CAP else None)
        if reason:
            counts[reason] += 1
            continue
        seen.add(mark)
        openers[start] += 1
        openers[shape] += 1
        kept.append(r | {"pieces": size})
    return kept


# ---------------------------------------------------------------- text that must not be bought again

def jsonl(path):
    with open(path, encoding="utf8") as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def texts(value):
    """Every string inside a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, dict)):
        for item in (value.values() if isinstance(value, dict) else value):
            yield from texts(item)


def project_text(root=Path(".")):
    """Templates of everything the project already holds: the corpus, the older public pairs, the suites
    and the synthetic chat. Context lines and intended meanings count too."""
    files = [*root.glob("data/public/corpus/*.jsonl"), *root.glob("data/public/*.jsonl"),
             *root.glob("suites/*.jsonl"), *root.glob("data/synthetic-chat-v1/*.jsonl")]
    return {digest(template(t)) for path in files for row in jsonl(path) for t in texts(row)}


def benchmark_text(root=Path(".")):
    """Exact templates and eight-word runs of the benchmarks' sentences, and of the older public pairs
    (ALT is English Wikinews). FLORES-200 stands in for FLORES+, which is gated: same sentences."""
    found = []
    with tarfile.open(RAW / "benchmarks/flores200_dataset.tar.gz") as archive:
        for member in archive:
            if re.search(r"/(eng_Latn|tha_Thai)\.(dev|devtest)$", member.name):
                found += archive.extractfile(member).read().decode("utf8").splitlines()
    found += [t for row in jsonl(RAW / "benchmarks/wmt24pp-en-th_TH.jsonl") for t in (row["source"], row["target"])]
    found += [t for path in root.glob("data/public/*.jsonl") for row in jsonl(path) for t in (row["source"], row["target"])]
    marks = {digest(template(t)) for t in found}
    return marks | {s for t in found if not THAI.search(t) for s in shingles(t)}


# ---------------------------------------------------------------- downloads

def downloads():
    """path under data/distill/raw -> url. Checksums are in PINS."""
    files = {
        "tatoeba/eng_sentences_detailed.tsv": f"{TATOEBA}/eng/eng_sentences_detailed.tsv.bz2",
        "tatoeba/tha_sentences_detailed.tsv": f"{TATOEBA}/tha/tha_sentences_detailed.tsv.bz2",
        "tatoeba/eng_sentences_CC0.tsv.bz2": f"{TATOEBA}/eng/eng_sentences_CC0.tsv.bz2",
        "sgd/sgd.zip": "https://codeload.github.com/google-research-datasets/dstc8-schema-guided-dialogue/zip/e852981ae34990f4358979625854259302feaa78",
        "abcd/abcd_v1.1.json.gz": f"{GH}/asappresearch/abcd/6b8700ce67c6b37b062dd7a60abc76d7ef832a97/data/abcd_v1.1.json.gz",
        "byu/dataset.json": f"{GH}/BYU-PCCL/chitchat-dataset/ff8b113e06146003fc22c91202a03786297d1709/chitchat_dataset/dataset.json",
        "dolly/databricks-dolly-15k.jsonl": f"{HF}/databricks/databricks-dolly-15k/resolve/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl",
        "benchmarks/flores200_dataset.tar.gz": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
        "benchmarks/wmt24pp-en-th_TH.jsonl": f"{HF}/google/wmt24pp/resolve/fd7405c06494bc66a57b25f55d217a72f96e60dc/en-th_TH.jsonl",
        "globalvoices/en.zip": "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/raw/en.zip",
        "mot/tha_voathai.tgz": "https://github.com/bltlab/mot/releases/download/v1.11/tha_voathai.tgz",
        "blognone/train.parquet": f"{HF}/Noxturnix/blognone-20230430/resolve/6f0562e9c1bde3e76916825409459a6edeaaec23/data/train-00000-of-00001-911f9b20518fd37c.parquet",
    }
    files |= {f"iapp/{s}.parquet": f"{HF}/iapp/iapp_wiki_qa_squad/resolve/c4e4341d5dca5bbf29e5c00f53e8391050050cd1/data/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")}
    files |= {f"soda/{s}.parquet": f"{HF}/allenai/soda/resolve/fdc848ab0183208ea7808206c91c724414d0a071/{s}.parquet" for s in ("train", "valid", "test")}
    files |= {f"taskmaster2/{n}.json": f"{TASKMASTER}/TM-2-2020/data/{n}.json"
              for n in ("flights", "food-ordering", "hotels", "movies", "music", "restaurant-search", "sports")}
    files |= {f"taskmaster3/data_{i:02}.json": f"{TASKMASTER}/TM-3-2020/data/data_{i:02}.json" for i in range(20)}
    files |= {f"multiwoz/{s}_{i:03}.json": f"{MULTIWOZ}/{s}/dialogues_{i:03}.json"
              for s, n in (("train", 17), ("dev", 2), ("test", 2)) for i in range(1, n + 1)}
    files |= {f"commonvoice/en-{n}": f"{COMMON_VOICE}/en/{n}" for n in ("wiki.en.txt", "wiki.en.rerun-2021-11.txt")}
    dumps = [(w, f"{w}-20261101-pages-articles.xml.bz2") for w in ("simplewiki", "thwiki", "enwikivoyage")]
    dumps += [("enwikinews", f"enwikinews-20261101-pages-articles1.xml-{p}.bz2") for p in ("p1p1500000", "p1500001p3000000", "p3000001p3095843")]
    return files | {f"{w}/{n}": f"{WM}/{w}/20261101/{n}" for w, n in dumps}


def fetch():
    for path, url in downloads().items():
        target = RAW / path
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            unpack = url.endswith(".bz2") and not path.endswith(".bz2")
            request = urllib.request.Request(url, headers={"User-Agent": "thaipoai-data-build/0.1 (github.com/sacreman/thaipoai)"})
            with urllib.request.urlopen(request, timeout=300) as response, open(partial, "wb") as out:
                shutil.copyfileobj(bz2.open(response) if unpack else response, out)
            partial.rename(target)
        if sha256(target) != PINS.get(path):
            raise SystemExit(f"{path} does not match its pinned checksum: {sha256(target)}")
        print(f"{path} verified")


# ---------------------------------------------------------------- build

def rank(row_id):
    """A fixed pseudo-random order, so taking part of a source is a spread sample and repeats exactly."""
    return digest("rank " + row_id)


def percentile(values, share):
    return sorted(values)[int(share * (len(values) - 1))] if values else 0


def verified_files():
    """Every input with its address, checksum and size. `upstream` says whether the same bytes can be fetched
    again: "pinned" (an immutable revision) or "not kept" (the publisher replaces or deletes it, so the copy in data/distill/raw is the record)."""
    files = {}
    for path, url in downloads().items():
        upstream = "not kept" if NOT_KEPT.match(path) else "pinned"
        files[path] = {"url": url, "sha256": sha256(RAW / path), "bytes": (RAW / path).stat().st_size, "upstream": upstream}
        if files[path]["sha256"] != PINS.get(path):
            raise SystemExit(f"{path} does not match its pinned checksum")
    return files


def candidates():
    """Every row that passes the row-by-row filters, to filtered.jsonl, and what was dropped and why, to filtered.json."""
    from train.tokenizer import DEFAULT, Tokenizer

    if sha256(DEFAULT) != TOKENIZER_SHA256:
        raise SystemExit(f"{DEFAULT} is not the tokenizer this build's length limits were measured with")
    encode = Tokenizer().encode
    files = verified_files()
    blocked, benchmark = project_text(), benchmark_text()
    seen, openers, report = set(), collections.Counter(), {}
    with open(OUT / "filtered.jsonl", "w", encoding="utf8") as out:
        for source in SOURCES:
            counts = collections.Counter()
            for batch in itertools.batched(source.extract(), 20000):
                counts["extracted"] += len(batch)
                for r in sift(source, batch, lambda t: map(len, encode(t)), blocked, benchmark, seen, openers, counts):
                    counts["usable"] += 1
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
            report[source.name] = dict(counts)
            print(source.name, report[source.name], file=sys.stderr)
    (OUT / "filtered.json").write_text(json.dumps({"files": files, "sources": report}, indent=1) + "\n", encoding="utf8")


def near_keys(norm):
    return digest("a" + norm[:NEAR_KEY]), digest("z" + norm[-NEAR_KEY:])


def alike(a, b):
    match = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return match.real_quick_ratio() >= NEAR_RATIO and match.quick_ratio() >= NEAR_RATIO and match.ratio() >= NEAR_RATIO


def without_near_duplicates(read, dropped):
    """The rows of read() in order, minus any of NEAR_LETTERS letters or more that begins or ends like an
    earlier kept row and is NEAR_RATIO similar to it: the same sentence with a name, a price or a clause
    changed. Letters are those of template(). read is called twice, first to learn which beginnings and
    endings occur more than once, so only those rows are held in memory."""
    once, shared = set(), set()
    for r in read():
        norm = template(r["text"])
        if len(norm) >= NEAR_LETTERS:
            for k in near_keys(norm):
                (shared if k in once else once).add(k)
    del once
    groups = collections.defaultdict(list)
    for r in read():
        norm = template(r["text"])
        mine = [groups[k] for k in near_keys(norm) if k in shared] if len(norm) >= NEAR_LETTERS else []
        if any(alike(norm, other) for group in mine for other in group):
            dropped[r["source"]] += 1
            continue
        for group in mine:
            if len(group) < NEAR_GROUP:
                group.append(norm)
        yield r


def near():
    """filtered.jsonl without its near-duplicates, to candidates.jsonl, with the counts in candidates.json.
    Sources are in the order of SOURCES, so the earlier source keeps the row, as with exact duplicates."""
    counted = json.loads((OUT / "filtered.json").read_text(encoding="utf8"))
    dropped, kept = collections.Counter(), collections.Counter()
    with open(OUT / "candidates.jsonl", "w", encoding="utf8") as out:
        for r in without_near_duplicates(lambda: jsonl(OUT / "filtered.jsonl"), dropped):
            kept[r["source"]] += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, counts in counted["sources"].items():
        counts["near duplicate"], counts["usable"] = dropped[name], kept[name]
    (OUT / "candidates.json").write_text(json.dumps(counted, indent=1) + "\n", encoding="utf8")
    print(json.dumps({"near duplicates": dict(dropped), "usable": sum(kept.values())}), file=sys.stderr)


def mix():
    """Take each source's share of the candidates and write sources.jsonl and manifest.json."""
    counted = json.loads((OUT / "candidates.json").read_text(encoding="utf8"))
    ranks = {s.name: array.array("Q") for s in SOURCES}
    for r in jsonl(OUT / "candidates.jsonl"):
        ranks[r["source"]].append(rank(r["id"]))
    cut = {s.name: (1 << 64) if s.take is None or s.take >= len(ranks[s.name]) else sorted(ranks[s.name])[s.take - 1] for s in SOURCES}
    total = collections.Counter()
    lengths = collections.defaultdict(lambda: (array.array("H"), array.array("H")))
    sha = hashlib.sha256()
    with open(OUT / "sources.jsonl", "wb") as out:
        for r in jsonl(OUT / "candidates.jsonl"):
            if rank(r["id"]) <= cut[r["source"]]:
                size = r.pop("pieces")
                for group in (r["lang"], "all"):
                    lengths[group][0].append(len(r["text"]))
                    lengths[group][1].append(size)
                total.update([("source", r["source"]), ("lang", r["lang"]), ("register", r["register"]), ("licence", r["licence"])])
                line = (json.dumps(r, ensure_ascii=False) + "\n").encode()
                sha.update(line)
                out.write(line)

    def by(kind):
        return {name: n for (k, name), n in sorted(total.items()) if k == kind}

    manifest = {
        "retrieved": RETRIEVED, "rows": sum(by("lang").values()), "sha256": sha.hexdigest(),
        "by_language": by("lang"), "by_register": by("register"), "by_licence": by("licence"),
        "lengths": {group: {"characters_mean": round(sum(chars) / len(chars), 1), "characters_p95": percentile(chars, 0.95),
                            "pieces_mean": round(sum(sizes) / len(sizes), 1), "pieces_p95": percentile(sizes, 0.95)}
                    for group, (chars, sizes) in sorted(lengths.items())},
        "sources": {s.name: {"title": s.title, "repairs": s.repairs, "register": s.register, "licence": s.licence, "licence_url": s.licence_url,
                             "credit": s.credit, "homepage": s.homepage, **counted["sources"][s.name],
                             "target": s.take, "in_mix": total["source", s.name]} for s in SOURCES},
        "files": counted["files"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps({k: manifest[k] for k in ("rows", "by_language", "by_register", "lengths")}, indent=1, ensure_ascii=False))


# sha256 of every download. A file that no longer matches stops the build.
PINS = {
    "mot/tha_voathai.tgz": "69c26f7e4e3d2ed33e193a4e41f021bd3ecfba21a89d1080fcb4960c66a1f4ef",
    "tatoeba/eng_sentences_detailed.tsv": "c9bf156a539da9047818f4dc9544a9da6272f8d3df7ab7bc589622f242a32504",
    "tatoeba/tha_sentences_detailed.tsv": "b7b7a3d2177f0fbf2d95eea3b0cfe8ea519d1b1cecd143d00ec76d95d37636a0",
    "tatoeba/eng_sentences_CC0.tsv.bz2": "c2e2edd063a31cbbc8e416a4858225e934bbb0afff485031821e59f230ccf1ca",
    "sgd/sgd.zip": "19da4aa3159d113a073ae1bce75ae46c290323d73e512df7d58c8e616dc776b9",
    "abcd/abcd_v1.1.json.gz": "2bdf53ac359543dcdc38d55bc6513e78df120363f8f44870716e909f4606de15",
    "byu/dataset.json": "d2a5d0bf6acf354fef51f3af39cbece2220d3b8f6086b658c3d335105798312a",
    "dolly/databricks-dolly-15k.jsonl": "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
    "benchmarks/flores200_dataset.tar.gz": "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6",
    "benchmarks/wmt24pp-en-th_TH.jsonl": "d50f224f9bdb92b02627eba752ff96f512cdf4717b40d36cf5a490a4854c462d",
    "globalvoices/en.zip": "974c2b104717a99662e6756272e2463989e7e78002e1a78e8e0e83c046fa9b3f",
    "blognone/train.parquet": "564ae9a5da15c88fd1941d39f8cccb6ab415ddfa82a95327a1665654a07acb97",
    "iapp/train.parquet": "7b7534700b931731e3d55cc0085f7c8e358b5fca68cc12d1ac5fca993ad27b3a",
    "iapp/validation.parquet": "e68af11dc6261f3dde6ae2463d9ee4d4e5f28993ab3fea4ddf00e1e7f89f3862",
    "iapp/test.parquet": "1b5f43a9f2e62e4932c6f30f4ccacfddd371605d93c6e4029d34e9d10aaf969b",
    "soda/train.parquet": "b21314575fdc4cbf9fc74f7af4854d55fba8cc9ba72f2f3791ae2df8b23032c4",
    "soda/valid.parquet": "3e813e5b14c4edca5849cb7922f9c38568ecb71f27931a7c4c29ec1160bd0d1c",
    "soda/test.parquet": "edd0a57b460fa7faa3782b332327f3a4bc9681dcaaec88e77727c47fd5870180",
    "taskmaster2/flights.json": "86b37b5ae25f530fd18ced78800d30c3b54f7b34bb208ecb51842718f04e760b",
    "taskmaster2/food-ordering.json": "0a042e566a816a5d0abebe6f7e8cfd6abaa89729ffc42f433d327df7342b12f8",
    "taskmaster2/hotels.json": "975b0242f1e37ea1ab94ccedd7e0d6ee5831599d5df1f16143e71110d6c6006a",
    "taskmaster2/movies.json": "6f67c9a1f04abc111186e5bcfbe3050be01d0737fd6422901402715bc1f3dd0d",
    "taskmaster2/music.json": "e5db60d6576fa010bef87a70a8b371d293d48cde8524c1d3ed7c3022f079d95d",
    "taskmaster2/restaurant-search.json": "fb9735f89e7ebc7c877f976da4c30391af6a6277991b597c0755564657ff8f47",
    "taskmaster2/sports.json": "8191531bfa5a8426b1508c396ab9886a19c7c620b443c436ec10d8d4708d0eac",
    "taskmaster3/data_00.json": "4edf97557e1aa7f654bf97994b5eae42653ef6d4f5e50136f7664d7e5cdff7b6",
    "taskmaster3/data_01.json": "a4fef75ec7824bb3fb29b8afe9ed63f3354b81077bb9b6f58f2349bb60d660fc",
    "taskmaster3/data_02.json": "25cb4788c4c857740152612397b0e687dfeade14c7dbfeed501a53290725aedf",
    "taskmaster3/data_03.json": "5e1f15544f259c804438458f34a6c7f00400e30dba7648c37a0b31886c5dd903",
    "taskmaster3/data_04.json": "17d8deb3bc6c3551c564cb9003a47202a835e011477b1e3c81711b083bf4064b",
    "taskmaster3/data_05.json": "12e58018be7b42aa16968ab914d03c56af9fae9fb5feec7448e01f738c50d2fe",
    "taskmaster3/data_06.json": "a4e290dafc6362ca8d4d786d6e70f6339297b0414cca4fe0ec79c1d383d65673",
    "taskmaster3/data_07.json": "becec1c0e9b88781fbb81f5d032826f85d77756cfae7c37acbca9b4b2a470fdb",
    "taskmaster3/data_08.json": "c2718197cedfb1ff7b830492ddbdc36c213ac98c83ba974ff0ea0eeeffdfe9b5",
    "taskmaster3/data_09.json": "a8867eb49010fa735b3c0b2f47df4cc225c79b6ca919009db857f454cccc986d",
    "taskmaster3/data_10.json": "88a8ce81f2c843c7a52a1a6c016d790e6547d5a3ffdd000ca00f0676c02e077b",
    "taskmaster3/data_11.json": "10e5d2dadb7a55b08d41605498b46320fa2df7b87c71a820df4245e9eeeba4d1",
    "taskmaster3/data_12.json": "be255d29439d10fc33bb222a1294da0af7ed4bdf8ede84c389d9b744e9cdf343",
    "taskmaster3/data_13.json": "a477d4668398b86bc7f68f86c20a423e81b4dbc41002a5d605dee0c6cee250b8",
    "taskmaster3/data_14.json": "92c7a0ca1fce8845610a7fdb00972750228351b2c2065e76120c328c806cc1e5",
    "taskmaster3/data_15.json": "e67e9d649d29f3edc12796cb4b1308cc7cb89f676c12e96d8fea558510b6ee18",
    "taskmaster3/data_16.json": "f0d3f22ee8b9224ea4efc57182f1620f849f3b90050cc04992319aab04e7b453",
    "taskmaster3/data_17.json": "15c469a030aae16a98969675b238f368436ec0a5074f3eb79458fdc81cf489d4",
    "taskmaster3/data_18.json": "28dfe3f65c1cd7bab8ed9da0e667ee1aec1561cf41237531f30c889cafb0997d",
    "taskmaster3/data_19.json": "7ae99d1237c25519269c3e6969ab0ecd88509828c436cbecb8a86831007c3868",
    "multiwoz/train_001.json": "895a8109bf01fa5ecf15ccdbd2dfe1628bd923f6b61dcd2e26b10ee5076a1596",
    "multiwoz/train_002.json": "2f3ea771d4e01cb2780357738cff7f7496b87d34c221cc240df74501312438d3",
    "multiwoz/train_003.json": "da24961d28486be2d8462ee4d86a809db819d588ba90ae1a783383d95eb85daa",
    "multiwoz/train_004.json": "30c1172db1071c853b16215d1946de908d68d2b6ff8da7801de307000a179106",
    "multiwoz/train_005.json": "eaf58716df5de99524b3e0e7edf477b74749512788a6a51f53f2bdd76768d39a",
    "multiwoz/train_006.json": "8e75fd543b1964bc5e7118085d977f479c98fcdf6d606b971f67a45fb1745c83",
    "multiwoz/train_007.json": "02323f8298439d713c6d7d226f4bd7ec246ec993ee11911b54f98cb8a598f206",
    "multiwoz/train_008.json": "1129fbed480352ae304f0ae5b4915c194e9619c43f1577ccb0d450e10d24ea98",
    "multiwoz/train_009.json": "87d9e43b1ba51a4a4688703da79d3a09b14d8013570545da24c597daa18e2f45",
    "multiwoz/train_010.json": "e7ad0d5da2909b08197295e45fe4695b9dc2f67d458374b2aab8db5094e97b26",
    "multiwoz/train_011.json": "82e2d2900a037b866a01d05974734dd419e89b329fe29ef93b35eea96d27feb8",
    "multiwoz/train_012.json": "b6bf292325db67682dd7b6fafbf1051cc2262e92f7c37cab213e975888594bb2",
    "multiwoz/train_013.json": "c33fe4b3952c016e1e1645f472a7097f93dfb476a19940fd9386865ef9adf685",
    "multiwoz/train_014.json": "ce33dbbf93a40d0dcc671a9d6a2ed1f987914f5b5f05f6df493a5b342efda954",
    "multiwoz/train_015.json": "d895c0439bc2ad89ef1689896e3be630eadebc33dae77b42a426fd16a271718e",
    "multiwoz/train_016.json": "3e6bc0bca4262022ccbce0d5ce3150e536e7d21aeb9fbdef9e83cced4dfd124b",
    "multiwoz/train_017.json": "b6ab2cd9b6e8983364526845b7cbec1749338209bf7ac2313c25e1e2226ebab5",
    "multiwoz/dev_001.json": "e7ddb563e4da5766ea820cc826dead77e7ca219c19b761e218d62d9c999a252e",
    "multiwoz/dev_002.json": "ede6a2c17fd6c5846214b8cabc1ef8f7cc8be01cfbacaa162bcafec9e87724e9",
    "multiwoz/test_001.json": "d6f43876cf130fdb2dfa8f96bc056b0995354137f02122e004925d01264ed386",
    "multiwoz/test_002.json": "89af95d8f596a448e733d59b31be78f1dd1632eddd99d5cb298a3fcb1ac9d185",
    "commonvoice/en-wiki.en.txt": "438e55e8bc91dfa6cbe098a7233702c6b177c5ae07137a01d66b9c75c9789ae5",
    "commonvoice/en-wiki.en.rerun-2021-11.txt": "16647e64ba4124ccfeb7061e88ffb53261a16a15b7c4abcd2122b168b9f29639",
    "simplewiki/simplewiki-20261101-pages-articles.xml.bz2": "d21f674336fecc84885b2e11d0d0e10ba5cdacafdaabb82d033ebfd465433c8e",
    "thwiki/thwiki-20261101-pages-articles.xml.bz2": "7d92554ec882cfe96e2d142a2eb70b027a0ff5c15def8425af868ef9f149c512",
    "enwikivoyage/enwikivoyage-20261101-pages-articles.xml.bz2": "df5aac6e6713cbdde73e805e9d10976a46850afc921b05d3e2cc022c9c8e1228",
    "enwikinews/enwikinews-20261101-pages-articles1.xml-p1p1500000.bz2": "eeb839f1009ef9ff903779cb876423ef66cb14159661737083386fa3414cd9d5",
    "enwikinews/enwikinews-20261101-pages-articles1.xml-p1500001p3000000.bz2": "fe633aaaedc8ec8768e083b04fe596e9a1c5d2b2030f5bf9a422f0a607e6b19a",
    "enwikinews/enwikinews-20261101-pages-articles1.xml-p3000001p3095843.bz2": "a4fc912a65f6ef035e437b7d080e1e421a9fe88f945a1d852b775bbb6d69b83d",
}

if __name__ == "__main__":
    command = sys.argv[1:] or ["candidates", "near", "mix"]  # mix alone re-balances without reading the downloads again
    for step in command:
        {"fetch": fetch, "candidates": candidates, "near": near, "mix": mix}[step]()
