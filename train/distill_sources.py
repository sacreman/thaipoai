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
import io
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
PREVIOUS_CAP = 2  # most task-dialogue rows that may answer the very same previous line
FRAME_CAP = 5  # most rows a task dialogue may give for one frame, once names, dates, numbers and slot values are masked
MAX_CONTEXT = 200  # characters of the previous turn carried with a dialogue row, and twice that of the sentence before a prose row
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
LOTUSDIS = f"{HF}/nectec/LOTUSDIS/resolve/45145d7ca2286498a31958b35628ab3c1969f808/data"
LOTUSDIS_PARTS = {"train": 17, "validation": 3, "test": 4}
YODAS = f"{HF}/espnet/yodas/resolve/0cad6a0ebec75e945fef92084052be69457fe9d1/data/th000/text"
BY_SA = "https://creativecommons.org/licenses/by-sa/4.0/"
BY = "https://creativecommons.org/licenses/by/4.0/"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
MIT = "https://opensource.org/license/mit"
APACHE = "https://www.apache.org/licenses/LICENSE-2.0"

URL = re.compile(r"https?://|www\.|\w\.(?:com|org|net)\b", re.I)
MARKUP = re.compile(r"[{}\[\]<>|\\=_#*~^`]|\w\(|\);|::|:[a-z]{2}:|&\w+;")
# Emails, handles, phone and account numbers, booking or order codes such as 7GAWK763, and the invented US
# addresses, UK postcodes and train numbers of the task dialogues.
CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.\w|(?<!\w)@\w{2,}|\d{7,}|\+?\d(?:[ .-]?\d){8,}|\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{5,}\b|, [A-Z]{2},? \d{5}\b|\b[a-z]{2}\d{1,2}\s?\d[a-z]{2}\b|\bTR\d{4}\b|\(\d{3}\)\s?\d{3}[-.\s]?\d{4}|\b\d{1,5} [A-Za-z0-9][A-Za-z0-9 ]{0,30}?(?i: ave| avenue| rd| road| st| street| dr| drive| lane| ln| blvd| boulevard| way| court| ct| highway| hwy| place| parkway| square| terrace)\b|\b[A-Z]{8}\b|\b(?:[a-z]+\d{2,}|\d{2,}[a-z]{3,})\b")
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
THAI_DANGLING = re.compile("(?:(?<!รูป)(?<!ต้น)แบบ|แบบว่า|(?<!ก)ว่า|ก็|แล้วก็|คือ|เช่น|และ|หรือ|เพื่อ|เพราะว่า|แต่ว่า|(?<!หน้า)(?<!สถาน)(?<!พื้น)(?<!เนื้อ)ที่|ที่จะ|(?<!สิ่ง)(?<!ข้าว)ของ|เป็น|โดย|(?<!งบ)ประมาณ|อาทิ|ดังนี้|ดังต่อไปนี้|ได้แก่|ประกอบด้วย|ประกอบไปด้วย|ดังนั้น|สมการ|หลังจาก|เพราะ|เนื่องจาก|ซึ่ง|\\.\\.|:)$")
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
# A task-dialogue line that gives or asks for the invented customer's details.
PERSONA = re.compile(r"(?i)\b(?:my (?:full )?name\b|name is\b|zip code|postal code|promo code|account id|username|user name|email address|phone number|order id)")
NAME_ONLY = re.compile(r"(?i:(?:it'?s|it is|my name is|i am|i'm|this is|name is|sure,?|yes,?|ok,?|okay,?)\s+)*[A-Z][a-z]+ [A-Z][a-z]+[.!]?$")
# Leading words that tie a sentence to the one before without changing what it says. They are removed, and the
# sentence stands alone. (Words that carry the logic, such as "therefore", make the row lean, and it is dropped.)
CONNECTIVE = re.compile(r"(?:However|Also|Additionally|In addition|Furthermore|Moreover|Nevertheless|Nonetheless|Meanwhile|Then|First|Firstly|Second|Secondly|Third|Finally|Next|And|But|So|Yet|Still|Later|"
                        r"At the same time|In fact|For example|For instance|Indeed|Besides|Now|Again|Likewise|Similarly|Overall|Ultimately|In general)\b,?\s+(?=\S)")
THAI_CONNECTIVE = re.compile(f"(?:นอกจากนี้|นอกจากนั้น|อย่างไรก็ตาม|อย่างไรก็ดี|ทั้งนี้|ต่อมา|ขณะเดียวกัน|ในขณะเดียวกัน|{THAI_AS_FOR})[ ,]*")
THAI_LOGIC = re.compile("(?:ด้วยเหตุผล|ดังนั้น|จากนั้น|หลังจากนั้น|ด้วยเหตุนี้|ทว่า|เรื่องนี้|ข่าวนี้|ต่อไปนี้|ด้าน|เหตุผลเป็นเพราะ|ที่สำคัญ|เรียกได้ว่า|นับเป็น|สิ่งที่ตามมา|อีกบริเวณ|อีกสองคน|คำถาม|ติดตามข่าว|บทความ(?:อ่านประกอบ|ที่เกี่ยวข้อง)|จากที่|ตามที่)")  # openers that cannot be removed without losing the point
THAI_BOILERPLATE = re.compile(r"เป็น \d+$|ตาราง(?:ด้านล่าง|ต่อไปนี้|ข้างล่าง|นี้)|ด้านล่าง|ข้างล่าง|ดังต่อไปนี้|ต่อไปนี้|ดังตาราง|^ด้วยการที่|ได้ที่นี่|ลุ้นรับ|ข้างต้น|(?:ของ|จาก|โดย|แห่ง|ออก|อย่าง) (?:ว่า|ที่|และ|ได้|ก็)|อ่านรายละเอียด|\(รายละเอียด\)|ตามภาพ|อาจหมายถึง$|\(\d+\)$|โดย \(")
CLOSING = re.compile(r"(?i)\bthank|\bthat (?:will|would) be all\b|\bthat's all\b|\bbye\b|\bgoodbye\b|\bproceed further\b")
STAMMER = re.compile(r"(?i)\b(\w+) \1\b|\bi mean\b|\b(?:uh|um|er)\b")  # the marks of a speech recogniser's transcript
BYU_INJECTED = re.compile(r"The \w+ \w+ \w+(?:ed|s) \w+ly\.$")
SPOKEN_TAIL = re.compile("(?: ?(?:แล้วก็|ก็|แต่ว่า|แต่|และ|เพราะว่า|เพราะ|ก็คือ|คือ|แบบว่า|(?<!รูป)(?<!ต้น)แบบ|(?<!ก)ว่า|เพื่อ|ซึ่ง|โดย|อ่า|เอ่อ))+$")  # what a speaker says while thinking of the next thing
SPOKEN_LEANS = re.compile(f"(?!แต่|ก็|แล้ว|เพราะ)(?:{THAI_LEANS})|ให้|เลย")  # speech may open with "but", "so", "then", "because", and with nothing else that joins
# What a name or formula leaves when the extractor or the publisher lost it: "นักวิจัยที่ในพรินซ์ตัน", "รายการ ของสถานี".
# Also editors' notes to each other, and a paragraph that points at numbered items of a list that is not there.
THAI_HOLE = re.compile("ที่ใน(?!อดีต|ปัจจุบัน|ขณะ|ปี|ช่วง|เวลา|ตอน|ภายหลัง|ที่สุด|วัน|สมัย|ยุค|ระหว่าง|บาง|หลาย|ทาง|ตัว|ส่วน|เดือน|ครั้ง)|จาก ใน|โดยมีเป็น|โดยที่ คือ"
                       r"|(?:รายการ|วารสาร|หนังสือพิมพ์|นิตยสาร|ชื่อ|ชื่อว่า|เรียกว่า) (?:ของ|และ|ที่(?!อยู่)|ว่า|ซึ่ง)|ลงในบทความ|ควรตรวจสอบ|\(\d{1,2}\)")
THAI_NEEDS_SUBJECT = re.compile("ได้รับ|ทรง|ยัง|จึง|พระนาง|พระองค์|เขา(?!ต)|เธอ")  # with no sentence before, nobody knows who
# Words that begin an English sentence without being anybody's name.
COMMON_STARTS = frozenset("""i the a an he she it we they you this that these those there here what who where when why how which do does did is are was
were can could will would should may might must have has had my your his her our their its if in on at for to from with as and but or so not no yes let please all some
many most every each both one two three after before since because although while even just only never always sometimes nobody everyone everybody someone somebody
something nothing everything anything people mr mrs dr now then today yesterday tomorrow maybe perhaps of by about during without until once any another such
nobody neither either few several much more less than too very really also still again""".split())
BRACKETED = re.compile(r"\([^()]*\)")
# Notes, timeline entries ("พ.ศ. 2546 จัดทำป้อมยามใหม่") and labelled lines ("ภาษาอังกฤษ : ...") are list items, not prose.
THAI_NOTE = re.compile(r"หมายเหตุ|(?:พ\.ศ\.|ค\.ศ\.) ?\d{4}|.{0,40} : ")
# A row cut straight after a reporting verb or a "but": "...ระบุว่า รัฐบาลอเมริกัน".
REPORTED = re.compile("(?:(?<!ก)ว่า|(?<!ตั้ง)แต่).{0,19}$")
# Stock openings of Simple English Wikipedia that survive STUB: the biography lead keeps one in four, footballer lines one in two.
BIOGRAPHY_LEAD = re.compile(r"[^()]{3,60} \([^)]*\d{3,4}[^)]*\) (?:is|was) an? ")
FOOTBALLER = re.compile(r"(?i)\b(?:footballer|football player|soccer player)\b|\b(?:plays|played) (?:for|as a)\b|\bnational (?:football )?team\b")
# The line before asks who the customer is: what follows is an invented identity.
ASKS_IDENTITY = re.compile(r"(?i)\b(?:name|pin|zip|phone|e-?mail|username|order id|account id|address)\b")
MONTHS = re.compile(r"(?i)\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a couple of|january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|tonight|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec|mon|tues?|wed|thur?s?|fri|sat|sun|am|pm|a\.m\.|p\.m\.)\b")
THAI_MIN = 30  # characters a Thai sentence needs before it may end


# ---------------------------------------------------------------- text to sentences

def thai_rows(prefix, paragraphs):
    """(id, sentence, "th", sentence before it) for the paragraphs of one Thai article. The sentence before,
    which may close the paragraph before, is carried as context, because Thai leaves out a subject that the
    sentence before has just named."""
    n, previous = 0, None
    for paragraph in paragraphs:
        for sentence in thai_sentences(clean(paragraph)):
            if HOLE in sentence:  # something was removed from it: not a row, and not context for the next
                sentence = None
            else:
                yield f"{prefix}-{n}", sentence, "th", previous
            previous, n = sentence, n + 1


def whole_paragraphs(rows):
    """Only the pieces too long for an ordinary Thai row: paragraphs the splitter found no certain cut in.
    They go in whole, under a source name of their own, with a longer limit."""
    return (r for r in rows if len(clean(r[1])) > MAX_CHARS["th"])


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
    therefore one sentence or several whole ones, never part of one.
    Merging comes before dropping: a sentence that would be dropped for leaning on the one before it (its
    opener carries the logic, or it has left its subject behind) or for being very short is joined to that
    sentence again when the two fit in one row."""
    cut, current = [], ""
    for chunk in text.split():
        closed = current.count("(") <= current.count(")") and current.count("“") <= current.count("”") and current.count('"') % 2 == 0
        waiting = THAI_IF.match(current.rsplit(" ", 1)[-1])  # "นอกจาก...แล้ว ปัจจุบัน...": the clause before is not a sentence yet
        if len(current) >= THAI_MIN and closed and not waiting and THAI_OPENER.match(chunk):
            cut.append(current)
            current = chunk
        else:
            current = f"{current} {chunk}" if current else chunk
    out = []
    for piece in cut + [current] * bool(current):
        leans = THAI_LOGIC.match(piece) or THAI_SUBJECTLESS.match(THAI_OPENER.sub("", piece, count=1).lstrip(" ,")) or len(piece) < 40
        if out and leans and len(out[-1]) + len(piece) < MAX_CHARS["th"]:
            out[-1] += " " + piece
        else:
            out.append(piece)
    return out


def sentences(text, lang, limit=PER_MESSAGE):
    """Sentences of a question or answer. Messages holding code or lists give nothing."""
    if "```" in text:
        return []
    lines = [line for line in text.split("\n") if line.strip() and not re.match(r"\s*(?:[-*#>|]|\d+[.)])", line)]
    split = thai_sentences if lang == "th" else english_sentences
    return [s for line in lines for s in split(clean(line))][:limit]


WIKI_DROP = re.compile(r"(?is)<!--.*?-->|<ref\b[^>]*/>|<(ref|math|gallery|syntaxhighlight|source|code|pre|nowiki|timeline|imagemap|poem|score)\b.*?</\1\s*>")
WIKI_NOTE = re.compile(r"(?is)<!--.*?-->|<ref\b[^>]*/>|<ref\b.*?</ref\s*>")  # stands beside the sentence, not in it
WIKI_INNERMOST = re.compile(r"\{\{[^{}]*\}\}|\{\|[^{}]*\|\}")
# Templates that carry words of the sentence: {{w|Paris}} on Wikinews, {{lang|fr|mot}}, {{nowrap|10 km}}. Their last part is kept.
WIKI_WORDS = re.compile(r"(?i)\{\{\s*(?:w|lang(?:x|-\w+)?|nowrap|nobr|small|em|transl|transliteration|linktext|wikt|l|ไม่ตัด|อังกฤษ|ภาษาอังกฤษ|ชื่อวิทยาศาสตร์)\s*\|(?:[^{}|]*\|)*([^{}|]*)\}\}")
WIKI_FIRST = re.compile(r"(?i)\{\{\s*(?:abbr|ill|interlanguage link|nihongo\d?)\s*\|([^{}|]*)[^{}]*\}\}")  # these show their first part
WIKI_CONVERT = re.compile(r"(?i)\{\{\s*(?:convert|cvt)\s*\|\s*([\d.,]+)\s*\|\s*([^{}|]+)[^{}]*\}\}")
# Templates that show nothing in the sentence, or only a note or a pronunciation in brackets.
WIKI_SILENT = re.compile(r"(?i)\{\{\s*(?:sfn\w*|efn[\w-]*|rp|r|refn|harv\w*|#tag:ref|อ้างอิง|citation needed|cn|fact|ต้องการอ้างอิง[^{}|]*|webarchive|ลิงก์เสีย|page needed|reference page|"
                         r"clear|nbsp|coord|flagicon|ญี่ปุ่น|zh[\w-]*|korean|เกาหลี|ipa[\w-]*|respell|audio)\s*(?:\|[^{}]*)?\}\}")
WIKI_FILE = re.compile(r"(?i)\[\[(?:file|image|category|ไฟล์|ภาพ|หมวดหมู่):[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]")
WIKI_LINK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]")
WIKI_EXTERNAL = re.compile(r"\[(?:https?:)?//\S+ ?([^\]]*)\]")
WIKI_TAGS = re.compile(r"<[^>]*>|'{2,}|__[A-Z]+__")
WIKI_HOLES = re.compile(r"(?<=\()[\s,;]+|[\s,;]+(?=\))| ?\(\)")  # what a removed pronunciation template leaves in brackets
HOLE = "⟨⟩"  # stands where a formula or a template with no plain text was, so the sentence around it can be dropped
LIST_LEAD = re.compile("(?:ได้แก่|ดังนี้|ดังต่อไปนี้|คือ|ประกอบด้วย|ประกอบไปด้วย|ดังนั้น|[:：])$")
# Articles Thai Wikipedia's own editors have marked as needing clean-up, copy-editing or translation.
WIKI_MAINTENANCE = re.compile(r"(?i)\{\{\s*(?:เก็บกวาด|ปรับภาษา|แก้ภาษา|ยังไม่ได้แปล|ไม่เป็นสารานุกรม|โฆษณา|โปรโมต|ต้องการตรวจสอบการแปล|แปลไม่ดี|(?:multiple )?issues|cleanup[\w -]*|copy ?edit|rough translation|tone|advert)\s*[|}]")


def wiki_paragraphs(text, careful=False):
    """Prose paragraphs of a wiki page: templates, tables, files, references, headings and list items go. A
    single line break inside a paragraph is only how the page was typed, so such lines are joined again.

    careful is for Thai, which has no full stop to show that something is missing. A reference or a line
    break becomes a space, so two sentences do not fuse. A formula, or a template that leaves no plain text,
    leaves HOLE in the sentence. A paragraph that leads into a list ("... ได้แก่") goes, with what follows
    it up to the next heading, and so does a glossary entry (the lines under a ";" line)."""
    if careful:
        text = WIKI_DROP.sub(f" {HOLE} ", WIKI_NOTE.sub(" ", text))
    else:
        text = WIKI_DROP.sub("", text)
    for _ in range(8):
        text = WIKI_CONVERT.sub(r"\1 \2", WIKI_FIRST.sub(r"\1", WIKI_WORDS.sub(r"\1", text)))
        text, changed = WIKI_INNERMOST.subn(f" {HOLE} " if careful else "", WIKI_SILENT.sub(" ", text) if careful else text)
        if not changed:
            break
    text = WIKI_TAGS.sub("", WIKI_EXTERNAL.sub(r"\1", WIKI_LINK.sub(r"\1", WIKI_FILE.sub("", text))))
    text = WIKI_HOLES.sub("", WIKI_HOLES.sub("", text))
    paragraphs, current, skip = [], "", ""
    for line in text.split("\n") + [""]:
        line = line.strip()
        while careful and line.startswith(HOLE):  # a template on a line of its own, or before the sentence starts
            line = line[len(HOLE):].lstrip()
        if not line or line[0] in "*#:;|!={}":
            if current and careful and LIST_LEAD.search(current):
                skip = "list"
            elif current:
                paragraphs.append(current)
            current = ""
            skip = "" if line[:1] == "=" or (not line and skip == "entry") else "entry" if careful and line[:1] == ";" else skip
        elif not skip:  # Thai is written without spaces, so a break between two Thai letters joins with nothing
            current += ("" if not careful and current and THAI.match(current[-1]) and THAI.match(line[0]) else " ") * bool(current) + line
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


def slot_values(turn, text):
    """What the dataset marks as a slot value in or beside this turn: a city, a time, a film."""
    for f in turn.get("frames", []):
        for slot in f.get("slots", []):
            if "start" in slot:
                yield text[slot["start"]:slot["exclusive_end"]]
            yield from (v for v in ([slot["value"]] if isinstance(slot.get("value"), str) else slot.get("value") or []))
        for values in f.get("state", {}).get("slot_values", {}).values():
            yield from values
    for segment in turn.get("segments", []):
        yield segment.get("text") or text[segment["start_index"]:segment["end_index"]]


def without_values(shape, values):
    """shape with every run of up to six words that is one of the dataset's slot values masked."""
    words, out, i = shape.split(), [], 0
    while i < len(words):
        n = next((n for n in range(min(6, len(words) - i), 0, -1) if " ".join(words[i:i + n]).strip(".,!?").casefold() in values), 0)
        out.append("Xx" if n else words[i])
        i += n or 1
    return " ".join(out)


def turns(name, files, list_key="turns", text_key="utterance"):
    """Task dialogues: a JSON list of dialogues, each a list of turns. files() gives (label, open file) pairs.
    Only what the user said is taken, with the turn before it as context: the system's side is written from
    templates. The fifth value is the turn with its slot values masked, for the frame cap: the values the
    dataset marks in this turn, and any value it marks anywhere, so "to Boston" is masked in a turn nobody
    annotated. The files are read twice, once to collect those values."""
    values = {v.casefold() for _, handle in files() for dialogue in json.load(handle) for turn in dialogue[list_key]
              for v in slot_values(turn, turn[text_key]) if len(v) >= 3}
    for label, handle in files():
        for d, dialogue in enumerate(json.load(handle)):
            previous = None
            for t, turn in enumerate(dialogue[list_key]):
                text = turn[text_key]
                if turn["speaker"].upper() == "USER":
                    spans = [(s["start"], s["exclusive_end"]) for f in turn.get("frames", []) for s in f.get("slots", []) if "start" in s]
                    spans += [(s["start_index"], s["end_index"]) for s in turn.get("segments", [])]
                    shape = text
                    for start, end in sorted(set(spans), reverse=True):
                        shape = shape[:start] + "Xx" + shape[end:]
                    yield f"{name}-{label}-{d}-{t}", text, "en", previous, without_values(shape, values)
                previous = text


def sgd(path="sgd/sgd.zip"):
    with zipfile.ZipFile(RAW / path) as archive:
        names = sorted(n for n in archive.namelist() if re.search(r"/(train|dev|test)/dialogues_\d+\.json$", n))
        yield from turns("sgd", lambda: ((re.sub(r".*/(\w+)/dialogues_(\d+)\.json", r"\1\2", n), archive.open(n)) for n in names))


def plain_files(name, pattern, **keys):
    paths = sorted((RAW / name).glob(pattern))
    yield from turns(name, lambda: ((p.stem, open(p, encoding="utf8")) for p in paths), **keys)


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


LETTER_SPACED = re.compile(r"\b(?:\w ){3,}\w\b")  # "h e l l o"


# A message that carries on the sentence the same person's last message left open: "...you get alot more" / "out of it".
CARRIES_ON = re.compile(r"(?:and|but|that|out|from|to|of|or|so|than|which|because)\b")
LEFT_OPEN = re.compile(r"(?i)(?:,|\b(?:but|and|more|the|a|to|like))$")
# A reply with nothing in it but pronouns and auxiliaries ("you do huh?") says something only as an answer.
EMPTY_WORDS = frozenset("""i you he she it we they me him her us them this that these those do does did is are am was were be been have has had
can could will would shall should may might must not no so too huh oh""".split())
CONTRACTIONS = re.compile(r"n't\b|'(?:s|re|m|ll|d|ve)\b|’(?:s|re|m|ll|d|ve|t)\b")


def byu(path="byu/dataset.json", counts=None):
    """Informal chat, one message a row, with the message before it in the conversation as context. The
    source lists each conversation's messages in the order they were sent, grouped into one sender's runs.

    People split one sentence over two messages. So a message is joined to the same sender's message just
    before it when that one was left open (it ends in a comma or a dangling word) or has no closing mark and
    this one opens with a lowercase joining word, as long as the two fit a row. A reply of five words or
    fewer made only of pronouns and auxiliaries is kept only after a question from the other person. Messages
    spelled out a letter at a time are left out, and so are the test lines the organisers planted.
    counts, if given, is a Counter that receives how many messages were joined and dropped."""
    counts = collections.Counter() if counts is None else counts
    with raw(path) as handle:
        for conversation, body in json.load(handle).items():
            previous, asked = None, False  # the message before, and whether it was the other person's question
            for t, turn in enumerate(body["messages"]):
                rows = []  # this sender's run, as [id, text]
                for k, message in enumerate(turn):
                    text = message["text"].strip()
                    if BYU_INJECTED.match(text) or LETTER_SPACED.search(text) or not text:
                        continue
                    last = rows[-1][1] if rows else ""
                    if last and len(last) + len(text) < MAX_CHARS["en"] and (LEFT_OPEN.search(last) or (last[-1] not in ".!?…" and CARRIES_ON.match(text))):
                        rows[-1][1] = f"{last} {text}"
                        counts["joined"] += 1
                    else:
                        rows.append([f"byu-{conversation}-{t}-{k}", text])
                for row_id, text in rows:
                    words = WORD.findall(CONTRACTIONS.sub(" ", text.casefold()))
                    if len(words) <= 5 and set(words) <= EMPTY_WORDS and not asked:
                        counts["empty reply"] += 1
                    else:
                        yield row_id, text, "en", previous
                    previous, asked = text, False  # the next row's context is this message, whole, kept or not
                asked = bool(rows) and rows[-1][1].endswith("?")


# Stage directions the model wrote into a turn, and short turns that only greet or thank.
SODA_SKIP = re.compile(r"\([^)]*\)|\*[^*]+\*|^(?:Hi|Hey|Thank you|Thanks|You're welcome)\b(?:\W+\w+){0,7}\W*$")


def soda(every=4):
    """Dialogue turns only, not the narrative, each with the turn before it as context. One dialogue in `every`:
    the whole set is 11 million turns."""
    import pyarrow.parquet as pq

    for split in ("train", "valid", "test"):
        for batch in pq.ParquetFile(RAW / f"soda/{split}.parquet").iter_batches(columns=["dialogue", "original_index"]):
            for dialogue, index in zip(batch["dialogue"].to_pylist(), batch["original_index"].to_pylist()):
                if index % every == 0:
                    for t, text in enumerate(dialogue):
                        if not SODA_SKIP.search(text):
                            yield f"soda-{index}-{t}", text, "en", dialogue[t - 1] if t else None


REFERS = re.compile(r"(?i)\b(?:reference text|paragraph|passage|the text|below|above|following|given|summary)\b")


def dolly(path="dolly/databricks-dolly-15k.jsonl"):
    """The questions people wrote, unless they ask about a text that is not here: a record with a context
    field (a Wikipedia excerpt) gives nothing. The answers are left out, and so is a question typed with no capital."""
    with raw(path) as lines:
        for i, line in enumerate(lines):
            record = json.loads(line)
            for n, sentence in enumerate(() if record["context"].strip() else sentences(record["instruction"], "en")):
                if sentence.endswith("?") and not sentence[0].islower() and not REFERS.search(sentence):
                    yield f"dolly-{i}-{n}", sentence, "en"


def oasst2(lang, path="oasst2/ready.messages.jsonl.gz"):
    """What people typed to the assistant in one language, each message whole: reviewed, not deleted, one line."""
    with raw(path) as lines:
        for line in lines:
            m = json.loads(line)
            if m["lang"] == lang and m["role"] == "prompter" and not m["deleted"] and m["review_result"] and not m["synthetic"] and "\n" not in m["text"].strip():
                yield f"oasst2-{m['message_id']}", m["text"], lang


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


def common_voice_th():
    """The two small files of sentences Thai speakers wrote for Common Voice. The large third file,
    sentence-collector.txt, is not read: about a third of it was translated from English by machine, it is
    sorted so the batches cannot be told apart, and no rule separated them well enough."""
    return itertools.chain.from_iterable(lines_of(f"commonvoice_th-{name[:2]}", f"commonvoice/th-{name}", "th") for name in ("Wannaphong.txt", "thsentences.txt"))


# Words that close what a speaker is saying. Topic markers (เนี่ย, อะ, อย่างนี้) are not among them: speakers pause on those mid-thought.
TURN_CLOSE = re.compile("(?:(?<!เนี่ย)(?<!เนี้ย)(?<!อะ)(?<!นะ)(?:ครับ|ครับผม|ค่ะ|คะ)|เลย|ไหม|มั้ย|ด้วย|เนอะ|เนาะ|หรอ|เหรอ)$")


def lotusdis(path="lotusdis/transcripts.jsonl"):
    """Transcripts of three-person Thai conversation, which arrive as short pieces. What one person says before
    another speaks is one turn. Its pieces are joined, and a row ends where a piece closes on a final particle.
    A run that passes SPOKEN_ROW characters with no such close is thrown away up to the next one, and so is
    what is left when the turn ends. Overlapped speech ends a turn and is left out, as are speakers reading out their code. Each turn
    repeats once per microphone, which the de-duplication removes."""
    with raw(path) as lines:
        rows = (json.loads(line) for line in lines)
        for n, (who, turn) in enumerate(itertools.groupby(rows, key=lambda r: (r["split"], r["mic"], r["speaker_id"]))):
            current, lost = "", "&" in who[2]
            for k, piece in enumerate([clean(r["sentence"]) for r in turn] + [None]):
                if piece and not re.search(r"\b[MF]\b", piece):
                    current = f"{current} {piece}".strip()
                    lost = lost or len(current) > SPOKEN_ROW
                if piece is None or TURN_CLOSE.search(piece):
                    if piece and not lost and len(current) >= 20:  # a turn that ends with no closing word was cut off by the next speaker
                        yield f"lotusdis-{n}-{k}", current, "th"
                    current, lost = "", "&" in who[2]


GAMBLING = re.compile(r"สล็อต|บาคาร่า|คาสิโน|แทงบอล|เว็บตรง|ฟรีสปิน|เครดิตฟรี|หวยออนไลน์|(?i:slot|casino|baccarat)")
CAPTION_NOISE = re.compile(r"\[[^\]]*\]|\([^)]{0,30}\)|[♪♫]")  # [music], a (speaker) or (sound) tag, notes
SHOUTED = 0.02  # a video with more than this share of lines ending in "!" is fan-translated subtitles, not Thai speech
# A polite particle closes an utterance, except after นะ, which narrators say in the middle of one.
SPOKEN_END = re.compile("(?<!นะ)(?<!น่ะ)(?:ครับ|ครับผม|ค่ะ|คะ|จ้ะ|จ้า)$")
PAUSE = 200  # hundredths of a second of silence that end an utterance
CAPTION_ROW = 100  # characters a joined caption utterance may reach
SPOKEN_ROW = 150  # characters a joined run of conversation may reach before it is given up


def utterances(captions):
    """Join one video's caption lines, (start, end, text) in order, into whole utterances. A caption line is
    a screenful, not a sentence. An utterance ends at a line that closes with a polite particle. Lines cut
    off by a pause of two seconds or by a "-" that marks another speaker are thrown away, so no half-utterance
    is kept, and so is an utterance of more than CAPTION_ROW characters: long ones are mostly several
    speakers or thoughts run together. (Utterances ending in punctuation or a looser closing word were tried
    and were whole less often. On fresh labelled samples the rule as it stands gave 88 whole utterances in
    100, against 81 without the two limits.) Yields (index of the first line, text)."""
    first, parts = 0, []
    for i, (start, end, text) in enumerate(captions):
        text = clean(CAPTION_NOISE.sub(" ", text))
        if text.startswith("-"):
            parts = []
        text = text.lstrip("- ")
        if not parts:
            first = i
        if text:
            parts.append(text)
        closed = bool(parts) and bool(SPOKEN_END.search(parts[-1]))
        if closed and len(" ".join(parts)) <= CAPTION_ROW:
            yield first, " ".join(parts)
        if closed or (parts and parts[-1][-1] in ".?!") or i + 1 == len(captions) or captions[i + 1][0] - end >= PAUSE:
            parts = []


def yodas(pattern="yodas/*.txt"):
    """Captions that YouTube uploaders supplied for videos they released under a Creative Commons licence.
    Text only. Videos that are not mostly Thai, that advertise gambling, or whose lines often end in "!"
    (subtitles translated from another language) are left out whole."""
    videos = {}
    for path in sorted(RAW.glob(pattern)):
        with open(path, encoding="utf8") as lines:
            for line in lines:
                name, _, text = line.rstrip("\n").partition(" ")
                start, end = name[12:].split("-")[1:3]
                videos.setdefault(name[:11], []).append((int(start), int(end), text))
    for video, captions in videos.items():
        solid = "".join(t for _, _, t in captions).replace(" ", "")
        shouted = sum(t.rstrip().endswith("!") for _, _, t in captions) / len(captions)
        if len(THAI.findall(solid)) >= 0.7 * len(solid) and not GAMBLING.search(solid) and shouted <= SHOUTED:
            for first, text in utterances(captions):
                yield f"yodas-{video}-{first}", text, "th"


SECTION = re.compile(r"มาตรา [\d๐-๙]+(?:/[\d๐-๙]+)?(?: (?:ทวิ|ตรี|จัตวา))?\s+")


LAW_BREAK = re.compile(r"หมวด|ส่วนที่|ลักษณะ|ภาค|บทเฉพาะกาล|หมายเหตุ|ผู้รับสนอง|อัตรา|บัญชี|\W|[\d๐-๙]")  # a heading, the signature, the notes and schedules


def thai_law(pattern="thailaw/*.parquet"):
    """Paragraphs of the sections of Thai Acts and decrees: from "มาตรา 5" to the next heading, each paragraph
    with the lines it was typed on joined again. Sections that hold a numbered list, or only amend or name another law, are left out."""
    import pyarrow.parquet as pq

    for path in sorted(RAW.glob(pattern)):
        for i, text in zip(*(pq.read_table(path, columns=[c])[c].to_pylist() for c in ("__index_level_0__", "text"))):
            inside = False
            for n, block in enumerate(re.split(r"\n\s*\n", text)):
                block = " ".join(block.split())
                inside = bool(SECTION.match(block)) or (inside and not LAW_BREAK.match(block))
                if inside and not re.search(r"\([\d๐-๙ก-ฮ]{1,2}\)|ให้ยกเลิก|ให้เพิ่ม|เรียกว่า|ให้ใช้บังคับ|รักษาการตาม|[“\"]", block):
                    yield f"thailaw-{i}-{n}", SECTION.sub("", block, count=1), "th"


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
    for path in sorted((RAW / name).glob("*.bz2")):
        with bz2.open(path, "rb") as handle:
            for page_id, text in wiki_pages(handle):
                if lang == "th":
                    if not WIKI_MAINTENANCE.search(text):
                        yield from thai_rows(f"{name}-{page_id}", wiki_paragraphs(text, careful=True))
                elif must_contain in text.lower():
                    found = (s for p in wiki_paragraphs(text) for s in english_sentences(clean(p)))
                    for n, sentence in enumerate(itertools.islice(found, per_page)):
                        yield f"{name}-{page_id}-{n}", sentence, lang


# Captions and credits, a link's language tag such as (RUS), and openers that point at a picture or a link.
GV_ASIDE = re.compile(r"^(?:[A-Z]{1,4}:|(?:Image|Photo|Photograph|Picture|Screenshot|Cartoon|Video)\b|In it |They are |Check out )"
                      r"|(?i:\b(?:image|photo|picture|photograph)s? (?:by|from|via|courtesy)\b|\bused under\b|\bCC[ -]BY\b|\boriginal post\b)"
                      r"|\((?:[A-Z]{2,3})\)|\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?$")


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
                if len(text.split()) >= 12 and not GV_ASIDE.search(text.strip()):  # shorter lines are captions and asides
                    yield f"globalvoices-{Path(name).stem}-{s.get('id')}", text, "en"


IAPP_ASKS = re.compile("อะไร|ใด|ใคร|ไหน|เท่าไร|เท่าไหร่|กี่|หรือไม่|ไหม|อย่างไร|ยังไง|เมื่อไร|เมื่อไหร่|ทำไม")
# A question that points into its article, has passage text spliced on after its subject, or has no subject.
IAPP_POINTS = re.compile("นี้|ดังกล่าว|ในที่นี้|^(?:เธอ|เขา|เกาะ|รายการ|ละคร|มี|ได้รับ|ภายใน|ในการ|ต่อมา|พื้นที่)|^ในปี .{0,12}(?:เธอ|เขา)| (?:ซึ่ง|ด้วย|หรือ|ในขณะที่|ปัจจุบัน)")
WIRE = re.compile(r"รอยเตอร์|เอพี|เอเอฟพี|Reuters|\bAP\b|\bAFP\b|Associated Press|Agence France")


def voa_thai(path="mot/tha_voathai.tgz"):
    """Voice of America's Thai news articles, from Multilingual Open Text. VOA's own writing is in the public
    domain, but what it takes from Reuters, AP and AFP is not, so an article that names any of them anywhere
    is left out whole. So are the oldest articles (numbers under a million), whose words are spaced wrongly.
    Captions and programme listings (the audio, video and photo records) are not read, and a paragraph that
    holds " / " is a digest of several headlines."""
    with tarfile.open(RAW / path) as archive:
        for member in archive:
            if member.isfile() and "/article/" in member.name:
                article = json.load(archive.extractfile(member))
                number = article["filename"].rsplit("_", 1)[-1]
                if not (number.isdigit() and int(number) < 1_000_000) and not WIRE.search("\n".join(article["paragraphs"])):
                    yield from thai_rows(f"voathai-{number}", (p for p in article["paragraphs"] if "ที่มา" not in p[:12] and " / " not in p))


def blognone(path="blognone/train.parquet"):
    """Technology news posts. The closing line that credits the story's source is left out, and so are list
    items and headings: a paragraph that opens with a bullet or other mark, or is under 100 characters."""
    import pyarrow.parquet as pq

    for i, content in enumerate(pq.read_table(RAW / path, columns=["content"])["content"].to_pylist()):
        yield from thai_rows(f"blognone-{i}", (p for p in content.split("\n") if len(p.strip()) >= 100 and re.match(r"\s*[\wก-๙\"“(]", p) and not p.lstrip().startswith("ที่มา")))


def iapp(pattern="iapp/*.parquet"):
    """Questions people wrote in Thai about Wikipedia articles. The articles themselves are not taken, so a
    question that points into its article ("this show", "she") or has no question word is left out."""
    import pyarrow.parquet as pq

    for path in sorted(RAW.glob(pattern)):
        table = pq.read_table(path, columns=["question_id", "question"]).to_pylist()
        for r in table:
            if len(r["question"].strip()) >= 25 and IAPP_ASKS.search(r["question"]) and not IAPP_POINTS.search(r["question"]):
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
    name_cap: int = 0  # most rows that may start with the same name ("Tom ...", "Yanni ...")
    dialogue: bool = False  # rows carry the turn before them, and one that leans on it and has none is dropped
    frames: bool = False  # a task dialogue: capped per masked frame, closings and recogniser garble dropped
    asked: bool = False  # questions and answers written to or for an assistant: instructions and chatbot voice go
    royal: bool = False  # encyclopaedia and news prose, the only places a mention of the Thai royal family stays
    stubs: bool = False  # keep one in five of the encyclopaedia's stock sentences ("was born in", "is a town in")
    strict: bool = False  # a demonstrative anywhere in the sentence drops it: for sources of loose single sentences
    modern: bool = False  # leave out sentences that read as old fiction or quote someone speaking
    spoken: bool = False  # speech and casual chat: a trailing connective is trimmed, and only a row that cannot stand as an utterance goes
    reported: bool = False  # news that quotes people: a row that stops within 20 characters of a "said that" or a "but" was cut there
    carried: bool = False  # a turn after the first is kept only with the line before it, and not when that line asks who the customer is
    max_chars: int = 0  # a row limit of its own, for the whole-paragraph sources
    max_pieces: int = 0
    use: str = ""  # "forward_only": the pair made from the row may be trained on as written, source to target, and never reversed
    repairs: str = ""  # what the extractor did to the text before any filter, recorded in the manifest


PLAIN = Source("", "", "", "", "", "", "", None, None)


# Whole paragraphs may be longer than a sentence row, but not so long that the English side of the pair passes the
# trainer's 128 pieces: English takes 1.3 times the pieces of its Thai, so 80 Thai pieces is 104 on average.
PARAGRAPH = {"max_chars": 600, "max_pieces": 80}
WIKIMEDIA = "contributors, each row's id holds the page id, https://{}/?curid=<page id>"
SOURCES = (
    # Scarce chat first, encyclopaedias last, so the plentiful sources lose every duplicate.
    # A take of None gives every usable row, and 0 none: the source is built and counted but not taken.
    # Tatoeba and SODA are capped at 100,000 because they repeat themselves.
    Source("byu_chitchat", "BYU Chit-Chat Dataset", "informal chat", "MIT", MIT,
           "Copyright (c) 2019 Perception, Control and Cognition Lab, Brigham Young University",
           "https://github.com/BYU-PCCL/chitchat-dataset", None, byu, dialogue=True),
    Source("abcd", "Action-Based Conversations Dataset", "customer-service chat", "MIT", MIT,
           "Copyright (c) 2021 ASAPP Research", "https://github.com/asappresearch/abcd", None, abcd, dialogue=True, frames=True, carried=True),
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
           "Copyright (c) 2021 iApp Technology Co., Ltd.", "https://github.com/iapp-technology/iapp-wiki-qa-dataset", 0, iapp),  # 76 good rows in 100 after the tidy pass: not taken
    Source("lotusdis", "LOTUSDIS", "spoken conversation", "CC BY-SA 4.0", BY_SA,
           "NECTEC, National Science and Technology Development Agency, Thailand",
           "https://huggingface.co/datasets/nectec/LOTUSDIS", 0, lotusdis, spoken=True),  # 78 whole utterances in 100, under the bar of 85: built and counted, not taken
    Source("yodas", "YODAS, Thai uploader-supplied captions (th000)", "spoken, video captions", "CC BY 3.0", "https://creativecommons.org/licenses/by/3.0/",
           "The uploader of each video. Each row's id holds the video id, https://www.youtube.com/watch?v=<id>. Collected by Li and others, YODAS, 2023",
           "https://huggingface.co/datasets/espnet/yodas", 0, yodas, spoken=True),  # failed three independent samples: presenter monologue, not chat
    Source("oasst2_th", "OpenAssistant Conversations (OASST2), Thai", "questions typed to an assistant", "Apache-2.0", APACHE,
           "OpenAssistant contributors, LAION", "https://huggingface.co/datasets/OpenAssistant/oasst2", None, lambda: oasst2("th"), spoken=True),
    Source("commonvoice_th", "Common Voice sentences, Thai", "everyday sentences", "CC0 1.0", CC0,
           "None required", "https://github.com/common-voice/common-voice", None, common_voice_th),
    Source("tatoeba_th", "Tatoeba, Thai", "everyday sentences", "CC BY 2.0 FR", "https://creativecommons.org/licenses/by/2.0/fr/",
           "Tatoeba contributors, each row's id holds the sentence number, https://tatoeba.org/sentences/show/<number>",
           "https://tatoeba.org", None, lambda: tatoeba("tatoeba/tha_sentences_detailed.tsv", "th")),
    Source("tatoeba_en", "Tatoeba, English", "everyday sentences", "CC BY 2.0 FR", "https://creativecommons.org/licenses/by/2.0/fr/",
           "Tatoeba contributors, each row's id holds the sentence number, https://tatoeba.org/sentences/show/<number>",
           "https://tatoeba.org", 100_000,
           lambda: tatoeba("tatoeba/eng_sentences_detailed.tsv", "en", "tatoeba/eng_sentences_CC0.tsv.bz2", min_words=6), opener_cap=10, modern=True, name_cap=2_000),
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
    Source("blognone_para", "Blognone posts, whole paragraphs", "technology news", "CC BY 3.0 TH", "https://creativecommons.org/licenses/by/3.0/th/",
           "As for blognone", "https://www.blognone.com", None, lambda: whole_paragraphs(blognone()), prose=True, **PARAGRAPH),
    Source("voathai", "Voice of America Thai, articles (Multilingual Open Text 1.11)", "news", "Public domain", "https://www.voanews.com/p/5338.html",
           "Voice of America, each row's id holds the article number, https://www.voathai.com/a/<number>.html. "
           "Collected by Palen-Michel, Kim and Lignos, Multilingual Open Text, whose collection is CC BY 4.0",
           "https://github.com/bltlab/mot", None, voa_thai, prose=True, royal=True, reported=True),
    Source("voathai_para", "Voice of America Thai, whole paragraphs", "news", "Public domain", "https://www.voanews.com/p/5338.html",
           "As for voathai", "https://github.com/bltlab/mot", None, lambda: whole_paragraphs(voa_thai()), prose=True, royal=True, reported=True, **PARAGRAPH),
    Source("thailaw", "Thai Law Dataset v1.0, sections of Acts and decrees", "legal", "Public domain (Thai Copyright Act, section 7)", CC0,
           "Collected by PyThaiNLP (Wannaphong Phatthiyaphaibun), released CC0. Each row's id holds the document's number in the dataset",
           "https://huggingface.co/datasets/pythainlp/thailaw-v1.0", 50_000, thai_law, prose=True),
    Source("thwiki", "Thai Wikipedia", "encyclopaedic", "CC BY-SA 4.0", BY_SA,
           "Wikipedia " + WIKIMEDIA.format("th.wikipedia.org"), "https://th.wikipedia.org", None,
           lambda: wiki("thwiki", "th"), prose=True, royal=True, use="forward_only"),  # good enough to read, not to imitate
    Source("thwiki_para", "Thai Wikipedia, whole paragraphs", "encyclopaedic", "CC BY-SA 4.0", BY_SA,
           "As for thwiki", "https://th.wikipedia.org", 0, lambda: whole_paragraphs(wiki("thwiki", "th")), prose=True, royal=True, **PARAGRAPH),  # 67 good rows in 100: not taken
    Source("simplewiki", "Simple English Wikipedia", "plain-language encyclopaedic", "CC BY-SA 4.0", BY_SA,
           "Wikipedia " + WIKIMEDIA.format("simple.wikipedia.org"), "https://simple.wikipedia.org", None,
           lambda: wiki("simplewiki", "en", 20), prose=True, strict=True, royal=True, stubs=True),
    Source("commonvoice_wiki", "Common Voice sentences taken from English Wikipedia", "encyclopaedic", "CC0 1.0", CC0,
           "None required", "https://github.com/common-voice/common-voice", None, common_voice_wiki, prose=True, strict=True, stubs=True),
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
    """A task-dialogue line with what varies masked: numbers, dates and times, and capitalised words inside the
    sentence (names, places, slot values). "Book a table at Nopa for 7 pm" and "Book a table at Zuni for 8 pm" share a frame."""
    words = text.split()
    masked = ["X" if w.startswith("Xx") or (i and w[:1].isupper() and not w.startswith("I") and words[i - 1][-1] not in ".?!") else w.casefold()
              for i, w in enumerate(words)]
    return PUNCTUATION.sub(" ", re.sub(r"(?:[X0]\W*)+", "X ", re.sub(r"\d+(?:st|nd|rd|th)?", "0", MONTHS.sub("X", " ".join(masked))))).strip()


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
    steps += [("trailing connective removed", lambda t: SPOKEN_TAIL.sub("", t))] * source.spoken
    for name, fix in steps:
        fixed = clean(fix(text))
        if fixed != text:
            text, repairs = fixed, repairs + [name]
    return text, repairs


def uncoupled(text, lang):
    pattern = THAI_CONNECTIVE if lang == "th" else CONNECTIVE
    rest = pattern.sub("", text, count=1) if pattern.match(text) else text
    return rest[:1].upper() + rest[1:] if rest != text else text


def stutters(text, run=10):
    """Whether the text says the same thing twice in a row: a run of at least `run` characters followed at once
    by itself, as a careless edit leaves. (A name that comes back later in the sentence is not that.)"""
    first = {}
    for i in range(len(text) - run + 1):
        start = first.setdefault(text[i:i + run], i)
        if i - start >= run and text.startswith(text[start:i].strip(), i):
            return True
    return False


def reject(text, lang, source=PLAIN, row_id=""):
    """Why a cleaned row is not worth translating, or None."""
    solid = text.replace(" ", "")
    words = text.split()
    thai, prose = lang == "th", source.prose
    if CONTACT.search(text) or (source.dialogue and NAME_ONLY.match(text)) or (source.frames and PERSONA.search(text)):
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
    if prose and thai and (not THAI_VERB.search(BRACKETED.sub("", text)) or not re.match("[\"“']?[ก-ฮเ-ไA-Z]", text) or THAI_BOILERPLATE.search(text) or THAI_NOTE.match(text) or THAI_HOLE.search(text) or stutters(text)):
        return "not a sentence"
    if thai and " - " in text:
        return "not a sentence"  # "song - artist", or two speakers on one line
    if ADULT.search(text) or ADULT_THAI.search(text):
        return "adult"
    if not source.royal and ROYAL.search(text):
        return "royal family"
    if (len(words) < 3 if not thai else len(text) < (40 if prose else 8)):
        return "too short"
    if len(text) > (source.max_chars or MAX_CHARS[lang]):
        return "too long"
    if thai and (THAI_DANGLING.search(text) or (THAI_IF.match(text) and not THAI_THEN.search(text)) or (source.reported and REPORTED.search(text))):
        return "unfinished"
    if source.spoken and thai and (text.count(" ") > len(text) / 10 or re.match(r"\d+\.", text) or re.search(r"\.\.|…|:|[一-鿿]|ซับโดย|ซับไทย|คำบรรยาย", text)):
        return "not a sentence"  # captions typed a word at a time, and numbered steps
    if source.spoken and thai:
        if SPOKEN_LEANS.match(text):
            return "leans on the row before"
    elif thai and (THAI_START.match(text) or THAI_LOGIC.match(text)):
        return "leans on the row before"
    elif thai and prose and THAI_SUBJECTLESS.match(THAI_OPENER.sub("", text, count=1).lstrip(" ,")):
        return "subject in the row before"
    if source.strict and DEMONSTRATIVE.search(text):
        return "leans on the row before"
    if (prose or source.asked) and not thai and (UNRESOLVED.match(text) or POINTER.match(text)):
        return "leans on the row before" if UNRESOLVED.match(text) else "pointer"
    if not thai and GAP.search(text):
        return "words missing"
    if source.asked and (INSTRUCTION.match(text) or CHATBOT.search(text) or (not thai and not SENTENCE_END.search(text))):
        return "instruction or chatbot voice"
    if source.stubs and (STUB.search(text) and digest(row_id) % 5 or BIOGRAPHY_LEAD.match(text) and digest(row_id) % 4 or FOOTBALLER.search(text) and digest(row_id) % 2):
        return "stock sentence"
    if source.modern and OLD_FICTION.search(text):
        return "old fiction"
    if source.frames and STAMMER.search(text):
        return "garbled speech"
    if source.frames and len(words) <= 8 and CLOSING.search(text):
        return "only a closing"
    return None


def before(row):
    return row["source"], "after", key(row.get("context", ""))


def name_of(row):
    """The row's first word when it is a name, for the cap on rows that begin with the same one."""
    first = row["text"].split()[0].strip(",.!?'\"")
    return row["source"], "name", first if first[:1].isupper() and first.replace("'", "’").split("’")[0].casefold() not in COMMON_STARTS else ""


def sift(source, batch, pieces, blocked, benchmark, seen, openers, counts):
    """Clean one batch of (id, text, language) or (id, text, language, turn before) and return the rows that
    pass every filter, counting the rest."""
    rows = []
    shapes = {}
    for row_id, text, lang, *more in batch:
        text, repairs = prepare(text, lang, source)
        context = clean(more[0]) if more and more[0] else None
        shapes[row_id] = more[1] if len(more) > 1 else text
        if context and (len(context) > MAX_CONTEXT * (1 + source.prose) or CONTACT.search(context) or PERSONA.search(context) or URL.search(context) or MARKUP.search(context)
                        or ADULT.search(context) or ROYAL.search(context)):
            context = None
        if context and source.prose and lang == "th" and (THAI_DANGLING.search(context) or THAI_HOLE.search(context) or template(text)[:20] in template(context)):
            context = None  # cut off, holed, or an earlier copy of the row itself
        if context and source.prose and lang == "th" and ((THAI_IF.match(context) and not THAI_THEN.search(context)) or REPORTED.search(context)):
            context = None  # a previous line that does not end a sentence
        reason = reject(text, lang, source, row_id)
        if reason == "subject in the row before" and context:
            reason = None  # the sentence that names the subject is carried with it
        if not reason and not context and source.prose and lang == "th" and THAI_NEEDS_SUBJECT.match(text):
            reason = "subject in the row before"
        if not reason and source.dialogue and not context and (len(text.split()) < 5 or LEANS.match(text)):
            reason = "needs the turn before"
        if not reason and source.carried and more and more[0] and (not context or ASKS_IDENTITY.search(context)):
            reason = "contact details" if context else "needs the turn before"
        if not reason and context and key(text) in key(context):
            reason = "repeats the turn before"
        if reason:
            counts[reason] += 1
        else:
            rows.append({"id": row_id, "text": text, "lang": lang, "source": source.name, "register": source.register, "licence": source.licence}
                        | ({"context": context} if context else {}) | ({"repairs": repairs} if repairs else {}) | ({"use": source.use} if source.use else {}))
    kept = []
    for r, size in zip(rows, pieces([r["text"] for r in rows])):
        mark = digest(template(r["text"]))
        start = (source.name, opener(r["text"]))
        shape = (source.name, frame(shapes[r["id"]]))
        reason = ("too long" if size > (source.max_pieces or MAX_PIECES[r["lang"]]) else "already in project data" if mark in blocked
                  else "benchmark overlap" if mark in benchmark or (r["lang"] == "en" and not benchmark.isdisjoint(shingles(r["text"])))
                  else "duplicate" if mark in seen
                  else "repeated opener" if source.opener_cap and openers[start] >= source.opener_cap
                  else "repeated frame" if source.frames and openers[shape] >= FRAME_CAP
                  else "repeated previous line" if source.frames and r.get("context") and openers[before(r)] >= PREVIOUS_CAP
                  else "repeated name" if source.name_cap and name_of(r)[2] and openers[name_of(r)] >= source.name_cap else None)
        if reason:
            counts[reason] += 1
            continue
        seen.add(mark)
        openers[start] += 1
        openers[shape] += 1
        openers[before(r)] += 1
        openers[name_of(r)] += 1
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
        "oasst2/ready.messages.jsonl.gz": f"{HF}/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/2023-11-05_oasst2_ready.messages.jsonl.gz",
        "lotusdis/transcripts.jsonl": LOTUSDIS,  # text columns only, read out of the audio files by fetch_lotusdis
    }
    files |= {f"thailaw/{i}.parquet": f"{HF}/pythainlp/thailaw-v1.0/resolve/4c1e2c091b15b5f01d836d5a23398594157e021c/data/train-0000{i}-of-00002.parquet" for i in range(2)}
    files |= {f"yodas/{i:08}.txt": f"{YODAS}/{i:08}.txt" for i in range(52)}
    files |= {f"commonvoice/th-{n}": f"{COMMON_VOICE}/th/{n}" for n in ("Wannaphong.txt", "thsentences.txt")}
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


class RangeFile(io.RawIOBase):
    """Read parts of a remote file, so a parquet reader can take two text columns and leave the audio."""

    def __init__(self, url):
        self.url, self.at = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.at

    def seek(self, offset, whence=0):
        self.at = offset + (0, self.at, self.size)[whence]
        return self.at

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.at + n, self.size)
        if end <= self.at:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.at}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
        self.at += len(data)
        return data


def fetch_lotusdis(target):
    import pyarrow.parquet as pq

    with open(target, "w", encoding="utf8") as out:
        for split, parts in LOTUSDIS_PARTS.items():
            for part in range(parts):
                table = pq.ParquetFile(RangeFile(f"{LOTUSDIS}/{split}-{part:05}-of-{parts:05}.parquet")).read(columns=["sentence", "speaker_id", "mic"])
                for r in table.to_pylist():
                    out.write(json.dumps(r | {"split": split}, ensure_ascii=False) + "\n")


def fetch():
    for path, url in downloads().items():
        target = RAW / path
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            if url == LOTUSDIS:
                fetch_lotusdis(partial)
            else:
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
    cut = {s.name: (1 << 64) if s.take is None or s.take >= len(ranks[s.name]) else sorted(ranks[s.name])[s.take - 1] if s.take else -1 for s in SOURCES}
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
                total.update([("source", r["source"]), ("lang", r["lang"]), ("register", r["register"]), ("licence", r["licence"]), ("use", r.get("use", "both_directions"))])
                line = (json.dumps(r, ensure_ascii=False) + "\n").encode()
                sha.update(line)
                out.write(line)

    def by(kind):
        return {name: n for (k, name), n in sorted(total.items()) if k == kind}

    manifest = {
        "retrieved": RETRIEVED, "rows": sum(by("lang").values()), "sha256": sha.hexdigest(),
        "by_language": by("lang"), "by_register": by("register"), "by_licence": by("licence"), "by_use": by("use"),
        "lengths": {group: {"characters_mean": round(sum(chars) / len(chars), 1), "characters_p95": percentile(chars, 0.95),
                            "pieces_mean": round(sum(sizes) / len(sizes), 1), "pieces_p95": percentile(sizes, 0.95)}
                    for group, (chars, sizes) in sorted(lengths.items())},
        "sources": {s.name: {"title": s.title, "repairs": s.repairs, "use": s.use or "both_directions", "register": s.register, "licence": s.licence, "licence_url": s.licence_url,
                             "credit": s.credit, "homepage": s.homepage, **counted["sources"][s.name],
                             "target": s.take, "in_mix": total["source", s.name]} for s in SOURCES},
        "files": counted["files"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps({k: manifest[k] for k in ("rows", "by_language", "by_register", "lengths")}, indent=1, ensure_ascii=False))


# sha256 of every download. A file that no longer matches stops the build.
PINS = {
    "thailaw/0.parquet": "e1f7c448cea6be2297f8ec3f6ddcfacdc8c13b441ef61e1ef8201a2f236ff9bc",
    "thailaw/1.parquet": "073e2d9571e2107fe7d17b94c5dcf565ea682c40d3f844c08bb2d7ccb85da303",
    "oasst2/ready.messages.jsonl.gz": "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751",
    "lotusdis/transcripts.jsonl": "afbd17850af003c18e76589cd320498ada8094870392f10e1dccef08ed571772",
    "yodas/00000000.txt": "f90b27ee2723577aa14999bbc818016bbbc3271d0eb4fab191be721ba8aac5e4",
    "yodas/00000001.txt": "593e0ba2e417b3fcea4d4d870a223131a5ed2e192e79bf332a6e8378f15169c1",
    "yodas/00000002.txt": "fb5c677a646102e743f064724f770c301c4321d118aea4766e9d533c24ad03e1",
    "yodas/00000003.txt": "b9cc899fc37877b295f1f42a3b1c4a7212ee4dc70d2a15f7c40daceaa79c3540",
    "yodas/00000004.txt": "cc434235ef42caca9d53ea099c11b7a41cebaa3d9d9cc2c8f640651c5f286394",
    "yodas/00000005.txt": "1a7a2c258e230b80e8ca0cb4e4fb02cf7e673483c12fa57fcca4106024080fa7",
    "yodas/00000006.txt": "889647afa13f612a678d015f4393ac85df78efa71e0b1510052899f990947407",
    "yodas/00000007.txt": "cc2d4522b0f79020bcb47bcd2736e56575b06eeaad82aa1a11a9c96eb39d7007",
    "yodas/00000008.txt": "b4dd4a57e2349d07075588118c4fa80d48db283ea5a58d62d4cbf0e801b871cc",
    "yodas/00000009.txt": "fbb352cea57fea105254d855930140fb5b1fb22b7b1654861d28743bb6d5511d",
    "yodas/00000010.txt": "07da61076199a9bb5eb3069cf6c91cf041c9b17ce641ed7c3add1b8d7ffc1738",
    "yodas/00000011.txt": "5cca58661f7e73647cb7b4f7d3e734aacee8a9f2c3a58f31153225a0c5498057",
    "yodas/00000012.txt": "7dbbfd5d868b6f6e26b12b14317926ec34423db7ccd666eee3d539d9890d385d",
    "yodas/00000013.txt": "d2b69878efd8f3de580b59eb506dc7dd5a71d516bf793a340b91ed374863893c",
    "yodas/00000014.txt": "1a2f5bad98ffd7623e70695596047c7c07d7a3096b129207a6c2f3293353c204",
    "yodas/00000015.txt": "f0e97774dbaf8680059a7a073aaac12b654653d192d4e0ac71b82f82ca225b8d",
    "yodas/00000016.txt": "4573e5a93a7188d519d144de876c4140eb9bef1f561fc2b62453a3e299736257",
    "yodas/00000017.txt": "54cec1e97fdb038b3d122c012f243a441be18f737c510da9cbfca360fbde1c54",
    "yodas/00000018.txt": "ad56c0bd9849e1cc03876725ce895110bfadc853f9a7c8fb36dbd17286d48656",
    "yodas/00000019.txt": "3bb35ec27c7d789872322cfd310eadc8f7cf30cc55afa32df2de33a4fde3daf5",
    "yodas/00000020.txt": "bae5b699bdd4159c138fabb35080ea073cddf2671f3025a0c18ff95662f2b8d9",
    "yodas/00000021.txt": "18638479d25a5df177d37807b2cb8037c051c96917718dba58c05609df0a765f",
    "yodas/00000022.txt": "da21cefb60b8c6daa0b2e17fccc4e24755dc11dc4e37f7ec84565d8875c4f4ec",
    "yodas/00000023.txt": "8f11231772a7751331b697a4fe5d7f7ea6f91a93555a8d92d6b41981c83e2ce9",
    "yodas/00000024.txt": "3588cf915163c490f730d500e5f623e3a0d0536f7d5b5cd4f53d9a40c9d5b456",
    "yodas/00000025.txt": "14b2921ffe78b82623a14a54c0003d0176fce693ad9ef3fc5a8423818fcc56af",
    "yodas/00000026.txt": "5ab3f7dd99fcf655726b601ea40ebefd0f52681cb0073e7d4f36edddb4404340",
    "yodas/00000027.txt": "c518ce757b1af111d20ca940fdb649e9c338a25d10f53bd4934f2fbe932a235e",
    "yodas/00000028.txt": "d7cb71cbc01cc83dab42f6400595a12d6f565a66004ccba2152be957cc0888d1",
    "yodas/00000029.txt": "442596159e42bca581f21926390227062762cfaee9070590b8faed25e66aadad",
    "yodas/00000030.txt": "8133974082af8dec0b30d3dab76ec7d5453f6867a9385e0d551701d66265e9ae",
    "yodas/00000031.txt": "bfbc26e164a0d569957a8248c7f2044ed6ddc5cc4983ca861db63c94aafb3556",
    "yodas/00000032.txt": "a52e49ada84ba9c03840d3bdf5e870ca0ebf070fcadc271f337ff746ddcca5f5",
    "yodas/00000033.txt": "918974c88ad9bbdc8c47e5eacdedf285a37170362f43ae7dd6d3cb81cf4bc17b",
    "yodas/00000034.txt": "2ae15b1c0510a00cf61b00d10219504269beded46aaf482e1f223649481075aa",
    "yodas/00000035.txt": "bac6794a5a032579c0525f4f40d6aeb2fb303789f85d695d6f9cab32f535d0f9",
    "yodas/00000036.txt": "a53ee0848ab6adb878bdcd5a9f0ab13a869169fe3d0f5b7650fc128023daa2b0",
    "yodas/00000037.txt": "de284cab98ea0c9489fadf3730c2389d607f785a8447104d6e9cf1a5a1136086",
    "yodas/00000038.txt": "ddd9e03a766b802ab61e2266e71d3dd0bd21919541e385b7d49db478aa643de9",
    "yodas/00000039.txt": "9c95c5dac7c1d1e5e7e53918d650cb3de986f3584cf2c67b1e7e18c756f97231",
    "yodas/00000040.txt": "b4e35695d5318771b7682283953cc56689bd1fd5d9f870468d7345c6848921fe",
    "yodas/00000041.txt": "61595e51133d5f59026d0722eb75abe50d6951d72a1a0fcb81f58d0b2df866ab",
    "yodas/00000042.txt": "78247abaa196079c2912cba936c19b688f358ca3cda34ac66be8769fc8dea8cd",
    "yodas/00000043.txt": "4133bfc618cc2d305b45df180ad49d203b1a23b818fb764d3ea8e489542c3405",
    "yodas/00000044.txt": "82ee61e8a9dc406bad3d5ad1bce5401eede2784ae95c2d42cf345517fb2bf7dc",
    "yodas/00000045.txt": "8ed2d4df62142a3df96a53ebb67a19881d60bc7561a6fa6a4455d6c662e50380",
    "yodas/00000046.txt": "f32e87a936cf11c195bd141d499a4be259ad4f488f70719972a6514fbab14d79",
    "yodas/00000047.txt": "4654de90583f889f8e9cfb4e6288319864f63b112eaa454343ef4da51806e748",
    "yodas/00000048.txt": "e7b1fe24f173e749c509903f73a531bc815c6274a8b5866bcd3fc1a95e60fa89",
    "yodas/00000049.txt": "202b83c8fbefef62ab536edc4235e36f9a34b0dfddd67b021404460cc8bb1374",
    "yodas/00000050.txt": "3c545faf681352c8f3ffd6656ef5f20308954351782811e77a4db3914793a0f1",
    "yodas/00000051.txt": "fd487477800edf94b2894dfb0b1beb3e4b5836c9ab7009ee95ca345ca6687bd1",
    "commonvoice/th-Wannaphong.txt": "32a2ea93e45d9a5c12e1a60aa21eef5b07f21b992e2d672417170dca720e5b5e",
    "commonvoice/th-thsentences.txt": "0f13e9bc804d07bf9848a36a201e834a41878e374371cf361098c69b0d272cf4",
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
