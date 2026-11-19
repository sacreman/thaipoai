"""Thai web text from crawl collections, a slice at a time, through the source builder's filters.

A crawl is too large to keep, so a slice is streamed, the rows that pass are written, and the slice is deleted
by whoever fetched it. Every row is tier 2 and carries its collection, the page's domain and a few statistics
to sort by later; nothing here calls a model. Run with the packages of the project's "crawl" dependency group:

uv run --group crawl python -m train.crawl fineweb2_th 000_00002 path/to/000_00002.parquet --listed          (short whole paragraphs)
uv run --group crawl python -m train.crawl fineweb2_th 000_00002.long path/to/000_00002.parquet --listed --long   (those of 331 to 700 characters)
uv run --group crawl python -m train.crawl pantip_th all path/to/Pantip_QA_200000_20220220.csv
uv run --group crawl python -m train.crawl tally fineweb2_th          (rewrite the slices to what counts, and count it)

hplt_th and mangosteen_th can still be read, as the first trials were, but are left out: the HPLT server now
refuses the files, and Mangosteen's web part gives no page addresses to cut by.

Each run writes data/distill/crawl/<collection>/<slice>.jsonl and <slice>.json, and skips a slice already done.

The build is the same reads made from an empty checkout, a job a download, each pinned in crawl_inputs.tsv:

uv run --group crawl python -m train.crawl jobs                 (the jobs)
uv run --group crawl python -m train.crawl build <job> | all    (fetch, check and read into data/distill/build)
uv run --group crawl python -m train.crawl merge                (settle repeats across slices: build/duplicates.tsv)
uv run --group crawl python -m train.crawl verify               (compare with crawl_build.json; `record` rewrites it)
"""

import argparse
import array
import collections
import dataclasses
import gzip
import hashlib
import html
import io
import itertools
import json
import multiprocessing
import os
import pathlib
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request

import train.distill_sources as d

OUT = d.OUT / "crawl"
ODC_BY = "https://opendatacommons.org/licenses/by/1-0/"
WEB = {"prose": True, "royal": True, "tier": 2, "pool": "candidate", "score": "not sampled yet", **d.PARAGRAPH}
COLLECTIONS = {s.name: s for s in (
    d.Source("hplt_th", "HPLT v2 cleaned, Thai", "web", "CC0 1.0 (the collection)", d.CC0, "HPLT project and each page's author", "https://hplt-project.org/datasets/v2.0", None, None,
             finding="the collection is CC0; the pages are their authors', taken from the Internet Archive and Common Crawl with robots.txt honoured", read_on="https://hplt-project.org/datasets/v2.0", **WEB),
    d.Source("fineweb2_th", "FineWeb2, Thai", "web", "ODC-By 1.0 (the collection)", ODC_BY, "HuggingFaceFW and each page's author", "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2", None, None,
             finding="the collection is ODC-By 1.0 and subject to Common Crawl's terms of use; the pages are their authors'", read_on="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2", **WEB),
    d.Source("mangosteen_th", "Mangosteen web part (WangchanLION-Web)", "web", "ODC-By 1.0 (the collection)", ODC_BY, "AI Singapore, VISTEC and each page's author",
             "https://huggingface.co/datasets/aisingapore/WangchanLION-Web", None, None,
             finding="the collection is ODC-By 1.0; the pages are their authors'. It gives no address for a page, so no domain is recorded", read_on="https://huggingface.co/datasets/aisingapore/WangchanLION-Web", **WEB),
    d.Source("thaifin_th", "Thai financial dataset (the cleaned Thai part of CMDF_VISTEC)", "news", "CC BY 4.0 (the collection)", "https://creativecommons.org/licenses/by/4.0/",
             "PyThaiNLP, VISTEC and each publisher", "https://huggingface.co/datasets/pythainlp/thai-financial-dataset", None, None,
             finding="the card says cc-by-4.0, but the news is its publishers' and the annual reports are company filings: tier 2. The sub-source stands where a domain would",
             read_on="https://huggingface.co/datasets/pythainlp/thai-financial-dataset", **WEB),
    d.Source("prd_th", "PRD News 30112023 (Government Public Relations Department)", "news", "CC0 1.0", d.CC0, "Government Public Relations Department, collected by PyThaiNLP",
             "https://huggingface.co/datasets/pythainlp/prd_news_30112023", None, None,
             finding="government news published as open data on gdcatalog.go.th; the card says cc0-1.0. The department stands where a domain would",
             read_on="https://huggingface.co/datasets/pythainlp/prd_news_30112023", **(WEB | {"tier": 1})),
    d.Source("pantip_th", "Pantip_QA_200000_20220220 (Pantip posts and replies)", "informal forum", "none stated", None, "Pantip members; collected by Amity Solution",
             "https://huggingface.co/datasets/amitysolution/Pantip_QA_200000_20220220", None, None,
             finding="no card and no licence, and not released by Pantip: tier 2",
             read_on="https://huggingface.co/datasets/amitysolution/Pantip_QA_200000_20220220", **WEB),
    d.Source("krathu_th", "krathu-500 (574 Pantip posts with their comments)", "informal forum", "none stated", None, "Pantip members; collected by Pittawat Taveekitworachai",
             "https://github.com/Pittawat2542/krathu-500", None, None, finding="no licence in the repository, and not released by Pantip: tier 2",
             read_on="https://github.com/Pittawat2542/krathu-500", **WEB),
    d.Source("wikitalk_th", "Thai Wikipedia discussion pages (dump of 1 October 2026)", "discussion", "CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/",
             "Wikipedia contributors", "https://dumps.wikimedia.org/thwiki/20261001/", None, None,
             finding="talk, project and project-talk pages, and user-talk pages over 3 KB, of the current-pages dump; contributions are CC BY-SA 4.0",
             read_on="https://dumps.wikimedia.org/legal.html", **(WEB | {"tier": 1})),
    d.Source("wikitalk_en", "English Wikipedia discussion pages (dump of 1 November 2026)", "discussion", "CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/",
             "Wikipedia contributors", "https://dumps.wikimedia.org/enwiki/20261101/", None, None,
             finding="talk, project and project-talk pages, and user-talk pages over 3 KB, of the current-pages dump; contributions are CC BY-SA 4.0",
             read_on="https://dumps.wikimedia.org/legal.html", **(WEB | {"tier": 1})),
    d.Source("stackexchange_en", "Stack Exchange data dump, sites on everyday subjects", "questions and answers", "CC BY-SA (3.0 or 4.0 by the post's date)",
             "https://creativecommons.org/licenses/by-sa/4.0/", "the post's author; each row's id holds the site and post id, https://<site>.stackexchange.com/q/<post id>",
             "https://archive.org/details/stackexchange", None, None,
             finding="the dump's license.txt: all user content is CC BY-SA and attribution is required (the site, a link to the question, the author's name and a link to the author)",
             read_on="https://archive.org/download/stackexchange/license.txt", **(WEB | {"tier": 1})),
)}
COLLECTIONS["gazette_th"] = d.Source(
    "gazette_th", "Royal Gazette of Thailand, text of the open-law-data set", "official", "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/",
    "Secretariat of the Cabinet; text extracted by Open Law Data Thailand", "https://huggingface.co/datasets/open-law-data-thailand/soc-ratchakitcha", None, None,
    finding="the card says cc-by-4.0; laws, regulations and official announcements are not copyright works under section 7 of the Thai Copyright Act",
    read_on="https://huggingface.co/datasets/open-law-data-thailand/soc-ratchakitcha", **(WEB | {"tier": 1}))
OPENS_CLAUSE = re.compile(r"(?:ข้อ ?[0-9๐-๙]+|มาตรา ?[0-9๐-๙]+|\(?[0-9๐-๙]+[).]|[0-9๐-๙]+\.[0-9๐-๙]*|อาศัยอำนาจ|โดยที่|ทั้งนี้|ประกาศ ณ|ให้ไว้ ณ|หมวด ?[0-9๐-๙]+|ส่วนที่ ?[0-9๐-๙]+|บทเฉพาะกาล|หมายเหตุ)")


def unwrapped(text):
    """A Gazette document's text as paragraphs. Its lines were wrapped at the width of the page, so a line is joined to
    the one before unless that one stopped well short of the page's width or this one opens a clause. A line set in a
    font that came out with sara am where sara aa stood ("มำตรำ") is left out with the paragraph it sits in, and a
    clause's number is taken off its front."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    width = sorted(map(len, lines))[int(len(lines) * 0.9)]
    paragraphs, broken, short = [], False, True
    for line in lines:
        if short or OPENS_CLAUSE.match(line):
            if paragraphs and broken:
                paragraphs.pop()
            paragraphs.append(line)
            broken = False
        else:
            paragraphs[-1] += ("" if d.THAI.match(line[0]) and d.THAI.match(paragraphs[-1][-1]) else " ") + line
        broken = broken or (line.count("\u0e33") >= 3 and line.count("\u0e33") > line.count("\u0e32"))
        short = len(line) < 0.8 * width
    if broken:
        paragraphs.pop()
    # a clause keeps its text and loses its number, which would read as a label fused on
    return "\n".join(re.sub(r"^(?:ข้อ ?[0-9๐-๙]+|มาตรา ?[0-9๐-๙]+|\(?[0-9๐-๙]+[).])\s+", "", paragraph) for paragraph in paragraphs)


# The builder's tier 1 Thai prose, read again as whole paragraphs under the crawl's rules. Each takes its source's own
# settings (and so its licence and its special rules) and a name of its own; the builder's sentence rows stay as they are.
PARAGRAPHS = {"thwiki_p": ("thwiki", lambda: d.wiki("thwiki", "th")), "voathai_p": ("voathai", d.voa_thai), "blognone_p": ("blognone", d.blognone),
              "thwikisource_p": ("thwiki", lambda: d.wiki("thwikisource", "th")),  # Thai Wikisource: laws, old prose; verse does not pass
              "court_p": ("court", d.supreme_court), "thailaw_p": ("thailaw", lambda: itertools.chain(d.thai_law(), d.thai_law("thailawc", d.CLAUSE)))}
COLLECTIONS |= {name: dataclasses.replace(next(s for s in d.SOURCES if s.name == base), name=name, pool="candidate", score="not sampled yet", **d.PARAGRAPH)
                for name, (base, _) in PARAGRAPHS.items()}
COLLECTIONS["opensubtitles_en"] = dataclasses.replace(next(s for s in d.SOURCES if s.name == "opensubtitles_th"), name="opensubtitles_en",
                                                      title="OpenSubtitles v2018, English", pool="candidate", score="not sampled yet")
COLLECTIONS["opensubtitles24_th"] = dataclasses.replace(next(s for s in d.SOURCES if s.name == "opensubtitles_th"), name="opensubtitles24_th", title="OpenSubtitles Thai (OPUS OpenSubtitles v2024)",
                                                        licence_url="https://opus.nlpl.eu/legacy/OpenSubtitles-v2024.php", pool="candidate", score="not sampled yet", leaning=True)
COLLECTIONS["forum2_th"] = dataclasses.replace(COLLECTIONS["fineweb2_th"], name="forum2_th", title="FineWeb2, Thai: pages of discussion boards, diaries and web fiction, a post or a line a row",
                                               register="informal forum", leaning=True)
FORUMS = frozenset("anowl.co bloggang.com dek-d.com keedkean.com minimore.com my.dek-d.com rakongseries.blogspot.com topicstock.pantip.com tunwalai.com writer.dek-d.com".split())  # the listed domains read again for forum2_th
HELD_AS = {"opensubtitles24_th": "already in the 2018 set", "forum2_th": "already kept from these pages"}  # what a row is counted under when an earlier collection holds it


class Marks:
    """Row marks in a sorted array: a membership test without a set of tens of millions of numbers."""
    def __init__(self, marks):
        import numpy
        self.numpy, self.sorted = numpy, numpy.sort(numpy.frombuffer(marks, dtype=numpy.uint64))

    def __contains__(self, mark):
        at = self.numpy.searchsorted(self.sorted, self.numpy.uint64(mark))
        return bool(at < len(self.sorted) and self.sorted[at] == mark)


SPOKEN = {"opensubtitles24_th"}  # rows of the builder's subtitle reader: they go to its filters as they are, and a row already in the 2018 set is left out
SUBTITLE_FILMS, SUBTITLE_FILM_WORDS, SUBTITLE_WORDS = 0.3, 5000, 150_000_000  # the share of films read, the most words taken from one, and where reading stops
EN_NOISE = re.compile(r"\[[^\]]*\]|\([^)]*\)|<[^>]+>|\{[^}]*\}|^[A-Z][A-Z .'-]{1,20}:\s*")  # sound and stage notes, markup, a speaker's name
EN_CREDIT = re.compile(r"(?i)subtitle|subtitulos|synced|sync by|corrected by|translat|caption|opensubtitles|www\.|\.com|\.org|ripped by|encoded by|downloaded|©|copyright")


def subtitle_files(archive):
    """The subtitle files english_subtitles() reads from the archive: the first file, by name, of each film chosen."""
    import hashlib
    done = set()
    for name in sorted(n for n in archive.namelist() if n.endswith(".xml")):
        film = name.split("/")[-2]
        if film not in done and int(hashlib.sha256(film.encode()).hexdigest()[:8], 16) < SUBTITLE_FILMS * 2 ** 32:
            done.add(film)
            yield name


def english_subtitles(path):
    """English film and television subtitles from OPUS OpenSubtitles v2018, an utterance a row with the one before as
    context. One subtitle file a film, for a share of films chosen by a hash of the film's number, and no more than
    SUBTITLE_FILM_WORDS words from one. Left out: lyrics, credits, sound and stage notes, speakers' names. A line is
    joined to the one before when that one has not finished, they are within the builder's gap, and no dash marks a
    new speaker."""
    import hashlib
    import xml.etree.ElementTree as ET
    import zipfile
    total = 0
    with zipfile.ZipFile(path) as archive:
        for name in subtitle_files(archive):
            film, file_id = name.split("/")[-2], name.split("/")[-1][:-4]
            try:
                lines = [("".join(s.itertext()), [t for t in map(d.subtitle_seconds, (t.get("value") for t in s.iter("time"))) if t is not None]) for s in ET.fromstring(archive.read(name)).iter("s")]
            except ET.ParseError:
                continue
            rows, ended, singing = [], None, False  # [text, start, end], or None where a line was left out
            for raw, times in lines:
                lyric = singing or bool(re.search(r"[♪♫#]", raw))
                singing = (singing or bool(re.match(r"\s*-?\s*[♪♫#]", raw))) and not re.search(r"[♪♫#]\s*$", raw)
                dash = raw.lstrip().startswith("-")
                text = d.clean(EN_NOISE.sub(" ", raw)).lstrip("- ").strip()
                start, ended = (times[0] if times else ended), (times[-1] if times else ended)
                if lyric or not text or EN_CREDIT.search(text):
                    rows.append(None)
                    continue
                last = rows[-1] if rows else None
                close = last and start is not None and last[2] is not None and start - last[2] <= d.SUBTITLE_GAP
                if close and not dash and not re.search(r"[.?!\"”]$", last[0]) and len(last[0]) + len(text) < 300:
                    last[0], last[2] = f"{last[0].rstrip(' .…-')} {text.lstrip(' .…')}", ended
                else:
                    rows.append([text, start, ended])
            previous, words = None, 0
            for n, row in enumerate(rows):
                if row is None:
                    previous = None
                    continue
                near = previous and row[1] is not None and previous[1] is not None and row[1] - previous[1] <= d.SUBTITLE_GAP
                yield f"{film}-{file_id}-{n}", row[0], "en", previous[0] if near else None
                previous, words = (row[0], row[2]), words + len(row[0].split())
                if words >= SUBTITLE_FILM_WORDS:
                    break
            total += words
            if total >= SUBTITLE_WORDS:
                return


COLLECTIONS["gutenberg_en"] = d.Source(
    "gutenberg_en", "Project Gutenberg, English fiction by authors born from 1840", "fiction", "Public domain in the USA", "https://www.gutenberg.org/policy/permission.html",
    "the book's author; each row's id holds the Project Gutenberg book number", "https://www.gutenberg.org/", None, None,
    finding="Project Gutenberg texts are in the public domain in the United States; its header, licence text and name are taken off, as its trademark terms ask",
    read_on="https://www.gutenberg.org/policy/permission.html", **(WEB | {"tier": 1, "prose": False}))  # not prose to the builder, whose prose rule drops a row that opens on a quotation mark; loose_en() still asks for a whole sentence
SPEECH = re.compile(r"[\"“”]|(?:^| )'[A-Z]")


def book_paragraphs(text):
    """The paragraphs of a Project Gutenberg text that hold speech, between its START and END lines: lines typed
    at a fixed width are joined, _emphasis_ marks go, and a paragraph with no quotation mark is left out."""
    body = re.split(r"\*\*\* ?START OF [^\n]*\n", text, maxsplit=1)[-1]
    body = re.split(r"\*\*\* ?END OF |End of (?:the )?Project Gutenberg", body, maxsplit=1)[0]
    out = []
    for paragraph in re.split(r"\n\s*\n", body.replace("\r", "")):
        paragraph = re.sub(r"\s+", " ", paragraph.replace("_", "")).strip()
        if SPEECH.search(paragraph) and "Gutenberg" not in paragraph:
            out.append(paragraph)
    return "\n".join(out)


ENGLISH = {"wikitalk_en", "stackexchange_en", "opensubtitles_en", "gutenberg_en"}  # read as English: words are counted between spaces, and the Thai-only rules are left off
WHOLE = {"thaifin_th", "prd_th", "pantip_th", "krathu_th", "wikitalk_th", "gazette_th", "forum2_th"} | ENGLISH | set(PARAGRAPHS)  # read as whole paragraphs only, like a listed crawl
FLOORS = {"pantip_th": (20, 30), "krathu_th": (20, 30), "forum2_th": (20, 30), "wikitalk_th": (40, 60), "wikitalk_en": (60, 80), "stackexchange_en": (60, 80), "opensubtitles_en": (12, 16), "gutenberg_en": (40, 60)}  # shortest line and shortest counted row, in characters, where talk is short
THAI_WORD = 4.7  # Thai letters to a word in web prose: newmm of PyThaiNLP 5.1.2 over the first trial slices
# A page that sells gambling, lottery tips, loans or adult services is left out whole when two of these are in it.
SPAM = re.compile(d.GAMBLING.pattern + r"|พนัน|หวย(?!งาน)|เดิมพัน|แจ็คพอต|แจ็กพอต|โบนัส|ฝากถอน|ฝาก-ถอน|ถอนเงิน|เว็บพนัน|ยูสเซอร์|สมัครสมาชิก|โปรโมชั่น|ดูหนังออนไลน์|หนังโป๊|คลิปหลุด|เย็ด|ควย"
                  r"|เงินกู้|สินเชื่อด่วน|รับทำ ?seo|(?i:\bufa|pg ?slot|joker|sbobet|gclub|sagame|betflik|lotto|porn|xxx|viagra|backlink|1xbet|poker|jackpot|bonus|bet\d)")
MENU_MARKS = re.compile(r"[|›»•·▶►→←↑↓★☆✓✔©®™=_#*<>{}\[\]\\]|\.\.\.|…|:\s*$|^[^\w\"“'(]")
ASKS_A_CLICK = re.compile(r"คลิก|อ่านต่อ|อ่านเพิ่มเติม|ดูเพิ่มเติม|แชร์|กดไลค์|กดติดตาม|ติดตามได้ที่|สอบถาม(?:เพิ่มเติม|ได้ที่|รายละเอียด)|ติดต่อ(?:เรา|สอบถาม|ได้ที่)|โทร\.? ?[0-9๐-๙]|โทรสอบถาม|ไลน์ ?(?:ไอดี|@)|(?i:line ?(?:id|@|:))|inbox|เข้าสู่ระบบ|ตะกร้า|สั่งซื้อ|ราคา.{0,12}บาท|คุกกี้|ลิขสิทธิ์|สงวนสิทธิ์|แสดงความคิดเห็น|ตอบกลับ|โพสต์เมื่อ|เขียนโดย|แท็ก|หมวดหมู่|หน้าแรก|เมนู|"
                           r"ดาวน์โหลด|(?i:copyright|cookie|click|login|tags?:|posted)")
KINDS = (("fiction", re.compile(r"gutenberg|writer\.dek-d|my\.dek-d|tunwalai|keedkean|minimore|rakongseries|rakongtvch7|anowl\.co|storylog|thai-novel|readfree|niyay|novel|fictionlog|readawrite|valentinepublication")),
         ("government", re.compile(r"\.go\.th$|\.mi\.th$")), ("education", re.compile(r"\.ac\.th$|\.edu$")), ("organisation", re.compile(r"\.or\.th$|\.org$")),
         ("forum", re.compile("pantip|dek-d|sanook.*webboard|webboard|forum|board|kaskus|reddit|siamphone|thaimtb|soccersuck")),
         ("blog", re.compile("blog|wordpress|medium\\.com|storylog|exteen|oknation|bloggang")),
         ("news", re.compile("news|thairath|matichon|khaosod|dailynews|prachachat|bangkokbiz|mgronline|manager|posttoday|thaipbs|nationtv|sanook|kapook|komchadluek|naewna|siamrath|bbc|voathai|thestandard|workpoint|pptvhd36|ch3|ch7|tnn|springnews|sondhi|isranews|ryt9|infoquest|efinancethai|kaohoon")),
         ("encyclopaedia", re.compile("wikipedia|wiki")), ("shop", re.compile("shop|store|lazada|shopee|kaidee|market|sale|buy")))
# What a kept row must not be. Each pattern was checked against 600 rows labelled by hand (docs/distillation-sources.md, section 6).
MARK = "\u0e31\u0e34-\u0e3a\u0e47-\u0e4e"
OPENS_LEANING = re.compile(r"^[\"“'(]?(ด้วยความที่|ทั้ง ?ๆ ?ที่|กลางคือ|แถม|พร้อมทั้ง|พร้อมกับ|ทั้งใน|ทั้งยัง|อีกทั้ง|ซึ่ง|โดย(?!ทั่วไป|เฉพาะ|ปกติ|สรุป|ส่วนตัว|ส่วนใหญ่|รวม|พื้นฐาน|ธรรมชาติ)|และ|แต่(?!ละ|ก่อน|เดิม|ไหนแต่ไร)|จึง|พบว่า|รวมถึง|รวมไปถึง|ส่งผล|ทำให้|นั่นคือ|นั้นเป็น|อย่างร้าน|แทนที่จะ|หรือ(?!ว่า)|ก่อนที่|ดังจะเห็น|ขึ้นอยู่กับ|เพราะ(?!ฉะนั้น)|เนื่องจาก|ยัง(?!ไง)|ก็|เลย|ถือเป็น|ถือว่า|เป็นการ|กล่าว|ระบุ|เผย|ทวีต|คั่น|ครั้งที่|ประจำ|เหลือ|อาจ(?!ารย์)|จะ|ได้|ต้อง|ไม่)")
ENDS_OPEN = re.compile(r"(?:(?:^| )(?:แม้|หาก|ถ้า|เพราะ|เนื่องจาก|ทว่าเมื่อ|เมื่อ|ขณะที่|ดังนั้น|กล่าวคือ|อย่างไรก็ตาม|นอกเหนือจาก|นอกจาก|ส่วน|แต่เมื่อ|ซึ่งหาก|และเมื่อ|คือ)[^ ]{0,25}(?: [^ ]{0,12})?|(?:บอก|กล่าว|เล่า|ระบุ|เผย|ยืนยัน|ชี้|มอง|เห็น|พบ|คิด|เชื่อ|คาด)ว่า|ว่า|โดยมี|โดย|และ|หรือ|แต่|ซึ่ง|ที่|คือ|เช่น|อาทิ|ได้แก่|ของ|ใน|กับ|แก่|จาก|เพื่อ|ทั้งนี้|นอกจากนี้|พล\.[อทตร]\.(?:[อทต]\.)?|ร\.[อทต]\.|พ\.[อทต]\.[อทต]?\.?|จ\.ส\.[อทต]\.|ส\.[อท]\.|ด\.ต\.|นพ\.|พญ\.|ดร\.|ศ\.|รศ\.|ผศ\.|นาย|นาง|นางสาว|น\.ส\.|ม\.ล\.|ม\.ร\.ว\.|คุณ|มี\.|[A-Za-z]!|,|:|–|-)\s*[\"”')]?$")
LABEL = re.compile(r"^(ความรัก|การเงิน|การงาน|สุขภาพ|ดูดวง|โชคลาภ|การเรียน|No\.?\s*\d+|จุดที่\s*\d+|ข้อ\s*\d+|ข้อที่\s*\d+|\d+[.)])\s")
CAPTION = re.compile(r"\((?:ที่ ?\d+ ?จาก(?:ซ้าย|ขวา)|ซ้าย|ขวา|กลาง|คนซ้าย|คนขวา|ภาพ[^)]{0,20}|แฟ้มภาพ)\)|\(ภาพ ?:|Read more$|สำนักข่าวไทย$")
HANDLE = re.compile(r"@[A-Za-z0-9_.]{3,}|(?i:\b(?:ig|fb|facebook|twitter|instagram|youtube|tiktok)\s*:)|#[^\s#]{2,}")
VERBISH = re.compile("ได้|จะ|เป็น|มี|คือ|ให้|ไม่|แล้ว|อยู่|ว่า|ครับ|ค่ะ|คะ|นะ|ต้อง|ควร|สามารถ|กำลัง|ถูก|ไป|มา")
ORPHAN = re.compile(rf"(?:^|[^\u0e01-\u0e2e{MARK}\u0e33])[{MARK}\u0e33\u0e30\u0e32]|[\u0e40-\u0e44] |[\u0e01-\u0e2e] [{MARK}\u0e33]")
HEADLINE_VERB = re.compile("^[\"“'‘]?(?:ชี้|แนะ|คาด|เผย|เตรียม|มั่นใจ|ท้า|จ่อ|ลุ้น|หวัง|ย้ำ|ปัด|เร่ง|สั่ง|ยัน|เตือน|ขอ|พร้อม)")
CONNECTIVE = re.compile("ซึ่ง|เพราะ|(?:^| )(?:โดย|ที่)|(?<!ก)ว่า")  # ที่ and โดย only where they open a segment: inside one they are in half of all rows
TEASER = re.compile(r"!!|มาดูกัน|กันได้เลย|กันเลย(?:ค่ะ|ครับ|จ้า)?$|ตามคลิป|อย่าลืมเช็ค|ไปดูกัน|ไปชมกัน|เชิญชม$|[\u0e01-\u0e4e] - [\u0e01-\u0e4e]")
THAI_HYPHEN = re.compile(r"[\u0e01-\u0e4e]-[\u0e01-\u0e4e]")
OPENS_POINTING = re.compile("^[\"“'‘]?(?:คำตอบคือ|หมายความว่า|ยิ่งบวกกับ|บวกกับ|มองว่า|นี่เลย|ครับ ?-|ค่ะ ?-|นั่นหมายความว่า|ด้วยเหตุนี้|ดังกล่าว)")
POINTER = re.compile("ขวดนี้|ตัวนี้|รุ่นนี้|แบบนี้$|เดียวกันนี้|ภาพด้านบน|ภาพด้านล่าง|รูปด้านบน|รูปด้านล่าง|ข้างต้น|ดังภาพ|ดังรูป|ตามภาพ|ตามรูป")
SELF = re.compile("(?<![\u0e01-\u0e4e])ตน(?:เอง)?(?![\u0e01-\u0e4e])|ของตน|ที่ตน|ว่าตน")
SPEAKER = re.compile("กล่าว|บอก|เผย|เล่า|ระบุ|ยืนยัน|ให้สัมภาษณ์|ชี้แจง|อ้าง|แถลง|ตอบ|นาย|นาง|น\\.ส\\.|คุณ|พล\\.|ดร\\.")
MISTYPED = re.compile(rf"[\u0e48-\u0e4b]{{2}}|([\u0e31\u0e34-\u0e3a\u0e47\u0e4c])\1|\u0e40\u0e40|\u0e30\u0e30|[\u0e01-\u0e4e] ?[A-Za-z]$|[^ ]{{150,}}")


def loose(text, previous=""):
    """Why a row does not stand as a whole, clean paragraph, or None: a heading, label or caption fused on, an
    account name, garbled or mistyped letters, an opening or an ending inside a sentence, a short line with no
    verb, a headline or teaser, or (with no text before it on the page) words that point at something not shown."""
    if LABEL.search(text) or CAPTION.search(text):
        return "label or caption fused on"
    if HANDLE.search(text):
        return "account name or tag"
    if len(ORPHAN.findall(text)) >= 3:
        return "garbled"
    if OPENS_LEANING.search(text):
        return "opens inside a sentence"
    if ENDS_OPEN.search(text):
        return "stops inside a sentence"
    for a, b in ("“”", "()"):
        if text.count(a) != text.count(b):
            return "stops inside a sentence"
    if text.count('"') % 2:
        return "stops inside a sentence"
    if len(d.THAI.findall(text)) < 120 and len(VERBISH.findall(text)) < 2:
        return "short with no verb"
    if MISTYPED.search(text):
        return "mistyped"
    segments = text.split(" ")
    if (len(segments) >= 3 and sum(1 for part in segments if HEADLINE_VERB.match(part)) >= 2 and not CONNECTIVE.search(text)) or TEASER.search(text) or len(THAI_HYPHEN.findall(text)) >= 2:
        return "headline or teaser"
    if not previous and (OPENS_POINTING.search(text) or POINTER.search(text) or (SELF.search(text) and not SPEAKER.search(text))):
        return "points at something not shown"
    return None


LISTED = pathlib.Path(__file__).with_name("crawl_domains.txt")  # the domain cut, see the head of that file
WHOLE_MIN = 120  # characters: of 300 labelled rows from listed domains, shorter ones were clean 63 times in 100, longer ones 80
LONG_MAX = 700  # characters of a whole paragraph kept as one row marked "long": over today's trainer limit, so counted apart
BUILD = d.OUT / "build"  # the deterministic build: what the jobs of build() write, and merge() and verify() read
INPUTS = pathlib.Path(__file__).with_name("crawl_inputs.tsv")  # every download of the build: job, path under data/distill/raw, address, hash, bytes
LABELS = OUT / "domain_labels.tsv"  # optional: a label for a domain, a sub-source or a whole collection, carried onto its rows at the count
BATCH = 400  # documents a worker takes at a time


USER_LINK = r"'*\[\[(?:ผู้ใช้|คุยกับผู้ใช้|User|User talk|พิเศษ|Special)\s*:"
SIGNED = re.compile(rf"\s*(?:(?:--+|—|–)\s*{USER_LINK}.*|{USER_LINK}[^\]]*\]\][^\u0e01-\u0e2e]*\d{{1,2}}:\d{{2}}, .*|\d{{1,2}}:\d{{2}}, \d{{1,2}} \S+ \d{{4}} \((?:ICT|UTC|\+07)\).*)$")  # a dash before a user link, or a user link or nothing before a time


TAIL_SIGNED = re.compile(r"\s*(?:(?:--+|—|–|~~+)\s*[^.?!]{0,40}|\(?\b(?:talk|contribs|คุย)\)?|[Uu]ser:\S+|ผู้ใช้:\S+)\s*$")


def unsigned(line):
    """A comment without a name typed at its end: a dash and a short name, "(talk)", "user:Name"."""
    for _ in range(3):
        line = TAIL_SIGNED.sub("", line)
    return line.strip()


def talk(text):
    """A wiki discussion page as lines of what people wrote: comments, templates, tables, headings and files go;
    indentation marks, signatures with their names and times, and link and emphasis marks are stripped."""
    text = re.sub(r"<!--.*?-->|<ref[^>]*>.*?</ref>|<[^>]+>", " ", text, flags=re.S)
    for _ in range(4):
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    lines = []
    for line in text.split("\n"):
        line = SIGNED.sub("", line.lstrip(":*#; ").strip())
        if not line or line[0] in "=|!{" or "{{" in line or "}}" in line:
            continue
        line = re.sub(r"\[\[(?:ไฟล์|ภาพ|File|Image|หมวดหมู่|Category)\s*:[^\]]*\]\]", " ", line)
        line = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", line)
        line = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", line)
        lines.append(unsigned(re.sub(r"'{2,}", "", line)))
    return "\n".join(lines)


# English discussion pages: what a kept comment must not carry. Checked against 200 rows labelled by hand.
VOTE = re.compile(r"^(?:(?:Strong(?:ly)?|Weak|Speedy|Very|Conditional|Procedural|Mild) )*(?:Delete|Keep|Support|Oppose|Comment|Merge|Redirect|Neutral|Agree|Disagree|Endorse|Object|Abstain|Reply|Question|"
                  r"Answer|Note|Remove|Rename|Move|Userfy|Transwiki|Relist|Overturn|Close|Withdraw|Approve|Reject|Decline|Done|Not done|Fixed|Resolved|Update|Response)"
                  r"(?: (?:and|or|/) (?:delete|keep|merge|redirect|rename|move|close))?(?:\s*[-–—:.,;!]+\s*|\s+(?=[\"“'(]?[A-Z]))")
VOTE_ONLY = re.compile(r"^(?:(?:Strong(?:ly)?|Weak|Speedy) )*(?:Delete|Keep|Support|Oppose|Merge|Redirect|Neutral|Comment|Endorse|Userfy|Transwiki|Relist) (?:as|per|unless|because|since|but|for now|if|all|both|this|it|the)\b")
CANNED = re.compile(r"is the most vandalized page|Today's featured article is|has been contacted for more input|An editor has asked for a discussion|[Pp]lease participate in the|featured pictures were displayed|"
                    r"reference desk works like|Welcome to Wikipedia|[Tt]hank you for (?:experimenting|your contributions)|(?:nominated|listed|proposed|tagged|considered) for (?:speedy )?(?:deletion|discussion|merging)|"
                    r"If you (?:think|feel|believe) (?:that )?this (?:notice|page|message|file|image)|edits? (?:to|on) .{0,80}(?:has|have) been (?:reverted|undone|removed)|\b(?!Abbot|Talbot|Robot)[A-Z]\w*[Bb]ot\b|\bbots?\b|automated (?:message|process|notice)|"
                    r"do not (?:edit|modify) (?:it|this)|discussion is (?:closed|preserved|an archive)|[Ss]ubsequent comments should be made|No further edits should be made|is admonished|permissions-en|four tildes|"
                    r"sign your (?:posts|comments|name)|[Tt]his (?:message|newsletter) (?:was|is) (?:delivered|sent)|conflict of interest guideline|criteria for speedy deletion|copyright status|fair use rationale|"
                    r"[Yy]ou (?:have been|are) (?:temporarily )?blocked|may be blocked from editing|[Ii]f you (?:continue to|vandali[sz]e)|the sandbox|your (?:first|recent) (?:article|contribution)")
SHORTCUT = re.compile(r"\b(?:WP|MOS|WT|CAT|MOSNUM|NPOV|BLP|AfD|RfC|RfA|CSD|AFD|RFC|RFA|TfD|CfD|FAC|GAN|DYK|ANI|AN/I|3RR|COI|POV|OR|RS)\b[:/]?|\b(?:Wikipedia|Wikipedia talk|Template|Category|User|User talk|Special|Help|Portal|File|Image|Talk):\w")
HOLE_EN = re.compile(r"\ban (?![aeiouAEIOU8]|h(?:our|on|eir|ist)|[FHLMNRSXfhlmnrsx][A-Z0-9-]|[A-Z]{2}|nth|[flmnrsx]\b|[^\x00-\x7f]|1[18]|\"|'|\()\w|\b(?:[Ss]ee|[Ff]ound|[Aa]vailable|[Rr]ead|[Vv]iew(?:ed)?|[Ll]ook|[Cc]lick) here(?:,? (?:and|or) here)*\s*[.,;:)!?]|"
                     r"\b(?:[Ss]ee|[Uu]se|using|[Tt]ry) the (?:template|tag)\s*[.,;:)]|\b(?:the|a|an|this) (?:template|tag) (?:and|or) (?:the|a|an) (?:template|tag)\b|\(\s*\)|\"\s*\"")
RESIDUE = re.compile(r"\s*[—–-]*\s*(?:The )?[Pp]receding .{0,40}comment.*$|\s*[—–-]*\s*\S*\s*\(?\d{1,2}:\d{2}, \d{1,2} \w+ \d{4}.{0,12}$", re.S)
RESIDUE_LEFT = re.compile(r"(?i)unsigned comment|comment (?:was )?added by|\(UTC\)|\btalk\s*[•·|]?\s*contribs\b|~~~|\bcontribs\b|\d{1,2}:\d{2}, \d{1,2} \w+ \d{4}")


REPEATS = 5  # rows that open alike before a row is taken for a canned message


def opening(text):
    """How a row begins, for counting rows that begin alike: its first 50 letters, digits and case aside."""
    return re.sub(r"[^a-z0]", "", re.sub(r"\d", "0", text.lower()))[:50]


def discussion_en(text):
    """(text, None) for an English discussion comment that stands, or (text, why not). A vote or comment word fused
    to the front goes when a whole sentence is left; a signature left at the end goes; a canned or automated
    message, a comment that leans on policy shortcuts, and one with a hole where a template or link stood, are left out."""
    text = RESIDUE.sub("", text).strip()
    if RESIDUE_LEFT.search(text):
        return text, "signature left in the text"
    if CANNED.search(text):
        return text, "canned or automated message"
    if VOTE_ONLY.match(text):
        return text, "a vote without a sentence"
    rest = VOTE.sub("", text, count=1)
    if rest != text:
        if not WHOLE_EN.match(rest) or len(rest.split()) < 8:
            return text, "a vote without a sentence"
        text = rest
    if SHORTCUT.search(text):
        return text, "policy shortcut"
    if HOLE_EN.search(text):
        return text, "a template or link is missing"
    return text, None


WHOLE_EN = re.compile(r"^[\"“'(]?[A-Z0-9].*[.?!][\"”')]?$", re.S)
MARKUP = re.compile(r"\[\[|\]\]|\{\{|\}\}|https?://|www\.|<|>|\|")


def weight(text, english=False):
    """What a text weighs in the counts: its Thai letters, or for English its words scaled so that one word weighs
    as much as one Thai word (THAI_WORD letters)."""
    return len(text.split()) * 4.7 if english else len(d.THAI.findall(text))


def loose_en(text, previous=""):
    """Why an English row does not stand as a whole paragraph, or None: it does not open on a capital and close on
    a full stop, question or exclamation mark; it carries wiki or web markup; or it has under eight words."""
    if MARKUP.search(text):
        return "markup or an address"
    if not WHOLE_EN.match(text):
        return "opens inside a sentence" if not re.match(r"[\"“'(]?[A-Z0-9]", text) else "stops inside a sentence"
    if len(text.split()) < (4 if previous is not None and len(text) < 60 else 8):  # a reply to something said may be short
        return "short with no verb"
    return None


def domain_of(url):
    host = (urllib.parse.urlsplit(url).hostname or "") if url else ""
    return host[4:] if host.startswith("www.") else host


def kind_of(domain):
    """What sort of site a domain is, by its name alone: enough to sort by, not to rely on."""
    return next((kind for kind, pattern in KINDS if pattern.search(domain)), "other" if domain else "unknown")


def paragraphs(text, floor=60, english=False):
    """The paragraphs of a page that read as prose. Left out: lines under 60 characters (headings, captions,
    list items), menus and link lists (many short chunks between spaces, or menu marks), and lines that ask
    for a click, a call or a purchase."""
    for line in text.split("\n"):
        line = d.clean(line)
        if len(line) < floor:
            yield None, "short line"
        elif (not english and len(line) / (line.count(" ") + 1) < 12) or MENU_MARKS.search(line):
            yield None, "menu or list"
        elif ASKS_A_CLICK.search(line):
            yield None, "asks for a click or a purchase"
        else:
            yield line, None


def page_rows(prefix, text, counts, cut=True, before=None, floors=(60, WHOLE_MIN), ceiling=None, english=False):
    """(id, text, "th", row before) for one page. A paragraph that fits a row goes in whole; a longer one is
    cut into rows of whole sentences by the builder's thai_cut. A row carries the row before it from the
    same page as context."""
    n, previous = 0, before
    for paragraph, reason in paragraphs(text, floors[0], english):
        if reason:
            counts[reason] += 1
            previous = None  # what was left out stood between the two
            continue
        whole = len(paragraph) <= (ceiling or d.CUT_CHARS)
        if not cut and not floors[1] <= len(paragraph) <= (ceiling or d.CUT_CHARS):
            counts["longer than a row" if not whole else "shorter than a counted row"] += 1
            previous = None
            continue
        for piece in ([paragraph] if whole else d.thai_cut(paragraph)):
            reason = (loose_en if english else loose)(piece, previous)
            if reason:
                counts[reason] += 1
                previous = None
                continue
            yield f"{prefix}-{'p' if whole else 'c'}{n}", piece, "en" if english else "th", previous  # p: a whole paragraph; c: cut from a longer one
            previous, n = piece, n + 1


_state = {}


def known_share(text):
    """The share of a row's Thai letters that fall in dictionary words of two letters or more, under newmm."""
    if "words" not in _state:
        from pythainlp.corpus import thai_words
        from pythainlp.tokenize import word_tokenize
        _state["words"], _state["tokenize"] = thai_words(), word_tokenize
    known = total = 0
    for token in _state["tokenize"](text, engine="newmm"):
        letters = len(d.THAI.findall(token))
        total += letters
        known += letters if len(token) > 1 and token in _state["words"] else 0
    return round(known / total, 2) if total else 0.0


def work(job):
    """One batch of pages through the page rule, the paragraph rules, the cutter and the builder's filters."""
    name, pages, listed, long = job
    english = name in ENGLISH
    if listed and "listed" not in _state:
        _state["listed"] = {line for line in LISTED.read_text(encoding="utf8").splitlines() if line and not line.startswith("#")}
    source = dataclasses.replace(COLLECTIONS[name], max_chars=LONG_MAX, max_pieces=10 ** 6) if long else COLLECTIONS[name]
    if "encode" not in _state:
        _state["encode"] = __import__("train.tokenizer", fromlist=["Tokenizer"]).Tokenizer().encode
        _state["blocked"], _state["benchmark"] = d.project_text(), d.benchmark_text()
    counts, domains, batch, where, letters_in, page_letters = collections.Counter(), collections.Counter(), [], {}, 0, {}
    for doc_id, url, text, *before in pages:
        domain = domain_of(url)
        letters = weight(text, english)
        letters_in += letters
        domains[domain, "in"] += letters
        if listed and domain not in _state["listed"]:
            counts["page: domain not on the list"] += 1
            continue
        if len(SPAM.findall(text)) >= 2 or SPAM.search(url or ""):
            counts["page: gambling, adult or spam"] += 1
            continue
        if not english and letters < 0.5 * len(text.replace(" ", "")):
            counts["page: not mostly Thai"] += 1
            continue
        if name in SPOKEN:  # one utterance: the file is its page
            page = f"{name}-{doc_id}".rsplit("-", 1)[0]
            page_letters[page] = page_letters.get(page, 0) + letters
            batch.append((f"{name}-{doc_id}", text, "th", before[0] if before else None))
            where[f"{name}-{doc_id}"] = domain
            continue
        page_letters[f"{name}-{doc_id}"] = letters
        for row in page_rows(f"{name}-{doc_id}", text, counts, cut=not (listed or name in WHOLE), before=before[0] if before else None, floors=FLOORS.get(name, (60, WHOLE_MIN)), ceiling=LONG_MAX if long else None, english=english):
            batch.append(row)
            where[row[0]] = domain
    rows = d.sift(source, batch, lambda texts: map(len, _state["encode"](texts)), _state["blocked"], _state["benchmark"], set(), collections.Counter(), counts)
    kept_of_page = collections.Counter()
    for r in rows:
        kept_of_page[r["id"].rsplit("-", 1)[0]] += weight(r["text"], english)
    for r in rows:
        if len(r["text"]) > d.CUT_CHARS:
            r["long"] = True  # and it keeps its piece count
        else:
            del r["pieces"]
        page = r["id"].rsplit("-", 1)[0]
        # statistics to sort by later: dictionary words, and how much of the page read as prose (an article keeps most of itself, a listing little)
        r |= {"collection": name, "domain": where[r["id"]], "known": None if english else known_share(r["text"]), "page_kept": round(kept_of_page[page] / max(1, page_letters[page]), 2)}
        domains[r["domain"], "kept"] += weight(r["text"], english)
    return rows, counts, domains, len(pages), letters_in


def lines_of(handle):
    for line in handle:
        try:
            yield json.loads(line)
        except json.JSONDecodeError:  # the first or last line of a slice cut at a byte, or the end of a part file
            continue


def pages_of(name, where, start, size):
    """(page id, address, text) for a slice: a .zst of HPLT lines, a FineWeb2 parquet file, or a byte range of a Mangosteen JSON-lines address."""
    if name == "hplt_th":
        import zstandard
        with open(where, "rb") as file:
            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(file), encoding="utf8", errors="replace")
            try:
                for page in lines_of(reader):
                    if page.get("filter", "keep") == "keep":
                        yield page["id"], page.get("u"), page["text"]
            except zstandard.ZstdError:  # a part of the file: it ends mid-frame
                return
    elif name in ("fineweb2_th", "forum2_th"):
        import pyarrow.parquet as pq
        for group in pq.ParquetFile(where).iter_batches(batch_size=2000, columns=["id", "url", "text"]):
            yield from (page for page in zip(*(group.column(c).to_pylist() for c in ("id", "url", "text"))) if name == "fineweb2_th" or domain_of(page[1]) in FORUMS)
    elif name == "pantip_th":  # one line a reply, its topic's post repeated beside it: the post once, then each reply with the post's opening before it
        import csv
        csv.field_size_limit(10 ** 9)
        with open(where, encoding="utf8", newline="") as file:
            posted = set()
            for n, topic, post, reply in itertools.islice(csv.reader(file), 1, None):
                if topic not in posted:
                    posted.add(topic)
                    yield f"t{topic}", "//pantip.com", post
                yield f"r{n}", "//pantip.com", reply, d.clean(post.split("\n")[0])[:d.CUT_CHARS] or None
    elif name == "krathu_th":  # the folder of the repository: a comment carries its post's title before it
        import csv
        csv.field_size_limit(10 ** 9)
        with open(pathlib.Path(where) / "posts.csv", encoding="utf8", newline="") as file:
            titles = {r["post_id"]: d.clean(r["title"])[:d.CUT_CHARS] for r in csv.DictReader(file)}
        with open(pathlib.Path(where) / "comments.csv", encoding="utf8", newline="") as file:
            for r in csv.DictReader(file):
                yield r["comment_id"], "//pantip.com", r["comment_body"] or "", titles.get(r["comment_id"].split("-")[0]) or None
    elif name == "gazette_th":  # a month's JSON-lines file: documents whose text came from the PDF's text layer and passed the set's own check
        with open(where, encoding="utf8") as file:
            for page in lines_of(file):
                if page.get("method") == "direct" and page.get("is_valid"):
                    yield page["doc_id"], "//ratchakitcha.soc.go.th", unwrapped(page.get("text") or "")
    elif name == "gutenberg_en":  # a folder of <book number>.txt
        for file in sorted(pathlib.Path(where).glob("*.txt")):
            yield file.stem, "//gutenberg.org", book_paragraphs(file.read_bytes().decode("utf8", "replace"))
    elif name in SPOKEN:  # the zip of raw subtitle files; the id keeps the film's and the file's number
        for row in d.opensubtitles(pathlib.Path(where).resolve()):
            yield row[0].split("-", 1)[1], "//opensubtitles.org", row[1], row[3]
    elif name == "opensubtitles_en":
        for row in english_subtitles(where):
            yield row[0], "//opensubtitles.org", row[1], row[3]
    elif name in PARAGRAPHS:  # one paragraph of the builder's reader a time, with the text it gives as context
        for row in PARAGRAPHS[name][1]():  # (id, text, language) and, from some readers, the text before
            yield row[0], "//" + PARAGRAPHS[name][0], row[1], (row[3] if len(row) > 3 else None) or None
    elif name == "stackexchange_en":  # a folder of <site>.stackexchange.com.7z; an answer carries its question's title before it
        import tempfile
        import xml.etree.ElementTree as ET
        import py7zr
        for archive in sorted(pathlib.Path(where).glob("*.7z")):
            site = archive.name[:-3]
            with tempfile.TemporaryDirectory(dir=archive.parent) as folder:
                with py7zr.SevenZipFile(archive) as packed:
                    packed.extract(path=folder, targets=["Posts.xml"])
                titles = {}
                for _, post in ET.iterparse(pathlib.Path(folder) / "Posts.xml"):
                    if post.tag == "row":
                        body = re.sub(r"<(pre|code|blockquote)\b.*?</\1>", " ", post.get("Body") or "", flags=re.S)
                        body = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"</p>|<br ?/?>|</li>|</h\d>", "\n", body)))
                        if post.get("PostTypeId") == "1":
                            titles[post.get("Id")] = d.clean(post.get("Title") or "")[:d.CUT_CHARS]
                            yield f"{site.split('.')[0]}-{post.get('Id')}", "//" + site, body, titles[post.get("Id")] or None
                        elif post.get("PostTypeId") == "2":
                            yield f"{site.split('.')[0]}-{post.get('Id')}", "//" + site, body, titles.get(post.get("ParentId"))
                        post.clear()
    elif name in ("wikitalk_th", "wikitalk_en"):  # a pages-meta-current dump: talk (1), project (4) and project talk (5), and user talk (3) over 3 KB, which leaves out the welcome templates
        import bz2
        import xml.etree.ElementTree as ET
        with bz2.open(where, "rb") as handle:
            for _, page in ET.iterparse(handle):
                if page.tag.endswith("}page"):
                    ns = page.tag[:-4]
                    space, text = page.findtext(f"{ns}ns"), page.findtext(f"{ns}revision/{ns}text") or ""
                    if space in ("1", "4", "5") or (space == "3" and len(text.encode()) > 3000):
                        yield page.findtext(f"{ns}id"), "//en.wikipedia.org" if name in ENGLISH else "//th.wikipedia.org", talk(text)
                    page.clear()
    elif name in WHOLE:  # a folder of parquet files; "//name" gives domain_of the sub-source or department
        import pyarrow.parquet as pq
        for file in sorted(pathlib.Path(where).glob("*.parquet")):
            for n, r in enumerate(pq.read_table(file).to_pylist()):
                if name == "thaifin_th":
                    yield f"{file.stem[6:11]}-{n}", "//" + r["meta"]["subset"].removeprefix("dataset."), r["text"] or ""
                else:
                    # an article comes as one line: sara am typed as two marks, hashtags at the end, two spaces where a paragraph ended
                    text = re.sub(r"\s*#\S+", "", (r["Detail"] or "").replace("\u0e4d\u0e32", "\u0e33"))
                    yield f"{r['__index_level_0__']}", "//" + re.sub(r"\s+", "-", r["Department"] or "none"), re.sub(r" {2,}", "\n", text)
    else:
        request = urllib.request.Request(where, headers={"Range": f"bytes={start}-{start + size - 1}", "User-Agent": "corpus-builder/1.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            for page in lines_of(io.TextIOWrapper(response, encoding="utf8", errors="replace")):
                yield f"{page['id']}", None, page["text"]


def content_key(row):
    """The key a row's scores are carried over by: blake2b, 8 bytes in hex, of the row's page id (its id without the
    last "-" part), its text and the last 200 characters of its context (empty when it has none), joined by U+001F."""
    return hashlib.blake2b("\x1f".join((row["id"].rsplit("-", 1)[0], row["text"], (row.get("context") or "")[-200:])).encode(), digest_size=8).hexdigest()


def collect(name, slice_name, where, start=0, size=0, workers=8, limit=None, listed=False, long=False, again=False, clean=False):
    """Read a slice into data/distill/crawl. With clean, into data/distill/build instead, as one job of the
    deterministic build: nothing is carried from any other read (duplicates across slices are left to merge()),
    the count rules of tally() are applied here, and every row gets its content key and kind of page. The rows
    go to <slice>.jsonl.gz, and id, key, mark and words of each row to <slice>.marks.tsv.gz for the merge."""
    out = (BUILD if clean else OUT) / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{slice_name}.json").exists():
        return json.loads((out / f"{slice_name}.json").read_text(encoding="utf8"))
    seen_file = OUT / "seen.bin"  # the mark of every row kept so far, in any collection, so a later slice does not repeat it
    seen = array.array("Q")
    if seen_file.exists() and not clean:
        seen.frombytes(seen_file.read_bytes())
    known, fresh = set() if again or clean or name == "forum2_th" else set(seen), array.array("Q")  # again: a slice lost and read once more, its marks already in the file
    counts, domains, pages, letters_in, letters_kept, rows_kept, started = collections.Counter(), collections.Counter(), 0, 0, 0, 0, time.perf_counter()
    held = (set() if clean else {d.digest(d.template(r["text"])) for r in d.jsonl(d.OUT / "sources.jsonl") if r["source"] == "opensubtitles_th"} if name in SPOKEN
            else Marks(seen) if name == "forum2_th" else set())
    words_kept, checksum = 0.0, hashlib.sha256()
    marks = gzip.open(out / f"{slice_name}.marks.tsv.gz", "wt", encoding="utf8", compresslevel=4) if clean else None
    jobs = ((name, batch, listed, long) for batch in itertools.batched(itertools.islice(pages_of(name, where, start, size), limit), BATCH))
    part = out / (f"{slice_name}.jsonl.gz.part" if clean else f"{slice_name}.jsonl.part")
    with multiprocessing.Pool(workers) as pool, (gzip.open(part, "wt", encoding="utf8", compresslevel=4) if clean else open(part, "w", encoding="utf8")) as file:
        for rows, counted, by_domain, n, letters in pool.imap(work, jobs):
            counts += counted
            pages += n
            letters_in += letters
            for (domain, side), value in by_domain.items():
                domains[domain, side] += value if side == "in" else 0
            for r in rows:
                mark = d.digest(d.template(r["text"]))
                if mark in known or mark in held:
                    counts["duplicate" if mark in known else HELD_AS[name]] += 1
                    if mark not in known:
                        counts["words " + HELD_AS[name]] += round(weight(r["text"]) / THAI_WORD)
                    continue
                if clean:
                    reason = uncounted(name, r)
                    if reason:
                        counts[reason] += 1
                        continue
                    r |= {"key": content_key(r), "kind": kind_of(r["domain"])}
                    words_kept += d.words(r)
                    checksum.update(f"{r['id']}\t{r['text']}\n".encode())
                    marks.write(f"{r['id']}\t{r['key']}\t{mark:016x}\t{d.words(r):.2f}\n")
                known.add(mark)
                fresh.append(mark)
                kept = weight(r["text"], name in ENGLISH)
                letters_kept += kept
                rows_kept += 1
                domains[r["domain"], "kept"] += kept
                file.write(json.dumps(r, ensure_ascii=False) + "\n")
    part.rename(part.with_suffix(""))
    if clean:
        marks.close()
    if name in SPOKEN:  # what the subtitle reader saw of each film and file, for the checks by film
        (out / f"{slice_name}.films.json").write_text(json.dumps({"lines_joined_mixed_by_film": d.FILM_LINES, "thai_particle_possessive_by_file": d.FILE_SIGNS, "reader": d.SUBTITLE_DROPS}) + "\n", encoding="utf8")
    if not again and not clean:
        with open(seen_file, "ab") as file:
            file.write(fresh.tobytes())
    by_domain = {dom: [round(domains[dom, "in"] / THAI_WORD), round(domains[dom, "kept"] / THAI_WORD)] for dom in {dom for dom, _ in domains}}
    by_kind = collections.Counter()
    for dom, (_, kept) in by_domain.items():
        by_kind[kind_of(dom)] += kept
    record = {"collection": name, "slice": slice_name, "from": str(where), "start": start, "bytes": size, "listed_domains_only": listed, "long_paragraphs": long, "pages": pages, "words_in": round(letters_in / THAI_WORD),
              "rows": rows_kept, "words_kept": round(letters_kept / THAI_WORD), "seconds": round(time.perf_counter() - started),
              "removed": dict(counts.most_common()), "kept_words_by_kind_of_site": dict(by_kind.most_common()),
              "domains": dict(sorted(by_domain.items(), key=lambda item: -item[1][1]))}
    if clean:  # what verify() checks: the rows, their words by the builder's count, and a hash of every id and text in order
        record |= {"words": round(words_kept), "sha256": checksum.hexdigest()}
    (out / f"{slice_name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    return record


def uncounted(name, row):
    """Why a row that passed the filters is still not counted, or None: the rules tally() applies to a slice."""
    if name in SPOKEN:
        return None
    if row["id"].rsplit("-", 1)[1][0] != "p":
        return "cut from a longer paragraph"
    if len(row["text"]) < FLOORS.get(name, (60, WHOLE_MIN))[1]:
        return "shorter than a counted row"
    return (loose_en if name in ENGLISH else loose)(row["text"], row.get("context"))


def tally(name):
    """Rewrite a collection's slices (trials aside) to what counts: whole paragraphs of listed domains that pass
    loose(). Thai words of each slice before and after go to words.json beside them; a slice already rewritten
    keeps the first count taken of it."""
    listed = {line for line in LISTED.read_text(encoding="utf8").splitlines() if line and not line.startswith("#")}
    out = OUT / name
    earlier = json.loads((out / "words.json").read_text(encoding="utf8")).get("slices", {}) if (out / "words.json").exists() else {}
    slices, removed, by_kind, with_previous, long_letters = {}, collections.Counter(), collections.Counter(), 0, collections.Counter()
    field, labels, by_label = None, {}, collections.Counter()
    if LABELS.exists():  # tab-separated: a header line naming the field, then one line a domain
        lines = [line.split("\t") for line in LABELS.read_text(encoding="utf8").splitlines() if line and not line.startswith("#")]
        field, labels = lines[0][1], {line[0]: line[1] for line in lines[1:]}
    openings = collections.Counter()  # a canned message opens alike on many pages
    if name == "wikitalk_en":
        for file in sorted(out.glob("*.jsonl")):
            openings.update(d.digest(opening(json.loads(line)["text"])) for line in open(file, encoding="utf8"))
    for file in sorted(out.glob("*.jsonl")):
        if file.stem.startswith("trial"):
            continue
        before = after = 0
        with open(file, encoding="utf8") as rows, open(file.with_suffix(".tmp"), "w", encoding="utf8") as kept:
            for line in rows:
                r = json.loads(line)
                if name.startswith("wikitalk"):
                    r["text"] = unsigned(r["text"])
                n = weight(r["text"], name in ENGLISH)
                before += n
                canned = None
                if name == "wikitalk_en":
                    canned = "opens like many other rows" if openings[d.digest(opening(r["text"]))] >= REPEATS else None
                    r["text"], canned = discussion_en(r["text"]) if not canned else (r["text"], canned)
                reason = canned or ("domain not on the list" if name not in WHOLE and r["domain"] not in listed else  # the list is for the crawls
                          "cut from a longer paragraph" if r["id"].rsplit("-", 1)[1][0] != "p" else
                          "shorter than a counted row" if len(r["text"]) < FLOORS.get(name, (60, WHOLE_MIN))[1] else (loose_en if name in ENGLISH else loose)(r["text"], r.get("context")))
                if reason:
                    removed[reason] += n
                    continue
                after += n
                if field:
                    site = r["domain"]  # a sub-source of the financial set is named without its ending
                    r[field] = labels.get(site) or labels.get(site + ".com") or labels.get(site + ".net") or labels.get(name) or "none"
                    by_label[r[field], kind_of(site)] += n
                long_letters[500 if len(r["text"]) <= 500 else LONG_MAX] += n if len(r["text"]) > d.CUT_CHARS else 0
                by_kind[kind_of(r["domain"])] += n
                with_previous += n if r.get("context") else 0
                kept.write(json.dumps(r, ensure_ascii=False) + "\n")
        file.with_suffix(".tmp").replace(file)
        was = earlier.get(file.stem, {}).get("collected") if isinstance(earlier, dict) else None
        slices[file.stem] = {"collected": was or round(before / THAI_WORD), "counted": round(after / THAI_WORD)}
    done = {"collection": name, "slices": slices, "collected": sum(s["collected"] for s in slices.values()),
            "counted": sum(s["counted"] for s in slices.values()),
            "counted_in_long_rows": {f"up to {top} characters": round(n / THAI_WORD) for top, n in sorted(long_letters.items())},
            "counted_with_text_before_them_on_the_page": round(with_previous / THAI_WORD),
            "counted_by_label": {c: {k: round(n / THAI_WORD) for (c2, k), n in by_label.most_common() if c2 == c} for c in sorted({c for c, _ in by_label})},
            "counted_by_kind_of_site": {key: round(value / THAI_WORD) for key, value in by_kind.most_common()},
            "words_removed_this_run": {key: round(value / THAI_WORD) for key, value in removed.most_common()}}
    (out / "words.json").write_text(json.dumps(done, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    return done


# A collection larger than its share of the English is reduced to (words, most from one group). Stack Exchange is
# grouped by site, and a site of everyday talk fills at four times the rate of the rest; discussion pages by page.
CAPS = {"stackexchange_en": (100_000_000, 8_000_000), "wikitalk_en": (100_000_000, None)}
EVERYDAY = {s + ".stackexchange.com" for s in "workplace travel parenting cooking interpersonal money english ell diy".split()}
SENTENCES = re.compile(r"[.?!][\"”')]*(?:\s|$)")


def quotas(have, target, ceiling=None, weights={}):
    """Words to take from each group of {group: words} so that they sum to target: every group is filled to the
    same level times its weight, or to what it has, or to the ceiling, whichever is least."""
    def at(level):
        return {g: min(n, ceiling or n, level * weights.get(g, 1)) for g, n in have.items()}
    low, high = 0, max(have.values(), default=0)
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if sum(at(mid).values()) < target else (low, mid)
    return at(high)


def cap(name):
    """Mark the rows of a collection that fit under its cap with `use`, and write the counts to capped.json. Within
    a group a row of two sentences or more is taken before a single sentence; among equals the choice is by hash
    of the row id, so it does not lean to the early posts."""
    target, ceiling = CAPS[name]
    group = (lambda r: r["domain"]) if name == "stackexchange_en" else (lambda r: r["id"].rsplit("-", 1)[0])
    full = lambda r: len(SENTENCES.findall(r["text"])) >= 2
    files, have = sorted(f for f in (OUT / name).glob("*.jsonl") if not f.stem.startswith("trial")), collections.defaultdict(lambda: [0, 0])
    for file in files:
        for r in d.jsonl(file):
            have[group(r)][full(r)] += len(r["text"].split())
    quota = quotas({g: a + b for g, (a, b) in have.items()}, target, ceiling, dict.fromkeys(EVERYDAY, 4))
    share = {g: (max(0, quota[g] - b) / a if a else 0, min(1, quota[g] / b) if b else 0) for g, (a, b) in have.items()}
    used = collections.Counter()
    for file in files:
        with open(file.with_suffix(".tmp"), "w", encoding="utf8") as out:
            for r in d.jsonl(file):
                r.pop("use", None)
                if d.digest(r["id"]) / 2 ** 64 < share[group(r)][full(r)]:
                    r["use"] = True
                    used[r["domain"] if name == "stackexchange_en" else "all", full(r)] += len(r["text"].split())
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        file.with_suffix(".tmp").replace(file)
    done = {"collection": name, "words": sum(a + b for a, b in have.values()), "groups": len(have), "words_used": sum(used.values()),
            "words_used_in_rows_of_two_sentences_or_more": sum(n for (_, f), n in used.items() if f),
            "words_used_by_group": {g: n for g, n in sorted(collections.Counter({g: used[g, False] + used[g, True] for g, _ in used}).items(), key=lambda x: -x[1])}}
    (OUT / name / "capped.json").write_text(json.dumps(done, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    return done


# ---------------------------------------------------------------- the deterministic build
# The same rows from the same bytes: every download is pinned in crawl_inputs.tsv, every job reads its own input
# and nothing else, and what repeats across jobs is settled once, in merge(), in a fixed order. A job is small
# enough for a hosted runner; `jobs` lists them for a matrix.
BULKY = ("fineweb2-", "opensubtitles", "stackexchange")  # jobs whose download is deleted once read
ORDER = ("prd_th thwiki_p thwikisource_p court_p thailaw_p blognone_p voathai_p gutenberg_en stackexchange_en "  # tier 1 first, so it keeps a row tier 2 repeats
         "pantip_th krathu_th thaifin_th fineweb2_th forum2_th opensubtitles_en opensubtitles24_th").split()


def inputs(job=None):
    """(job, path, address, hash, bytes) of every pinned download, or of one job's."""
    lines = (line.split("\t") for line in INPUTS.read_text(encoding="utf8").splitlines() if line and not line.startswith("#"))
    return [(j, d.RAW / path, url, digest, int(size)) for j, path, url, digest, size, *_ in lines if job in (None, j)]


def jobs():
    return sorted({job for job, *_ in inputs()} | {"paragraphs"})


def reads(job):
    """(collection, slice, where, listed) for each read a job makes."""
    if job.startswith("fineweb2-"):
        return [("fineweb2_th", job[9:], d.RAW / "fineweb2" / f"{job[9:]}.parquet", True), ("forum2_th", job[9:], d.RAW / "fineweb2" / f"{job[9:]}.parquet", False)]
    if job == "paragraphs":  # the pool's own downloads, read again as whole paragraphs: `python -m train.distill_sources fetch` gets them
        return [(name, "all", "-", False) for name in sorted(PARAGRAPHS)]
    name, where = {"thaifin": ("thaifin_th", "thaifin"), "prd": ("prd_th", "prd"), "pantip": ("pantip_th", "pantip/Pantip_QA_200000_20220220.csv"), "krathu": ("krathu_th", "krathu/krathu-500"),
                   "opensubtitles_en": ("opensubtitles_en", "opensubtitles/en.zip"), "opensubtitles24": ("opensubtitles24_th", "opensubtitles24/th.zip"),
                   "stackexchange": ("stackexchange_en", "stackexchange"), "gutenberg": ("gutenberg_en", "gutenberg")}[job]
    return [(name, "all", d.RAW / where, False)]


def get(url, start, end, tries=40):
    """Bytes start to end of an address; a broken connection is tried again."""
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "corpus-builder/1.0", "Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
            if response.status == 206 and len(body) == end - start:
                return body
        except OSError as error:
            print(f"{url} {start}-{end}: {error}; trying again", file=sys.stderr)
        time.sleep(15)
    raise OSError(f"{url}: no answer for bytes {start}-{end}")


def fetch_part(path, url, digest, size):
    """Fetch of an archive only the files read from it. The address ends in #index=<where the archive's index
    starts>; the index is fetched first and checked against the pin, then the files subtitle_files() names, into
    a file with holes of the archive's full size. The index holds a checksum of every file, which the reader checks."""
    import zipfile
    from multiprocessing.pool import ThreadPool
    if path.exists():
        return
    url, index = url.split("#index=")[0], int(url.split("#index=")[1])
    part = path.with_name(path.name + ".part")
    file = os.open(part, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(file, size)
    tail = get(url, index, size)
    if "sha256:" + hashlib.sha256(tail).hexdigest() != digest:
        raise SystemExit(f"{path}: the index is not the one in the pin table")
    os.pwrite(file, tail, index)
    with zipfile.ZipFile(part) as archive:
        spans = sorted((i.header_offset, min(index, i.header_offset + 30 + len(i.filename.encode()) + 256 + i.compress_size)) for i in map(archive.getinfo, subtitle_files(archive)))
    runs = []
    for start, end in spans:  # files that lie close are fetched in one request
        if runs and start - runs[-1][1] <= 65536:
            runs[-1][1] = max(end, runs[-1][1])
        else:
            runs.append([start, end])
    with ThreadPool(24) as pool:  # some 29,000 small requests, a second each
        for start, body in pool.imap_unordered(lambda run: (run[0], get(url, *run)), runs):
            os.pwrite(file, body, start)
    os.close(file)
    part.rename(path)


def fetch_file(pin):
    """Download one file if it is not yet on disk, and check its size and hash. Returns what is wrong, or None."""
    _, path, url, digest, size = pin
    path.parent.mkdir(parents=True, exist_ok=True)
    if "#index=" in url:
        return fetch_part(path, url, digest, size)
    for attempt in range(40):  # a large file is resumed where a broken connection left it
        have = path.stat().st_size if path.exists() else 0
        if have >= size:
            break
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "corpus-builder/1.0", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(request, timeout=120) as response, open(path, "ab" if response.status == 206 else "wb") as file:
                shutil.copyfileobj(response, file, 1 << 20)
        except OSError as error:
            print(f"{path.name}: {error}; trying again", file=sys.stderr)
            time.sleep(30)
    algorithm, _, wanted = digest.partition(":")
    found = hashlib.new(algorithm)
    if path.exists():
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""):
                found.update(block)
    if not path.exists() or path.stat().st_size != size or found.hexdigest() != wanted:
        return f"{path}: {path.stat().st_size if path.exists() else 0} bytes, {algorithm} {found.hexdigest()}; the pin table says {size} bytes, {wanted}"


def fetch(job):
    """Download what a job reads and is not yet on disk, eight files at a time, and check every file against the pin table."""
    from multiprocessing.pool import ThreadPool
    with ThreadPool(8) as pool:
        wrong = [w for w in pool.map(fetch_file, inputs(job)) if w]
    if wrong:
        raise SystemExit("\n".join(wrong))


def build(job, workers=4, keep=False):
    """One job of the build: fetch and check its downloads, read them into data/distill/build, and delete a bulky download.
    A read also needs what the project already holds, so that it does not repeat it: the corpus and tokenizer of
    the corpus-v1 release, extracted at the root, and data/public/train.jsonl and validation.jsonl."""
    d.fetch("" if job == "paragraphs" else "benchmarks/")  # every read checks its rows against the benchmarks; the paragraphs are read from the pool's own downloads
    fetch(job)
    done = [collect(name, slice_name, where, workers=workers, listed=listed, long=True, clean=True) for name, slice_name, where, listed in reads(job)]
    if job.startswith(BULKY) and not keep:
        for _, path, *_ in inputs(job):
            path.unlink(missing_ok=True)
    return [{k: r[k] for k in ("collection", "slice", "rows", "words", "sha256", "seconds")} for r in done]


def merge():
    """Settle what repeats across the build's slices: of rows with one mark, the first in ORDER (then by slice name,
    then as read) stays, and the others go to build/duplicates.tsv (collection/slice, id, key). A subtitle row of
    the pool, sources.jsonl, comes before them all, so a later release of the same subtitles does not repeat it.
    Only the marks files are read, twice, so no row text is needed here."""
    import numpy
    slices = [file for name in ORDER for file in sorted((BUILD / name).glob("*.marks.tsv.gz"))]
    pool = d.OUT / "sources.jsonl"
    marks = array.array("Q", (d.digest(d.template(r["text"])) for r in d.jsonl(pool) if r["source"] == "opensubtitles_th") if pool.exists() else ())
    held = len(marks)
    for file in slices:
        with gzip.open(file, "rt", encoding="utf8") as lines:
            marks.extend(int(line.split("\t")[2], 16) for line in lines)
    first = numpy.zeros(len(marks), dtype=bool)
    first[numpy.unique(numpy.frombuffer(marks, dtype=numpy.uint64), return_index=True)[1]] = True
    table, n = {}, held
    with open(BUILD / "duplicates.tsv", "w", encoding="utf8") as out:
        for file in slices:
            where, t = f"{file.parent.name}/{file.name[:-13]}", collections.Counter()
            with gzip.open(file, "rt", encoding="utf8") as lines:
                for line in lines:
                    row_id, key, _, words = line.rstrip("\n").split("\t")
                    state = "left" if first[n] else "repeats"
                    t[state + " rows"] += 1
                    t[state + " words"] += float(words)
                    if not first[n]:
                        out.write(f"{where}\t{row_id}\t{key}\n")
                    n += 1
            table[where] = {k: round(v) for k, v in sorted(t.items())}
    done = {"order": ORDER, "pool_subtitle_rows_held": held, "slices": table,
            "all": {k: sum(t.get(k, 0) for t in table.values()) for k in ("left rows", "left words", "repeats rows", "repeats words")}}
    (BUILD / "duplicates.json").write_text(json.dumps(done, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    return done["all"]


EXPECTED = pathlib.Path(__file__).with_name("crawl_build.json")  # rows, words and sha256 of every slice of the build, as recorded


def built():
    return {f"{file.parent.name}/{file.stem}": {k: r[k] for k in ("rows", "words", "sha256")}
            for file in sorted(BUILD.glob("*/*.json")) for r in [json.loads(file.read_text(encoding="utf8"))] if "sha256" in r}


def verify():
    """Compare every slice built here with the record: its rows, its words and the hash of its ids and text."""
    expected, found = json.loads(EXPECTED.read_text(encoding="utf8")), built()
    wrong = {name: {"built": found[name], "recorded": expected.get(name)} for name in found if found[name] != expected.get(name)}
    print(json.dumps({"slices checked": len(found), "slices recorded and not built here": len(expected.keys() - found.keys()), "differ": wrong}, ensure_ascii=False, indent=1))
    return not wrong


if __name__ == "__main__":
    if sys.argv[1:2] in (["tally"], ["cap"]):
        print(json.dumps({"tally": tally, "cap": cap}[sys.argv[1]](sys.argv[2]), ensure_ascii=False, indent=1))
        sys.exit()
    if sys.argv[1:2] in (["jobs"], ["fetch"], ["build"], ["merge"], ["verify"], ["record"]):
        command, options = sys.argv[1], [a for a in sys.argv[2:] if a.startswith("--")]
        names = [a for a in sys.argv[2:] if not a.startswith("--")]
        if command == "jobs":
            print("\n".join(jobs()))
        elif command in ("fetch", "build"):
            for job in (jobs() if names == ["all"] else names):
                print(json.dumps({"job": job, "read": fetch(job) if command == "fetch" else build(job, keep="--keep" in options)}, ensure_ascii=False), flush=True)
        elif command == "merge":
            print(json.dumps(merge(), indent=1))
        elif command == "record":
            EXPECTED.write_text(json.dumps(built(), indent=1) + "\n", encoding="utf8")
        else:
            sys.exit(0 if verify() else 1)
        sys.exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("collection", choices=sorted(COLLECTIONS))
    parser.add_argument("slice")
    parser.add_argument("where")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--bytes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pages", type=int, default=None, help="stop after this many pages, for a trial")
    parser.add_argument("--listed", action="store_true", help="read only pages of the domains in crawl_domains.txt")
    parser.add_argument("--long", action="store_true", help=f"keep whole paragraphs up to {LONG_MAX} characters, marked long")
    parser.add_argument("--again", action="store_true", help="read a lost slice once more: rows kept before are not dropped as duplicates")
    args = parser.parse_args()
    done = collect(args.collection, args.slice, args.where, args.start, args.bytes, args.workers, args.pages, args.listed, args.long, args.again)
    print(json.dumps({k: v for k, v in done.items() if k != "domains"} | {"top_domains": dict(itertools.islice(done["domains"].items(), 25))}, ensure_ascii=False, indent=1))
