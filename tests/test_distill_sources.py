import bz2
import collections
import gzip
import io
import json
import tarfile
import zipfile

import pytest

import train.distill_sources as d

CHAT = d.Source("chat", "Chat", "chat", "MIT", "", "", "", None, None, opener_cap=2)
PROSE = d.Source("prose", "Prose", "news", "CC BY 4.0", "", "", "", None, None, prose=True)
TALK = d.Source("talk", "Talk", "task dialogue", "MIT", "", "", "", None, None, dialogue=True, frames=True)
ASKED = d.Source("asked", "Asked", "questions and answers", "MIT", "", "", "", None, None, asked=True)


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "RAW", tmp_path)
    return tmp_path


def write(path, text, opener=open):
    path.parent.mkdir(parents=True, exist_ok=True)
    with opener(path, "wt", encoding="utf8") as out:
        out.write(text)


def sift(source, texts, blocked=frozenset(), benchmark=frozenset(), lang="en"):
    counts = collections.Counter()
    kept = d.sift(source, [(f"r{i}", t, lang) for i, t in enumerate(texts)], lambda batch: [len(t.split()) for t in batch],
                  blocked, set(benchmark), set(), collections.Counter(), counts)
    return [r["text"] for r in kept], counts


def test_reject_names_what_is_wrong():
    assert d.reject("I would like a table for two tonight.", "en") is None
    assert d.reject("ขอจองโต๊ะสำหรับสองคนคืนนี้", "th") is None
    assert d.reject("see https://example.org for more", "en") == "not a sentence"
    assert d.reject("for i in range(n): print(i)", "en") == "not a sentence"
    assert d.reject("| 12 | 40 | 7 | total", "en") == "not a sentence"
    assert d.reject("Alan Smith, Brenda Jones, Carl Young, Dora Hill", "en") == "not a sentence"
    assert d.reject("สวัสดี this line is mostly English words", "th") == "not a sentence"
    assert d.reject('He said "this is not closed and walked away.', "en") == "not a sentence"  # an odd number of quotes
    assert d.reject("เพลงรักเธอคนเดียว - วงดนตรีแห่งหนึ่ง", "th") == "not a sentence"  # song - artist
    assert d.reject("History of the town", "en", PROSE) == "not a sentence"  # a heading: no full stop
    assert d.reject("The town was founded long ago.", "en", PROSE) is None
    assert d.reject("you can mail me at someone@example.org okay", "en") == "contact details"
    assert d.reject("my number is 081 234 5678 call me", "en") == "contact details"
    assert d.reject("Your booking reference is 7GAWK763 for tonight.", "en") == "contact details"
    assert d.reject("I live in Raleigh, NY, 40147 near the park.", "en") == "contact details"  # an invented address
    assert d.reject("The postcode is cb58rs and the train is TR1234.", "en") == "contact details"
    assert d.reject("I live at 8061 Primrose Ave now.", "en") == "contact details" and d.reject("call (817) 721-3816 after five", "en") == "contact details"
    assert d.reject("My login is davidwilliams576 on the site.", "en") == "contact details"
    assert d.reject("It's Joseph Banter.", "en", TALK) == "contact details" and d.reject("Crystal Minh", "en", TALK) == "contact details"
    assert d.reject("The war lasted from 1939-1945 in Europe.", "en") is None
    assert d.reject("คณะวิศวกรรมศาสตร์ มหาวิทยาลัยแห่งหนึ่ง", "th", PROSE) == "not a sentence"  # a label: no verb
    assert d.reject("อำเภอนี้แบ่งการปกครองออกเป็นสามตำบลและยี่สิบหมู่บ้าน", "th", PROSE) is None
    assert d.reject("อำเภอนี้แบ่งเป็นสามตำบล", "th", PROSE) == "too short"  # written Thai this short is a caption or a heading
    assert d.reject("เขาถูกจับในข้อหาค้าภาพโป๊เมื่อปีที่แล้ว", "th") == "adult"
    assert d.reject("Thank you!", "en") == "too short"
    assert d.reject("ดีจ้า", "th") == "too short"
    assert d.reject("word " * 80, "en") == "too long"


def test_thai_rows_that_stop_or_start_mid_thought_are_dropped():
    assert d.reject("อำเภอนี้แบ่งการปกครองออกเป็นสามตำบลและยี่สิบหมู่บ้าน ได้แก่", "th", PROSE) == "unfinished"  # leads into a list
    assert d.reject("เขาบอกกับผู้สื่อข่าวเมื่อวานนี้ว่า", "th") == "unfinished" and d.reject("อาหารที่นี่อร่อยกว่า", "th") is None  # กว่า is not ว่า
    assert d.reject("เขาทำงานเป็นเจ้าหน้าที่", "th") is None and d.reject("ทั้งหมดนี้เป็นงบประมาณ", "th") is None  # whole words, not endings
    assert d.reject("ถ้าฝนตกหนักในช่วงบ่ายวันนี้", "th") == "unfinished"  # a condition with no result
    assert d.reject("ถ้าฝนตกหนักเราจะไม่ออกไปข้างนอก", "th") is None
    assert d.reject("เมื่อวานนี้ฉันไปตลาดกับแม่", "th") is None  # yesterday, not "when"
    assert d.reject("ซึ่งเป็นเมืองหลวงของประเทศ", "th") == "leans on the row before"
    assert d.reject("ทำให้การจราจรติดขัดทั่วทั้งเมือง", "th") == "leans on the row before"
    assert d.reject("สามารถเดินทางไปยังตัวเมืองได้ด้วยรถไฟสายเหนือทุกวัน", "th", PROSE) == "leans on the row before"  # a verb with no subject
    assert d.reject("ก่อนหน้านี้ ยังเป็นที่ตั้งของมหาวิทยาลัยที่เก่าแก่ที่สุดของประเทศ", "th", PROSE) == "leans on the row before"
    assert d.reject("ก่อนหน้านี้ เมืองนี้ยังเป็นที่ตั้งของมหาวิทยาลัยที่เก่าแก่ที่สุด", "th", PROSE) is None


def test_english_rows_that_point_elsewhere_or_lost_a_word_are_dropped():
    assert d.reject("It was built in 1890 by the town.", "en", PROSE) == "leans on the row before"
    assert d.reject("These were later sold to a museum.", "en", PROSE) == "leans on the row before"
    assert d.reject("They were later sold to a museum.", "en", PROSE) is None  # Thai drops the pronoun, so this translates
    assert d.reject("See the next section for the full list.", "en", PROSE) == "pointer"
    assert d.reject("Most of this page is about the old town.", "en", PROSE) == "pointer"
    assert d.reject("The mayor was replaced by .", "en", PROSE) == "words missing"
    assert d.reject("The mayor was replaced by a younger man.", "en", PROSE) is None
    assert d.reject("Trains connect the city with Vienna Intl.", "en", PROSE) == "not a sentence"  # cut at an abbreviation
    loose = d.Source("loose", "", "", "", "", "", "", None, None, prose=True, strict=True)
    assert d.reject("The town paid for both of these bridges.", "en", loose) == "leans on the row before"  # a demonstrative anywhere
    assert d.reject("The town paid for both of these bridges.", "en", PROSE) is None


def test_instructions_and_chatbot_voice_are_dropped_from_question_and_answer_sources():
    assert d.reject("Write a poem about the sea at night.", "en", ASKED) == "instruction or chatbot voice"
    assert d.reject("As an AI, I cannot give medical advice.", "en", ASKED) == "instruction or chatbot voice"
    assert d.reject("I hope this helps you save some money.", "en", ASKED) == "instruction or chatbot voice"
    assert d.reject("The sea is calm at night", "en", ASKED) == "instruction or chatbot voice"  # no closing punctuation: a heading or a list item
    assert d.reject("จงสรุปบทความต่อไปนี้ให้สั้นลง", "th", ASKED) == "instruction or chatbot voice"
    assert d.reject("Why is the sea salty?", "en", ASKED) is None and d.reject("Write a poem about the sea at night.", "en") is None


def test_royal_family_stock_sentences_and_old_fiction():
    news = d.Source("news", "", "", "", "", "", "", None, None, prose=True, royal=True)
    assert d.reject("พระบาทสมเด็จพระเจ้าอยู่หัวเสด็จเยือนจังหวัดเชียงใหม่", "th") == "royal family"
    assert d.reject("พระบาทสมเด็จพระเจ้าอยู่หัวเสด็จเยือนจังหวัดเชียงใหม่", "th", news) is None
    assert d.reject("King Bhumibol opened the bridge in 1972.", "en") == "royal family"
    stubs = d.Source("wiki", "", "", "", "", "", "", None, None, prose=True, stubs=True)
    kept = [i for i in range(200) if d.reject("She was born in a small town near Lyon.", "en", stubs, f"wiki-{i}") is None]
    assert 20 <= len(kept) <= 60  # one stock sentence in five stays
    modern = d.Source("cv", "", "", "", "", "", "", None, None, modern=True)
    assert d.reject("I would fly to do your bidding, madam.", "en", modern) == "old fiction"
    assert d.reject('"Come here," he whispered to the boy.', "en", modern) == "old fiction"  # quoted speech
    assert d.reject("Can we meet tonight after work?", "en", modern) is None


def test_labels_option_markers_and_foreign_script_brackets_are_stripped():
    assert d.prepare("Answer: The capital is Paris.", "en", ASKED) == ("The capital is Paris.", ["label removed"])
    assert d.prepare("(ก) ผู้เช่าต้องชำระค่าเช่าทุกเดือน", "th", ASKED) == ("ผู้เช่าต้องชำระค่าเช่าทุกเดือน", ["option marker removed"])
    assert d.prepare("ฮงโง (本郷) เป็นย่านหนึ่งในโตเกียว", "th") == ("ฮงโง เป็นย่านหนึ่งในโตเกียว", ["bracket removed"])
    assert d.prepare("Answer: yes", "en") == ("Answer: yes", [])  # chat keeps what was typed


def test_dialogue_rows_carry_the_turn_before_and_need_it_when_they_lean():
    counts = collections.Counter()
    batch = [("a", "Yes, that works for me.", "en", "Shall I book the table for seven?"), ("b", "Yes, that works for me too.", "en", None),
             ("c", "I would like to book a table for two people tonight.", "en", None),
             ("d", "Please book the second one for me.", "en", "Call me on 081 234 5678 to confirm."),
             ("e", "I want a table at Nopa for 7 pm.", "en", "Where to?", "I want a table at Xx for Xx."),
             ("f", "I want a table at Zuni for 8 pm.", "en", "Where to?", "I want a table at Xx for Xx.")]
    monkey = d.FRAME_CAP
    d.FRAME_CAP = 1
    try:
        kept = d.sift(TALK, batch, lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), counts)
    finally:
        d.FRAME_CAP = monkey
    assert [(r["id"], r.get("context")) for r in kept] == [("a", "Shall I book the table for seven?"), ("c", None), ("d", None), ("e", "Where to?")]
    assert counts["needs the turn before"] == 1 and counts["repeated frame"] == 1  # b leans on nothing, f repeats e's frame


def test_duplicates_differ_only_by_case_digits_or_punctuation():
    kept, counts = sift(CHAT, ["Your table is booked for 7 pm.", "your table is booked for 9 PM", "Your table is ready now."])
    assert kept == ["Your table is booked for 7 pm.", "Your table is ready now."]
    assert counts["duplicate"] == 1
    assert d.template("ไป  เที่ยว!") == d.template("ไปเที่ยว") != d.template("ไปเทียว")  # Thai marks are kept


def test_text_the_project_already_holds_is_left_out(tmp_path):
    write(tmp_path / "data/public/corpus/test.jsonl", json.dumps({"en": "Where is the station?", "th": "สถานีอยู่ที่ไหน"}) + "\n")
    write(tmp_path / "data/public/train.jsonl", json.dumps({"source": "An older public pair.", "target": "คู่เก่า"}) + "\n")
    write(tmp_path / "suites/dev.jsonl", json.dumps({"source": "a suite line", "context": ["A: a context line here"]}) + "\n")
    write(tmp_path / "data/synthetic-chat-v1/train.jsonl", json.dumps({"source": "a synthetic chat line", "target": "x"}) + "\n")
    blocked = d.project_text(tmp_path)
    kept, counts = sift(CHAT, ["where is the station", "An older public pair.", "A suite line!", "A: a context line here",
                               "a synthetic chat line", "something new to translate"], blocked)
    assert kept == ["something new to translate"] and counts["already in project data"] == 5
    assert sift(CHAT, ["สถานีอยู่ที่ไหน"], blocked, lang="th")[1]["already in project data"] == 1


def test_benchmark_sentences_and_their_fragments_are_left_out(raw, tmp_path):
    flores = "The quick brown fox jumped over the lazy dog near the old stone bridge.\n".encode()
    (raw / "benchmarks").mkdir()
    with tarfile.open(raw / "benchmarks/flores200_dataset.tar.gz", "w:gz") as archive:
        for name, data in (("flores200_dataset/devtest/eng_Latn.devtest", flores), ("flores200_dataset/dev/fra_Latn.dev", b"Une phrase.\n")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    write(raw / "benchmarks/wmt24pp-en-th_TH.jsonl", json.dumps({"source": "Short test line.", "target": "บรรทัดทดสอบสั้น"}) + "\n")
    write(tmp_path / "root/data/public/train.jsonl", json.dumps({"source": "A news sentence from the older public pairs set here.", "target": "ข่าว"}) + "\n")
    benchmark = d.benchmark_text(tmp_path / "root")
    kept, counts = sift(PROSE, ["The quick brown fox jumped over the lazy dog near the old stone bridge.",
                                "Yesterday the quick brown fox jumped over the lazy dog near the river.",  # shares eight words in a row
                                "Short test line.", "He read a news sentence from the older public pairs set.",
                                "The slow brown fox walked under the bridge.", "Une phrase courte."], benchmark=benchmark)
    assert kept == ["The slow brown fox walked under the bridge.", "Une phrase courte."] and counts["benchmark overlap"] == 4
    assert sift(CHAT, ["บรรทัดทดสอบสั้น"], benchmark=benchmark, lang="th")[1]["benchmark overlap"] == 1


def test_repeated_openers_are_capped_and_long_rows_dropped():
    kept, counts = sift(CHAT, ["Tom wants to eat rice", "Mary wants to eat noodles", "Tom wants to eat 2 eggs", "She wants to eat too"])
    assert len(kept) == 3 and counts["repeated opener"] == 1  # Tom and Mary are one name
    assert sift(CHAT, ["word " * 55 + "end"])[1]["too long"] == 0 and sift(CHAT, ["a " * 101 + "b"])[1]["too long"] == 1


def test_sentence_splitting():
    assert d.english_sentences('Dr. Lee met J. Smith at 5 p.m. in Paris. "Why?" He left. It cost $3.50 then.') == [
        "Dr. Lee met J. Smith at 5 p.m. in Paris.", '"Why?"', "He left.", "It cost $3.50 then."]
    first, second = "กรุงเทพมหานครเป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย", "แม่น้ำเจ้าพระยาไหลผ่านใจกลางเมืองลงสู่อ่าวไทย"
    assert d.thai_sentences(f"{first} {second}") == [f"{first} {second}"]  # a plain space is not a sure boundary
    assert d.thai_sentences(f"{first} นอกจากนี้ {second}") == [first, f"นอกจากนี้ {second}"]  # a word that only opens a sentence is
    assert d.thai_sentences(f"{first} ส่วนใหญ่{second}") == [f"{first} ส่วนใหญ่{second}"]  # ส่วนใหญ่ is not the opener ส่วน
    assert d.thai_sentences(f"{first} (ดูเพิ่มที่ นอกจากนี้ {second})") == [f"{first} (ดูเพิ่มที่ นอกจากนี้ {second})"]  # inside brackets
    assert d.thai_sentences("สั้น ต่อมา ก็ยังสั้น") == ["สั้น ต่อมา ก็ยังสั้น"]
    assert d.sentences("Here is how:\n- one\n1. two\nIt works. Try it.", "en") == ["Here is how:", "It works.", "Try it."]
    assert d.sentences("Run this:\n```\nls\n```", "en") == []


def test_wiki_dump_gives_prose_sentences_only(raw):
    page = ("{{Infobox|name=X {{nested|a}}}}\n'''Rome''' ({{lang|it|Roma}}; {{IPA|x}}) is the [[capital city|capital]] of [[Italy]].<ref>A book</ref> The city\n"
            "is old and was led by {{w|Mayor Rossi|its mayor}} for {{convert|10|years}}.\n"
            "== History ==\n* a list item\n{| class=wikitable\n| cell\n|}\n[[File:Rome.jpg|thumb|A [[view]] of Rome]]\n"
            "The town has [https://example.org many] visitors.<!-- hidden -->\n[[Category:Cities]]")
    xml = ('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
           f"<page><title>Rome</title><ns>0</ns><id>7</id><revision><text>{page}</text></revision></page>"
           '<page><title>Roma</title><ns>0</ns><id>8</id><redirect title="Rome"/><revision><text>#REDIRECT [[Rome]]</text></revision></page>'
           "<page><title>Talk:Rome</title><ns>1</ns><id>9</id><revision><text>Talk text here.</text></revision></page></mediawiki>")
    write(raw / "tinywiki/dump.xml.bz2", xml.replace("<ref>", "&lt;ref&gt;").replace("</ref>", "&lt;/ref&gt;").replace("<!--", "&lt;!--").replace("-->", "--&gt;"), bz2.open)
    assert list(d.wiki("tinywiki", "en", 10)) == [
        ("tinywiki-7-0", "Rome (Roma) is the capital of Italy.", "en"),  # the template that held a word keeps it
        ("tinywiki-7-1", "The city is old and was led by its mayor for 10 years.", "en"),  # a typed line break is not a paragraph
        ("tinywiki-7-2", "The town has many visitors.", "en")]
    assert len(list(d.wiki("tinywiki", "en", 2))) == 2 and list(d.wiki("tinywiki", "en", 10, "{{publish")) == []
    assert d.wiki_paragraphs("ประโยคนี้ถูกตัด\nบรรทัดกลางคำ\n\nย่อหน้าใหม่") == ["ประโยคนี้ถูกตัดบรรทัดกลางคำ", "ย่อหน้าใหม่"]


def test_global_voices_leaves_out_quoted_posts_and_digests(raw):
    own = "".join(f'<s id="2.{i}">Our own sentence number {i} is long enough to be more than a caption.</s>' for i in range(1, 7))
    (raw / "globalvoices").mkdir()
    with zipfile.ZipFile(raw / "globalvoices/en.zip", "w") as archive:
        archive.writestr("README", "notes")
        archive.writestr("GlobalVoices/raw/en/2010_01_02_a-story_.xml", '<document><URL>u</URL><TITLE><s id="1.1">A title</s></TITLE>'
                         f'<P id="1">{own}<s id="2.7">Photo by someone, used with permission.</s><s id="2.8">GV: A question we asked?</s>'
                         '<QUOTE><P id="2"><s id="3.1">Somebody else wrote this.</s></P></QUOTE></P></document>')
        archive.writestr("GlobalVoices/raw/en/2010_01_03_a-digest_.xml", '<document><P id="1"><s id="1.2">Some blog has a round-up.</s></P></document>')
    assert [r[0] for r in d.global_voices()] == [f"globalvoices-2010_01_02_a-story_-2.{i}" for i in range(1, 7)]


def test_tatoeba_leaves_out_the_cc0_sentences(raw):
    write(raw / "t/eng.tsv", "1\teng\tI like tea.\tann\t\\N\t\\N\n2\teng\tShe runs.\tbob\t\\N\t\\N\n3\teng\tHe reads a book every single night.\tbob\t\\N\t\\N\n")
    write(raw / "t/cc0.tsv.bz2", "2\teng\tShe runs.\t2019\n", bz2.open)
    assert list(d.tatoeba("t/eng.tsv", "en", "t/cc0.tsv.bz2")) == [("tatoeba-1", "I like tea.", "en"), ("tatoeba-3", "He reads a book every single night.", "en")]
    assert [r[0] for r in d.tatoeba("t/eng.tsv", "en", "t/cc0.tsv.bz2", min_words=6)] == ["tatoeba-3"]


def test_dialogue_formats(raw):
    dialogue = [{"turns": [{"speaker": "USER", "utterance": "Find me a hotel in Leeds.", "frames": [{"slots": [{"slot": "city", "start": 19, "exclusive_end": 24}]}]},
                           {"speaker": "SYSTEM", "utterance": "For which night?"}, {"speaker": "USER", "utterance": "Tonight please."}]}]
    write(raw / "multiwoz/train_001.json", json.dumps(dialogue))
    assert list(d.plain_files("multiwoz", "*.json")) == [("multiwoz-train_001-0-0", "Find me a hotel in Leeds.", "en", None, "Find me a hotel in Xx."),
                                                         ("multiwoz-train_001-0-2", "Tonight please.", "en", "For which night?", "Tonight please.")]
    talk = [{"utterances": [{"speaker": "ASSISTANT", "text": "Where to?"}, {"speaker": "USER", "text": "A flight to Rome please.", "segments": [{"start_index": 12, "end_index": 16}]}]}]
    write(raw / "taskmaster2/flights.json", json.dumps(talk))
    assert list(d.plain_files("taskmaster2", "*.json", list_key="utterances", text_key="text")) == [
        ("taskmaster2-flights-0-1", "A flight to Rome please.", "en", "Where to?", "A flight to Xx please.")]
    (raw / "sgd").mkdir()
    with zipfile.ZipFile(raw / "sgd/sgd.zip", "w") as archive:
        archive.writestr("repo/train/dialogues_001.json", json.dumps(dialogue))
        archive.writestr("repo/train/schema.json", "[]")
    assert [r[0] for r in d.sgd()] == ["sgd-train001-0-0", "sgd-train001-0-2"]
    write(raw / "abcd/abcd_v1.1.json.gz", json.dumps({"train": [{"convo_id": 3, "original": [
        ["agent", "How can I help?"], ["action", "Searching the FAQ pages ..."], ["customer", "My order is late."]]}]}), gzip.open)
    assert list(d.abcd()) == [("abcd-3-2", "My order is late.", "en", "How can I help?")]  # the customer's side, the agent's as context
    write(raw / "byu/dataset.json", json.dumps({"c1": {"messages": [[{"text": "hey", "sender": "a"}, {"text": "you there?", "sender": "a"}], [{"text": "yep", "sender": "b"}, {"text": "The purple elephant danced gracefully.", "sender": "b"}]]}}))
    assert [r[:2] + r[3:] for r in d.byu()] == [("byu-c1-0", "hey you there?", None), ("byu-c1-1", "yep", "hey you there?")]  # one person's run of messages is one line
    assert d.frame("Book a table at Nopa for 7 pm, I think") == d.frame("Book a table at Zuni for 8 pm, I think") != d.frame("Book a room at Nopa for 7 pm")


def test_question_and_answer_formats(raw):
    write(raw / "dolly/databricks-dolly-15k.jsonl", "".join(json.dumps({"instruction": q, "context": "Left out.", "response": "Left out too."}) + "\n"
                                                             for q in ("Which fruit is red?", "Name a fruit.", "What is the main idea of the paragraph?")))
    assert [r[1] for r in d.dolly()] == ["Which fruit is red?"]  # questions only, and not about a text that is not here
    write(raw / "commonvoice/en-wiki.en.txt", "They moved to the city in the spring of that year.\nShort line.\nThe river runs through the old town to the sea.\n"
                                              "The prize was first given to the chemist J. O.\n")
    write(raw / "commonvoice/en-wiki.en.rerun-2021-11.txt", "")
    assert list(d.common_voice_wiki()) == [("commonvoice_wiki-a-2", "The river runs through the old town to the sea.\n", "en")]


def test_parquet_formats(raw):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    def table(path, rows):
        (raw / path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), raw / path)

    for split in ("train", "valid", "test"):
        table(f"soda/{split}.parquet", [{"dialogue": ["Hi there!", "Hello."], "original_index": 8}, {"dialogue": ["Skipped."], "original_index": 9}])
    assert list(d.soda())[:2] == [("soda-8-0", "Hi there!", "en", None), ("soda-8-1", "Hello.", "en", "Hi there!")] and len(list(d.soda())) == 6
    table("blognone/train.parquet", [{"title": "t", "author": "a", "content": "บริษัทเปิดตัวโทรศัพท์รุ่นใหม่เมื่อวานนี้\n\nที่มา - สำนักข่าว"}])
    assert list(d.blognone()) == [("blognone-0-0", "บริษัทเปิดตัวโทรศัพท์รุ่นใหม่เมื่อวานนี้", "th")]
    table("iapp/train.parquet", [{"question_id": "q_000", "question": "ใครเป็นผู้คิดค้นหลอดไฟ", "context": "ไม่ใช้"}])
    assert list(d.iapp()) == [("iapp-q_000", "ใครเป็นผู้คิดค้นหลอดไฟ", "th")]


def test_every_source_is_pinned_and_capped_sources_are_the_repetitive_ones():
    take = {s.name: s.take for s in d.SOURCES}
    assert {n for n, t in take.items() if t} == {"soda", "tatoeba_en", "simplewiki", "commonvoice_wiki", "iapp"}
    assert take["soda"] == take["tatoeba_en"] == 100_000 and set(d.PINS) == set(d.downloads())


def test_mix_takes_each_sources_share_and_repeats_exactly(tmp_path, monkeypatch):
    big, small = d.Source("big", "Big", "news", "CC0 1.0", "", "", "", 3, None), d.Source("small", "Small", "chat", "MIT", "", "", "", 5, None)
    monkeypatch.setattr(d, "OUT", tmp_path)
    monkeypatch.setattr(d, "SOURCES", (big, small))
    rows = [{"id": f"{s}-{i}", "text": f"text {s} {i}", "lang": lang, "source": s, "register": "r", "licence": "l", "pieces": 4}
            for s, n, lang in (("big", 10, "en"), ("small", 2, "th")) for i in range(n)]
    write(tmp_path / "candidates.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    write(tmp_path / "candidates.json", json.dumps({"files": {}, "sources": {"big": {"usable": 10}, "small": {"usable": 2}}}))
    d.mix()
    first = (tmp_path / "sources.jsonl").read_bytes()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["rows"] == 5 and manifest["by_language"] == {"en": 3, "th": 2}  # small has only two to give
    assert manifest["sources"]["big"]["in_mix"] == 3 and manifest["sources"]["small"] == manifest["sources"]["small"] | {"usable": 2, "target": 5, "in_mix": 2}
    assert all("pieces" not in json.loads(line) for line in first.decode().splitlines()) and manifest["lengths"]["all"]["pieces_mean"] == 4
    d.mix()
    assert (tmp_path / "sources.jsonl").read_bytes() == first


def test_near_duplicates_lose_to_the_earlier_row_and_short_rows_are_left_alone():
    booked = "I have booked a table for four people this Friday evening at the Golden Curry restaurant in the centre of town, as you asked me to."
    texts = [booked, booked.replace("Golden Curry", "Rice Boat"),  # same start and end, one name changed
             "Sorry, but " + booked[2:],  # same ending only
             booked[:55] + " and they also asked whether you could bring the birthday cake with you.",  # same start, different sentence
             "I like green tea.", "I like green tea!!",  # under 50 letters: never compared
             "A completely different sentence that is long enough to be checked but matches nothing else here."]
    rows = [{"id": str(i), "text": t, "source": "first" if i < 2 else "second"} for i, t in enumerate(texts)]
    dropped = collections.Counter()
    kept = [r["id"] for r in d.without_near_duplicates(lambda: iter(rows), dropped)]
    assert kept == ["0", "3", "4", "5", "6"] and dropped == {"first": 1, "second": 1}


def test_voice_of_america_articles_that_name_a_wire_agency_are_left_out(raw):
    def record(number, kind, paragraphs):
        data = json.dumps({"filename": f"a_slug_{number}", "paragraphs": paragraphs}, ensure_ascii=False).encode()
        info = tarfile.TarInfo(f"tha_voathai/{kind}/a_slug_{number}.json")
        info.size = len(data)
        return info, io.BytesIO(data)

    (raw / "mot").mkdir()
    with tarfile.open(raw / "mot/tha_voathai.tgz", "w:gz") as archive:
        archive.addfile(*record(1234567, "article", ["นายกรัฐมนตรีเดินทางเยือนกรุงวอชิงตันเมื่อวานนี้", "(ที่มา: วีโอเอ)"]))
        archive.addfile(*record(999, "article", ["บท ความ เก่า ที่ เว้น วรรค ผิด ทุก คำ ใน ประโยค นี้"]))
        archive.addfile(*record(2345678, "article", ["สำนักข่าวรอยเตอร์รายงานว่าราคาน้ำมันปรับตัวสูงขึ้นอีกครั้ง"]))
        archive.addfile(*record(3, "audio", ["รายการข่าวภาคเช้าประจำวันศุกร์ที่ผ่านมานี้"]))
    assert list(d.voa_thai()) == [("voathai-1234567-0", "นายกรัฐมนตรีเดินทางเยือนกรุงวอชิงตันเมื่อวานนี้", "th")]


def test_leading_connectives_are_removed_and_logical_ones_make_the_row_lean():
    assert d.prepare("However, the bridge was never finished.", "en", PROSE) == ("The bridge was never finished.", ["connective removed"])
    assert d.prepare("นอกจากนี้ เมืองนี้ยังเป็นที่ตั้งของมหาวิทยาลัยหลายแห่ง", "th", PROSE) == ("เมืองนี้ยังเป็นที่ตั้งของมหาวิทยาลัยหลายแห่ง", ["connective removed"])
    assert d.prepare("ส่วนใหญ่แล้วชาวบ้านทำอาชีพเกษตรกรรม", "th", PROSE)[1] == []  # ส่วนใหญ่ is a word, not the connective ส่วน
    assert d.prepare("ประเทศไทยเเละกัมพูชา", "th") == ("ประเทศไทยและกัมพูชา", ["sara ae repaired"])
    assert d.reject("Therefore the bridge was never finished.", "en", PROSE) == "leans on the row before"
    assert d.reject("ดังนั้นสะพานแห่งนี้จึงไม่เคยสร้างเสร็จสมบูรณ์จนถึงทุกวันนี้", "th", PROSE) == "leans on the row before"


def test_task_dialogues_lose_closings_garble_and_echoes():
    counts = collections.Counter()
    batch = [("a", "Thank you, that will be all.", "en", "Anything else?"), ("b", "I I want to to go to the the airport.", "en", "Where to?"),
             ("c", "I would like a table for two tonight.", "en", "My login is davidwilliams576 by the way."), ("d", "At seven", "en", "Shall I book it at seven tonight?"),
             ("e", "I would like a table near the window please.", "en", "Any other requests?")]
    kept = d.sift(TALK, batch, lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), counts)
    assert [(r["id"], r.get("context")) for r in kept] == [("c", None), ("e", "Any other requests?")]  # a context holding a username is not carried
    assert counts == {"only a closing": 1, "garbled speech": 1, "too short": 1}
