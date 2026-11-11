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
    assert d.reject("It's Joseph Banter.", "en", TALK) == "contact details"
    assert d.reject("Yes, my name is Joyce Wu and I ordered a coat.", "en", TALK) == "contact details"
    assert d.reject("Your new promo code is 0TAXW today.", "en") == "contact details" and d.reject("Crystal Minh", "en", TALK) == "contact details"
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
    assert d.reject("สามารถเดินทางไปยังตัวเมืองได้ด้วยรถไฟสายเหนือทุกวัน", "th", PROSE) == "subject in the row before"  # a verb with no subject
    assert d.reject("ก่อนหน้านี้ ยังเป็นที่ตั้งของมหาวิทยาลัยที่เก่าแก่ที่สุดของประเทศ", "th", PROSE) == "subject in the row before"
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
    # merging before dropping: a sentence that leans on the one before, or is very short, is joined to it again
    assert d.thai_sentences(f"{first} ดังนั้น {second}") == [f"{first} ดังนั้น {second}"]
    assert d.thai_sentences(f"{first} นอกจากนี้ ยังมีตลาดน้ำที่นักท่องเที่ยวนิยมไปเยี่ยมชมกันมาก") == [f"{first} นอกจากนี้ ยังมีตลาดน้ำที่นักท่องเที่ยวนิยมไปเยี่ยมชมกันมาก"]
    assert d.thai_sentences(f"{first} ปัจจุบัน ปิดแล้ว") == [f"{first} ปัจจุบัน ปิดแล้ว"]
    assert len(d.thai_sentences(f"{first * 6} ดังนั้น {second}")) == 2  # unless the two no longer fit one row
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
    write(raw / "byu/dataset.json", json.dumps({"c1": {"messages": [[{"text": "hey", "sender": "a"}, {"text": "you there?", "sender": "a"}], [{"text": "yep", "sender": "b"}, {"text": "The purple elephant danced gracefully.", "sender": "b"}, {"text": "h e l l o", "sender": "b"}]]}}))
    assert [r[:2] + r[3:] for r in d.byu()] == [("byu-c1-0-0", "hey", None), ("byu-c1-0-1", "you there?", "hey"), ("byu-c1-1-0", "yep", "you there?")]  # one message a row; planted and letter-spaced lines go
    assert d.frame("Book a table at Nopa for 7 pm, I think") == d.frame("Book a table at Zuni for 8 pm, I think") != d.frame("Book a room at Nopa for 7 pm")


def test_question_and_answer_formats(raw):
    write(raw / "dolly/databricks-dolly-15k.jsonl", "".join(json.dumps({"instruction": q, "context": "Has a passage." if "moon" in q else "", "response": "Left out too."}) + "\n"
                                                             for q in ("Which fruit is red?", "Name a fruit.", "What is the main idea of the paragraph?")))
    assert [r[1] for r in d.dolly()] == ["Which fruit is red?"]  # questions only, and not about a text that is not here
    write(raw / "commonvoice/en-wiki.en.txt", "They moved to the city in the spring of that year.\nShort line.\nThe river runs through the old town to the sea.\n"
                                              "The prize was first given to the chemist J. O.\n")
    write(raw / "commonvoice/en-wiki.en.rerun-2021-11.txt", "")
    assert list(d.common_voice_wiki()) == [("commonvoice_wiki-a-2", "The river runs through the old town to the sea.\n", "en")]


POST = "บริษัทเปิดตัวโทรศัพท์รุ่นใหม่เมื่อวานนี้ โดยมีหน้าจอขนาดใหญ่ขึ้นและแบตเตอรี่ที่ใช้งานได้นานกว่ารุ่นก่อนหน้าถึงสองเท่า"


def test_parquet_formats(raw):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    def table(path, rows):
        (raw / path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), raw / path)

    for split in ("train", "valid", "test"):
        table(f"soda/{split}.parquet", [{"dialogue": ["Hi there!", "I was hoping to find you here today.", "*smiles* Me too."], "original_index": 8}, {"dialogue": ["Skipped."], "original_index": 9}])
    assert list(d.soda())[:1] == [("soda-8-1", "I was hoping to find you here today.", "en", "Hi there!")] and len(list(d.soda())) == 3  # greetings and stage directions go
    table("blognone/train.parquet", [{"title": "t", "author": "a", "content": f"{POST}\n\n- รายการสั้น\n\nหัวข้อย่อย\n\nที่มา - สำนักข่าว"}])
    assert list(d.blognone()) == [("blognone-0-0", POST, "th", None)]  # list items, headings and the source line go
    table("iapp/train.parquet", [{"question_id": "q_000", "question": "ใครเป็นผู้คิดค้นหลอดไฟฟ้าเป็นคนแรกของโลก", "context": "ไม่ใช้"}])
    assert list(d.iapp()) == [("iapp-q_000", "ใครเป็นผู้คิดค้นหลอดไฟฟ้าเป็นคนแรกของโลก", "th")]


def test_every_source_is_pinned_and_capped_sources_are_the_repetitive_ones():
    take = {s.name: s.take for s in d.SOURCES}
    assert {n for n, t in take.items() if t} == {"soda", "tatoeba_en"}
    assert take["soda"] == take["tatoeba_en"] == 100_000 and set(d.PINS) == set(d.downloads())


def test_mix_takes_each_sources_share_and_repeats_exactly(tmp_path, monkeypatch):
    big, small = d.Source("big", "Big", "news", "CC0 1.0", "", "", "", 3, None), d.Source("small", "Small", "chat", "MIT", "", "", "", 5, None)
    monkeypatch.setattr(d, "OUT", tmp_path)
    monkeypatch.setattr(d, "SOURCES", (big, small))
    rows = [{"id": f"{s}-{i}", "text": f"text {s} {i}", "lang": lang, "source": s, "register": "r", "licence": "l", "pieces": 4, "tier": 1}
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
    monkeypatch.setattr(d, "SOURCES", (d.Source("big", "Big", "news", "CC0 1.0", "", "", "", 0, None), d.Source("small", "Small", "chat", "MIT", "", "", "", None, None)))
    d.mix()
    assert [json.loads(line)["source"] for line in (tmp_path / "sources.jsonl").read_text().splitlines()] == ["small", "small"]  # 0 drops a source, None takes all


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
    assert list(d.voa_thai()) == [("voathai-1234567-0", "นายกรัฐมนตรีเดินทางเยือนกรุงวอชิงตันเมื่อวานนี้", "th", None)]


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


SPOKEN = d.Source("spoken", "Spoken", "spoken", "CC0 1.0", "", "", "", None, None, spoken=True)


def test_caption_lines_are_joined_into_whole_utterances_and_cut_off_ones_thrown_away():
    captions = [(0, 100, "วันนี้เราจะมาทำ"), (100, 200, "แกงเขียวหวานกันนะครับ"), (200, 300, "เริ่มจากเตรียมเครื่องแกงก่อนครับ"),  # นะครับ is said mid-sentence
                (300, 400, "[เสียงดนตรี]"), (400, 500, "ใส่กะทิลงไปแล้วก็"),  # cut off by the pause that follows
                (900, 1000, "อร่อยมากเลย"),  # a looser closing word counts just before a pause
                (2000, 2100, "- ไปไหนมา"), (2100, 2200, "ไปตลาดมาค่ะ")]
    assert list(d.utterances(captions)) == [(0, "วันนี้เราจะมาทำ แกงเขียวหวานกันนะครับ เริ่มจากเตรียมเครื่องแกงก่อนครับ"), (5, "อร่อยมากเลย"), (6, "ไปไหนมา ไปตลาดมาค่ะ")]


def test_spoken_rows_lose_a_trailing_connective_and_keep_a_spoken_opening():
    assert d.prepare("เมื่อวานไปกินข้าวกับเพื่อนมา แล้วก็", "th", SPOKEN) == ("เมื่อวานไปกินข้าวกับเพื่อนมา", ["trailing connective removed"])
    assert d.reject("แต่เมื่อวานฝนตกหนักมากเลยนะ", "th", SPOKEN) is None  # speech may open with "but"
    assert d.reject("ซึ่งเป็นร้านที่อร่อยมากเลยนะ", "th", SPOKEN) == "leans on the row before"
    assert d.reject("เมื่อวานเราไปกินข้าวที่", "th", SPOKEN) == "unfinished"
    assert d.reject("เสร็จ เรียบร้อย นะครับ ซึ่ง ใช้ เวลา นาน", "th", SPOKEN) == "not a sentence"  # typed a word at a time


def test_spoken_and_legal_formats(raw):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = [{"split": "train", "mic": "a", "speaker_id": "s1", "sentence": "เมื่อวานไปตลาดมา"}, {"split": "train", "mic": "a", "speaker_id": "s1", "sentence": "ซื้อปลามาสองตัวเลย"}, {"split": "train", "mic": "a", "speaker_id": "s1", "sentence": "แล้วก็ไปต่อที่"},
            {"split": "train", "mic": "a", "speaker_id": "s2", "sentence": "ปลาที่ตลาดนั้นแพงไหม"}, {"split": "train", "mic": "a", "speaker_id": "s1&s2", "sentence": "พูดพร้อมกัน"},
            {"split": "train", "mic": "a", "speaker_id": "s1", "sentence": "รหัส F 12"}]
    write(raw / "lotusdis/transcripts.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert [t for _, t, _ in d.lotusdis()] == ["เมื่อวานไปตลาดมา ซื้อปลามาสองตัวเลย", "ปลาที่ตลาดนั้นแพงไหม"]  # pieces join up to a closing word, and the cut-off rest goes
    messages = [{"message_id": "m1", "lang": "th", "role": "prompter", "deleted": False, "review_result": True, "synthetic": False, "text": "กาแฟช่วยให้ไม่ง่วงจริงหรือไม่"},
                {"message_id": "m2", "lang": "th", "role": "assistant", "deleted": False, "review_result": True, "synthetic": False, "text": "คำตอบของผู้ช่วย"},
                {"message_id": "m3", "lang": "th", "role": "prompter", "deleted": False, "review_result": True, "synthetic": False, "text": "สองบรรทัด\nไม่เอา"}]
    write(raw / "oasst2/ready.messages.jsonl.gz", "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in messages), gzip.open)
    assert list(d.oasst2("th")) == [("oasst2-m1", "กาแฟช่วยให้ไม่ง่วงจริงหรือไม่", "th")]
    law = "พระราชบัญญัติตัวอย่าง\n\nมาตรา ๑ พระราชบัญญัตินี้เรียกว่า พระราชบัญญัติตัวอย่าง\n\nมาตรา ๕ ให้นายทะเบียนมีอำนาจ\nออกใบอนุญาตตามพระราชบัญญัตินี้\n\nใบอนุญาตให้มีอายุห้าปี\n\nหมวด ๒ บทกำหนดโทษ\n\nหัวข้อที่ไม่ใช่มาตรา"
    (raw / "thailaw").mkdir()
    pq.write_table(pa.Table.from_pylist([{"title": "t", "text": law, "__index_level_0__": 7}]), raw / "thailaw/0.parquet")
    assert list(d.thai_law()) == [("thailaw-7-2", "ให้นายทะเบียนมีอำนาจ ออกใบอนุญาตตามพระราชบัญญัตินี้", "th"), ("thailaw-7-3", "ใบอนุญาตให้มีอายุห้าปี", "th")]


def test_thai_prose_carries_the_sentence_before_and_long_paragraphs_go_to_their_own_source():
    first, second = "กรุงเทพมหานครเป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย", "เมืองนี้ตั้งอยู่บนสามเหลี่ยมปากแม่น้ำเจ้าพระยาและมีประชากรราวสิบล้านคน"
    rows = list(d.thai_rows("x-1", [f"{first} ต่อมา {second}", "ย่อหน้ายาว " * 30]))
    assert [(r[0], r[3]) for r in rows] == [("x-1-0", None), ("x-1-1", first), ("x-1-2", f"ต่อมา {second}")]  # the sentence before may close the paragraph before
    assert [r[0] for r in d.whole_paragraphs(rows)] == ["x-1-2"]
    counts = collections.Counter()
    lean = "สามารถเดินทางไปยังตัวเมืองได้ด้วยรถไฟสายเหนือทุกวัน"
    batch = [("a", lean, "th", first), ("b", lean + "และรถประจำทาง", "th", None), ("c", "ซึ่งเป็นเมืองหลวงของประเทศมาตั้งแต่ปีพุทธศักราชสองพันสามร้อย", "th", first)]
    kept = d.sift(PROSE, batch, lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), counts)
    assert [(r["id"], r["context"]) for r in kept] == [("a", first)] and counts == {"subject in the row before": 1, "leans on the row before": 1}
    assert d.reject("หมายเหตุ รายชื่อนี้เรียงตามลำดับเวลาที่เข้ารับตำแหน่งของแต่ละคน", "th", PROSE) == "not a sentence"
    assert d.reject("พ.ศ. 2546 จัดทำป้อมยามใหม่และปรับปรุงซ่อมแซมพื้นหอประชุมของโรงเรียน", "th", PROSE) == "not a sentence"
    assert d.reject("สมชาย ใจดี (เกิด 1 มกราคม พ.ศ. 2500) นักแสดงและนักร้องชาวไทยจากจังหวัดเชียงใหม่", "th", PROSE) == "not a sentence"  # no verb outside the brackets
    news = d.Source("news", "News", "news", "CC0 1.0", "", "", "", None, None, prose=True, reported=True)
    assert d.reject("ในแถลงการณ์เมื่อเดือนสิงหาคม กระทรวงต่างประเทศจีนระบุว่า รัฐบาลอเมริกัน", "th", news) == "unfinished"
    assert d.reject("กระทรวงต่างประเทศจีนระบุว่า รัฐบาลอเมริกันควรยกเลิกมาตรการภาษีนำเข้าสินค้าทั้งหมดโดยเร็ว", "th", news) is None
    para = d.Source("para", "Para", "news", "CC0 1.0", "", "", "", None, None, prose=True, **d.PARAGRAPH)
    long_row = ("รัฐบาลประกาศมาตรการใหม่เพื่อช่วยเหลือเกษตรกรที่ได้รับผลกระทบจากภัยแล้งในปีนี้ โดยกระทรวงการคลังจะจัดสรรเงินกู้ดอกเบี้ยต่ำผ่านธนาคารของรัฐ "
                "ขณะที่กรมชลประทานเร่งขุดลอกคลองส่งน้ำในพื้นที่ภาคตะวันออกเฉียงเหนือ และหน่วยงานท้องถิ่นได้รับมอบหมายให้สำรวจความเสียหายของพืชผลภายในสิ้นเดือน")
    assert d.stutters("เขาเดินทางไปยังเมืองหลวง เดินทางไปยังเมืองหลวงในปีถัดมา") and not d.stutters("มหาวิทยาลัยเชียงใหม่ก่อตั้งขึ้นก่อน และต่อมามหาวิทยาลัยเชียงใหม่ได้ขยายวิทยาเขต")
    assert d.reject(long_row.strip(), "th", PROSE) == "too long" and d.reject(long_row.strip(), "th", para) is None


def test_templates_are_capped_and_customers_identities_dropped():
    assert d.frame("I need a flight to Boston on March 3rd at 5 pm") == d.frame("I need a flight to Denver on June 21st at 10 am")
    carried = d.Source("abcd", "A", "chat", "MIT", "", "", "", None, None, dialogue=True, frames=True, carried=True)
    counts = collections.Counter()
    batch = [("a", "I would like to return the boots I bought.", "en", None),  # first turn: no line before to carry
             ("b", "It is the one from last week I think.", "en", "What is the address on the order?"),
             ("c", "The boots are too small for my feet.", "en", "x" * 300),  # the line before cannot be carried
             ("e", "They were a gift for my sister.", "en", "Why are you returning them?")]
    kept = d.sift(carried, batch, lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), counts)
    assert [r["id"] for r in kept] == ["a", "e"] and counts == {"contact details": 1, "needs the turn before": 1}
    stubs = d.Source("wiki", "W", "e", "CC0 1.0", "", "", "", None, None, prose=True, stubs=True)
    lead = "Bob Kane (October 24, 1915 – November 3, 1998) was an American cartoonist."
    assert sum(d.reject(lead, "en", stubs, f"id-{i}") is None for i in range(400)) in range(70, 130)  # one in four stays



def test_thai_wiki_text_keeps_no_fused_sentences_holes_list_items_or_marked_articles(raw):
    page = """{{กล่องข้อมูล|ชื่อ=ตัวอย่าง}}
'''เมืองตัวอย่าง''' ({{langx|en|Sample City}}) เป็นเมืองหลวงของประเทศ<ref>อ้างอิง</ref>ตั้งอยู่ริมแม่น้ำ
และมีประชากรราวหนึ่งล้านคน

พลังงานของระบบเท่ากับ <math>E = mc^2</math> ตามทฤษฎีนี้

เมืองนี้แบ่งการปกครองออกเป็นสามเขต ได้แก่

เขตเหนือเป็นเขตที่ใหญ่ที่สุดของเมือง

== ตัวละคร ==
;สมชาย
พระเอกของเรื่องซึ่งเป็นตำรวจหนุ่ม

ย่อหน้าปกติที่อยู่หลังรายการตัวละครของเรื่องนี้
"""
    paragraphs = d.wiki_paragraphs(page, careful=True)
    assert paragraphs[0] == "เมืองตัวอย่าง (Sample City) เป็นเมืองหลวงของประเทศ ตั้งอยู่ริมแม่น้ำ และมีประชากรราวหนึ่งล้านคน"  # a space where the reference and the line break were
    assert d.HOLE in paragraphs[1] and paragraphs[2:] == ["ย่อหน้าปกติที่อยู่หลังรายการตัวละครของเรื่องนี้"]  # the list lead-in, its item and the glossary entry go
    assert [r[1] for r in d.thai_rows("x", paragraphs)] == [paragraphs[0], paragraphs[2]] and list(d.thai_rows("x", paragraphs))[1][3] is None  # a holed sentence is no row and no context
    assert d.WIKI_MAINTENANCE.search("{{ปรับภาษา}}\nข้อความ") and not d.WIKI_MAINTENANCE.search("{{ต้องการอ้างอิง}}\nข้อความ")
    assert d.reject("หลังสำเร็จการศึกษาระดับปริญญาเอกแล้วหวังเป็นนักวิจัยหลังปริญญาเอกที่ในพรินซ์ตันเป็นเวลาสองปี", "th", PROSE) == "not a sentence"
    assert d.reject("นายแพทย์แอนโธนี เฟาชี่ กล่าวต่อรายการ ของสถานีโทรทัศน์เอบีซีว่าตนจะไม่แปลกใจเลย", "th", PROSE) == "not a sentence"
    first = "นอกจากการยืนยันโดยบัตรยืนยันการติดต่อที่นักวิทยุสมัครเล่นนิยมแลกเปลี่ยนกันแล้ว"
    assert len(d.thai_sentences(f"{first} ปัจจุบันมีการยืนยันแบบอิเล็กทรอนิกส์ด้วยเช่นกัน")) == 1  # no cut after a clause that waits for its main clause
    counts = collections.Counter()
    row = "ลักษณะทางวากยสัมพันธ์ของภาษาไทยโดยรวมแล้วจะเรียงเป็นประธาน กริยา และกรรม"
    batch = [("a", row, "th", "คำคุณศัพท์จะวางไว้หลังคำนาม " + row), ("b", "เมืองนี้มีประชากรราวหนึ่งล้านคนและมีพื้นที่กว้างใหญ่มาก", "th", "เขากล่าวกับผู้สื่อข่าวว่า")]
    kept = d.sift(PROSE, batch, lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), counts)
    assert [(r["id"], r.get("context")) for r in kept] == [("a", None), ("b", None)]  # a context that holds the row, or is cut off, is not carried


def test_a_forward_only_source_marks_every_row_and_the_manifest(tmp_path, monkeypatch):
    marked = d.Source("marked", "M", "encyclopaedic", "CC0 1.0", "", "", "", None, None, use="forward_only")
    plain = d.Source("plain", "P", "chat", "MIT", "", "", "", None, None)
    rows = []
    for source in (marked, plain):
        rows += d.sift(source, [(f"{source.name}-1", f"This is one whole sentence from the {source.name} source.", "en")], lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), collections.Counter())
    assert [r.get("use") for r in rows] == ["forward_only", None]
    monkeypatch.setattr(d, "OUT", tmp_path)
    monkeypatch.setattr(d, "SOURCES", (marked, plain))
    write(tmp_path / "candidates.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    write(tmp_path / "candidates.json", json.dumps({"files": {}, "sources": {"marked": {}, "plain": {}}}))
    d.mix()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    written = [json.loads(line) for line in (tmp_path / "sources.jsonl").read_text().splitlines()]
    assert [r.get("use") for r in written] == ["forward_only", None] and manifest["by_use"] == {"both_directions": 1, "forward_only": 1}
    assert manifest["sources"]["marked"]["use"] == "forward_only" and manifest["sources"]["plain"]["use"] == "both_directions"
    assert {s.name for s in d.SOURCES if s.use} == {"marked"}


def test_thai_wikipedia_is_forward_only_and_under_bar_thai_sources_are_candidates_with_their_scores():
    by_name = {s.name: s for s in d.SOURCES}
    assert by_name["thwiki"].use == "forward_only" and [s.name for s in d.SOURCES if s.use] == ["thwiki"]
    candidates = {s.name for s in d.SOURCES if s.pool == "candidate"}
    assert candidates == {"thwiki_para", "iapp", "yodas", "lotusdis", "commonvoice_th_collector", "opensubtitles_th", "wisesight"}
    assert all(by_name[n].score and by_name[n].take is None for n in candidates)  # kept whole, with the score the reduction can weigh


def test_tidy_pass_rules(raw):
    assert d.GV_ASIDE.search("Photo by Jane Doe, used under a CC BY 2.0 licence and published on the site in March.")
    assert d.GV_ASIDE.search("The protest was covered widely by local bloggers and by the national press (RUS) all week long.")
    assert not d.GV_ASIDE.search("The protest was covered widely by local bloggers and by the national press all week long.")
    assert d.without_values("I want to fly to san francisco please.", {"san francisco"}) == "I want to fly to Xx please."
    assert d.frame("book a table for two on friday at seven") == d.frame("book a table for six on monday at nine")
    named = d.Source("names", "N", "everyday", "MIT", "", "", "", None, None, name_cap=1)
    talk = d.Source("talk", "T", "task dialogue", "MIT", "", "", "", None, None, dialogue=True, frames=True)
    counts = collections.Counter()
    kept = d.sift(named, [("a", "Yanni bought a new car last week.", "en"), ("b", "Yanni sold his old house in May.", "en"), ("c", "The weather was fine all day.", "en"), ("e", "The train left early this morning.", "en")],
                  lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), counts)
    assert [r["id"] for r in kept] == ["a", "c", "e"] and counts == {"repeated name": 1}
    lines = ["I need a cheap hotel in the north.", "Something with free parking would be best.", "A guesthouse would also be fine with me."]
    kept = d.sift(talk, [(str(i), t, "en", "How can I help you today?") for i, t in enumerate(lines)], lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), counts)
    assert len(kept) == d.PREVIOUS_CAP and counts["repeated previous line"] == 1
    first = "พระองค์ทรงเป็นพระราชโอรสองค์ที่สองของกษัตริย์แห่งอาณาจักรทางตอนเหนือ"
    rows = [("a", "เขาเริ่มทำงานเป็นครูสอนภาษาที่โรงเรียนประจำจังหวัดเมื่ออายุได้ยี่สิบปี", "th", None), ("b", "เขาเริ่มทำงานเป็นวิศวกรที่โรงงานผลิตรถยนต์ในกรุงเทพเมื่ออายุได้ยี่สิบสองปี", "th", "สมชายเกิดที่จังหวัดเชียงใหม่ในครอบครัวชาวนา")]
    kept = d.sift(PROSE, rows, lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), counts)
    assert [r["id"] for r in kept] == ["b"] and counts["subject in the row before"] == 1  # "he ..." with no sentence before it


def test_chat_messages_split_mid_sentence_are_joined_and_empty_replies_dropped(raw):
    def message(text, sender):
        return {"text": text, "sender": sender}

    talk = [[message("the best part is that they hold their value so you get a lot more", "a"), message("out of it", "a"), message("Have you seen the new one?", "a")],
            [message("you do huh?", "b")],  # only pronouns and auxiliaries, but it answers a question
            [message("I love rock climbing,", "a"), message("it is the best thing we do all year", "a"), message("it is", "a")],
            [message("that it is", "b"), message("We should go together sometime.", "b")]]
    write(raw / "byu/dataset.json", json.dumps({"c1": {"messages": talk}, "c2": {"messages": [[message("A new conversation starts here.", "c")]]}}))
    counts = collections.Counter()
    rows = list(d.byu(counts=counts))
    assert [r[1] for r in rows] == ["the best part is that they hold their value so you get a lot more out of it", "Have you seen the new one?", "you do huh?",
                                    "I love rock climbing, it is the best thing we do all year", "We should go together sometime.", "A new conversation starts here."]
    assert counts == {"joined": 2, "empty reply": 2}


def test_chat_context_is_the_message_just_before_in_the_same_conversation(raw):
    talk = {"c1": {"messages": [[{"text": "First message of the first chat.", "sender": "a"}, {"text": "Second message of the first chat.", "sender": "a"}],
                                [{"text": "Third message, from the other person.", "sender": "b"}]]},
            "c2": {"messages": [[{"text": "First message of the second chat.", "sender": "c"}], [{"text": "Second message of the second chat.", "sender": "d"}]]}}
    write(raw / "byu/dataset.json", json.dumps(talk))
    rows = list(d.byu())
    assert [r[3] for r in rows] == [None, rows[0][1], rows[1][1], None, rows[3][1]]  # never from another conversation, never skipping one
    assert [r[0] for r in rows] == ["byu-c1-0-0", "byu-c1-0-1", "byu-c1-1-0", "byu-c2-0-0", "byu-c2-1-0"]


def test_every_row_carries_its_sources_tier_and_tier_2_can_be_left_out(tmp_path, monkeypatch):
    clean = d.Source("clean", "C", "news", "CC0 1.0", "https://licence.example", "", "https://home.example", None, None)
    unclear = d.Source("unclear", "U", "reviews", "LGPL-3.0", "https://lgpl.example", "", "", None, None, tier=2, finding="a software licence; the text is users'", read_on="https://read.example")
    rows = []
    for source in (clean, unclear):
        rows += d.sift(source, [(f"{source.name}-1", f"This is one whole sentence from the {source.name} source.", "en")], lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), collections.Counter())
    assert [r["tier"] for r in rows] == [1, 2]
    monkeypatch.setattr(d, "OUT", tmp_path)
    monkeypatch.setattr(d, "SOURCES", (clean, unclear))
    write(tmp_path / "candidates.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    write(tmp_path / "candidates.json", json.dumps({"files": {}, "sources": {"clean": {}, "unclear": {}}}))
    d.mix()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["by_tier"] == {"1": 1, "2": 1} and manifest["tiers_included"] == [1, 2]
    assert manifest["sources"]["unclear"] == manifest["sources"]["unclear"] | {"tier": 2, "licence_finding": "a software licence; the text is users'", "licence_read_on": "https://read.example"}
    assert manifest["sources"]["clean"] == manifest["sources"]["clean"] | {"tier": 1, "licence_finding": "CC0 1.0", "licence_read_on": "https://licence.example"}
    d.mix(tiers=(1,))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [json.loads(line)["source"] for line in (tmp_path / "sources.jsonl").read_text().splitlines()] == ["clean"]
    assert manifest["by_tier"] == {"1": 1} and manifest["sources"]["unclear"]["in_mix"] == 0 and manifest["tiers_included"] == [1]


def test_new_thai_source_formats(raw):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    def table(path, rows):
        (raw / path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), raw / path)

    wrapped = "จำเลยให้การปฏิเสธว่าไม่เคยทำสัญญากู้\nและไม่เคยรับเงินจากโจทก์ (อันดับ 12 แผ่นที่ 3)\nจึงไม่ต้องรับผิดตามฟ้อง"  # typed with a break every few words
    table("court/2540.parquet", [
        {"docid": "1", "short_text": "ย่อของศาลเอง โจทก์ฟ้องขอให้บังคับจำเลยชำระหนี้ตามสัญญากู้ยืมเงิน", "source": "กองผู้ช่วยผู้พิพากษาศาลฎีกา", "litigants": "[{'role': 'โจทก์', 'name': 'นายสมชาติ ใจมั่น กับพวก'}]",
         "long_text": wrapped},
        {"docid": "2", "short_text": "ย่อของเนติบัณฑิตยสภา ไม่ใช่ของศาล", "long_text": "ศาลชั้นต้นพิจารณาแล้วเห็นว่าโจทก์มีอำนาจฟ้อง", "source": "เนติบัณฑิตยสภา", "litigants": "[]"},
        {"docid": "3", "short_text": "", "source": "กองผู้ช่วยผู้พิพากษาศาลฎีกา", "litigants": "[{'role': 'โจทก์', 'name': 'นายสมชาติ ใจมั่น'}]",
         "long_text": "พิพากษายืน\nนายวิชัย อ่อนถาวร เบิกความว่าเห็นเหตุการณ์\nพยานยืนยันว่าสมชาติเป็นผู้ลงลายมือชื่อในสัญญา " + "ซึ่งทำขึ้นที่สำนักงานของโจทก์ " * 6 + "\nที่ดินโฉนดเลขที่ 1234 เป็นของโจทก์\nเด็กหญิง ก. เป็นผู้เสียหาย"}])
    counts = collections.Counter()
    assert [(r[0], r[1]) for r in d.supreme_court(counts=counts)] == [("court-1-0", "ย่อของศาลเอง โจทก์ฟ้องขอให้บังคับจำเลยชำระหนี้ตามสัญญากู้ยืมเงิน"),
                                                                    ("court-1-1", "จำเลยให้การปฏิเสธว่าไม่เคยทำสัญญากู้ และไม่เคยรับเงินจากโจทก์ จึงไม่ต้องรับผิดตามฟ้อง")]  # wrapped lines are one paragraph, without the docket reference
    assert counts == {"kept": 2, "closing order": 1, "names a person": 3, "address, deed or company": 1, "Thai Bar record": 1}
    law = "ประกาศกระทรวงตัวอย่าง\n\nข้อ ๑ ประกาศนี้ให้ใช้บังคับตั้งแต่วันถัดจากวันประกาศ\n\nข้อ ๒ ผู้ประกอบการต้องยื่นคำขอต่อนายทะเบียน\n\nประกาศ ณ วันที่ ๑ มกราคม พ.ศ. ๒๕๖๐\n\nรัฐมนตรีว่าการ\n\nมาตรา ๕ ไม่ใช่ข้อ"
    table("thailaw/0.parquet", [{"title": "t", "text": law, "__index_level_0__": 3}])
    assert list(d.thai_law("thailawc", d.CLAUSE)) == [("thailawc-3-2", "ผู้ประกอบการต้องยื่นคำขอต่อนายทะเบียน", "th")]  # stops at the signature block
    table("wangchan/train.parquet", [{"ID": "F_1", "Instruction": "ทำไมธนาคารกลางจึงขึ้นอัตราดอกเบี้ย", "Input": "None", "License": "cc-by-sa-4.0"},
                                     {"ID": "F_2", "Instruction": "สรุปบทความนี้ให้หน่อย", "Input": "บทความ", "License": "cc-by-sa-4.0"},
                                     {"ID": "F_3", "Instruction": "คำถามจากแถวที่ห้ามใช้เชิงพาณิชย์", "Input": "None", "License": "cc-by-nc-4.0"}])
    assert list(d.wangchan()) == [("wangchan-F_1", "ทำไมธนาคารกลางจึงขึ้นอัตราดอกเบี้ย", "th")]
    for path, members in (("truevoice/mari-intent.zip", {"mari_train.csv": "texts,action\nอยากทราบว่าเหมาโทร 90 บาท 7 วัน รหัสอะไรคะ,enquire\nเค้าเช็ต . บาท เมื่อวานนี้,enquire\n", "mari_test.csv": "texts,action\n"}),
                          ("wongnai/review_dataset.zip", {"w_review_train.csv": '"ร้านนี้อร่อยมากเลยค่ะ บริการก็ดีมาก\n- ต้มยำกุ้ง 120 บาท\nอาหารมาช้าไปหน่อยแต่รสชาติใช้ได้เลย ??";5\n', "test_file.csv": 'reviewID;review\n'})):
        (raw / path).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw / path, "w") as archive:
            for name, text in members.items():
                archive.writestr(name, text)
    assert [r[1] for r in d.truevoice()] == ["อยากทราบว่าเหมาโทร 90 บาท 7 วัน รหัสอะไรคะ"]  # the row with a blanked number goes
    assert [r[1] for r in d.wongnai()] == ["ร้านนี้อร่อยมากเลยค่ะ บริการก็ดีมาก"]  # the dish line and the line with lost emoji go


def test_subtitles_keep_the_film_in_the_id_and_carry_the_line_before_when_it_is_close(raw, tmp_path, monkeypatch):
    def line(n, start, end, text):
        return f'<s id="{n}"><time id="T{n}S" value="{start}" />{text}<time id="T{n}E" value="{end}" /></s>'

    film = "<document>" + line(1, "00:00:01,000", "00:00:03,000", "- วันนี้เราจะไปไหนกันดี") + line(2, "00:00:03,500", "00:00:05,000", "[เสียงดนตรี] ไปทะเลกันเถอะ")
    film += line(3, "00:01:00,000", "00:01:02,000", "ผ่านไปหนึ่งนาทีแล้วนะ") + line(4, "00:01:02,500", "00:01:04,000", "Subtitles by somebody")
    film += line(5, "00:02:00,000", "00:02:02,000", "ถ้าฝนไม่ตกนะ...") + line(6, "00:02:02,500", "00:02:04,000", "เราจะไปเดินเล่นกัน") + line(7, "00:02:30,000", "00:02:31,000", "ฮิสะคุง!")
    film += line(8, "00:02:07,000", "00:02:08,000", "คุณแม่ คังโฮ") + line(9, "00:02:09,000", "00:02:10,000", "อะไร/กูเอาอักษรตัวแรกมา") + "</document>"
    (raw / "opensubtitles").mkdir()
    with zipfile.ZipFile(raw / "opensubtitles/th.zip", "w") as archive:
        archive.writestr("OpenSubtitles/raw/th/1999/331581/4758707.xml", film)
        archive.writestr("README", "not a subtitle file")
    monkeypatch.setattr(d, "FILM_LINES", collections.defaultdict(lambda: [0, 0, 0]))
    assert list(d.opensubtitles()) == [("opensubtitles-331581-4758707-1", "วันนี้เราจะไปไหนกันดี", "th", None), ("opensubtitles-331581-4758707-2", "ไปทะเลกันเถอะ", "th", "วันนี้เราจะไปไหนกันดี"),
                                       ("opensubtitles-331581-4758707-3", "ผ่านไปหนึ่งนาทีแล้วนะ", "th", None),  # the tag, the dash and the credit go; a minute's gap ends the context
                                       ("opensubtitles-331581-4758707-5", "ถ้าฝนไม่ตกนะ เราจะไปเดินเล่นกัน", "th", None)]  # a sentence split over two screens is one row
    assert dict(d.FILM_LINES) == {"331581": [9, 1, 0]}  # the two lines that only call somebody and the two speakers on one line go
    (raw / "wisesight").mkdir()
    write(raw / "wisesight/pos.txt", "นิสสันบายพาส\nรถคันนี้ขับดีมากเลยครับ ประหยัดน้ำมันด้วย\n“พรรคหนึ่ง” ชวนติดแฮชแท็ก “อย่าเลื่อน” ด้าน “อีกคน” ชี้ว่าไม่เป็นไร\n")
    assert [r[1].strip() for r in d.wisesight()] == ["รถคันนี้ขับดีมากเลยครับ ประหยัดน้ำมันด้วย"]  # the product name and the headline go
    pooled = d.Source("pooled", "P", "reviews", "MIT", "", "", "", None, None, pool="candidate")
    rows = d.sift(pooled, [("p-1", "This is one whole sentence from a source under the bar.", "en")], lambda texts: [5] * len(texts), set(), set(), set(), collections.Counter(), collections.Counter())
    assert rows[0]["pool"] == "candidate"
    monkeypatch.setattr(d, "OUT", tmp_path)
    monkeypatch.setattr(d, "SOURCES", (pooled, d.Source("opensubtitles_th", "O", "film", "Unstated", "", "", "", None, None, tier=2)))
    rows += [{"id": "opensubtitles-331581-4758707-1", "text": "วันนี้เราจะไปไหนกันดี", "lang": "th", "source": "opensubtitles_th", "register": "film", "licence": "Unstated", "tier": 2, "pieces": 6}]
    write(tmp_path / "candidates.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    write(tmp_path / "candidates.json", json.dumps({"files": {}, "sources": {"pooled": {}, "opensubtitles_th": {}}}))
    write(tmp_path / "films_read.json", json.dumps({"films": {"331581": [4, 1, 0], "999": [8, 0, 2]}, "files": {"331581/4758707": [4, 2, 0], "999/1": [8, 0, 4]}, "reader": {"lyric lines": 3}}))
    d.mix()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["by_pool"] == {"candidate": 1, "passed": 1} and manifest["films"] == {"file": "films.json", "films": 2, "with_rows": 1, "subtitle_files": 2, "reader": {"lyric lines": 3}}
    assert json.loads((tmp_path / "films.json").read_text()) == {"331581": {"read": 4, "in_mix": 1, "dropped_share": 0.5, "joined_share": 0.25, "mixed_register_lines": 0,
                                                                            "files": {"4758707": {"thai_lines": 4, "particle_rate": 0.5, "possessive_rate": 0.0}}},
                                                                 "999": {"read": 8, "in_mix": 0, "dropped_share": 1.0, "joined_share": 0.0, "mixed_register_lines": 2,
                                                                         "files": {"1": {"thai_lines": 8, "particle_rate": 0.0, "possessive_rate": 0.5}}}}


def test_subtitles_are_repaired_and_never_joined_across_speakers_or_songs(raw, monkeypatch):
    def film(*lines):  # each line shows for a second, half a second after the one before
        return "<document>" + "".join(f'<s id="{n}"><time value="00:00:{n * 2:02},000" />{text}<time value="00:00:{n * 2 + 1:02},500" /></s>' for n, text in enumerate(lines, 1)) + "</document>"

    spoken = film("♪ เพลงรักของเราสองคน", "ยังดังอยู่ในใจฉันเสมอ", "ตลอดไปไม่มีวันจาง ♪",  # a song: its first line, its last, and the one between
                  "คุณจะไปตลาดกับฉันไหมคะ", "แล้วผมจะรออยู่ที่บ้านครับ",  # a finished question, and a man answering a woman: two rows
                  "เมื่อวานฉันเห็นเขาที่ร้านกาแฟ,", "- แล้วเขาพูดอะไรกับเธอบ้าง",  # the dash marks another speaker
                  "ไซโต้ เคียวอิจิ", "ฉันอยากบอกเธอว่า...", "- วันนี้อากาศดีมากเลย")  # a half sentence and a bare name go, and are nobody's context
    (raw / "opensubtitles").mkdir()
    with zipfile.ZipFile(raw / "opensubtitles/th.zip", "w") as archive:
        archive.writestr("OpenSubtitles/raw/th/2001/1/1.xml", spoken)
        archive.writestr("OpenSubtitles/raw/th/2001/2/2.xml", film(*["วันนี้เราจะไปไหนกันดี".encode("utf8").decode("cp874", "replace").replace("\ufffd", "\x81")] * 3))  # ponytail: one undefined byte stands in for all
        archive.writestr("OpenSubtitles/raw/th/2001/3/3.xml", film(*["เธเธฃเธฑเธเธขเน"] * 3))  # scrambled past repair
    for name in ("FILM_LINES", "FILE_SIGNS", "SUBTITLE_DROPS"):
        monkeypatch.setattr(d, name, type(getattr(d, name))(getattr(d, name).default_factory) if name == "FILM_LINES" else type(getattr(d, name))())
    rows = list(d.opensubtitles())
    assert [(r[1], r[3]) for r in rows if "-1-1-" in r[0]] == [("คุณจะไปตลาดกับฉันไหมคะ", None), ("แล้วผมจะรออยู่ที่บ้านครับ", "คุณจะไปตลาดกับฉันไหมคะ"),
                                                              ("เมื่อวานฉันเห็นเขาที่ร้านกาแฟ,", "แล้วผมจะรออยู่ที่บ้านครับ"), ("แล้วเขาพูดอะไรกับเธอบ้าง", "เมื่อวานฉันเห็นเขาที่ร้านกาแฟ,"),
                                                              ("วันนี้อากาศดีมากเลย", None)]
    assert d.SUBTITLE_DROPS == {"lyric lines": 3, "joins refused at a change of speaker": 2, "half sentences": 1, "bare names and noun phrases": 1,
                                "files in the wrong encoding, read properly": 1, "files in the wrong encoding, left out": 1}
    assert d.FILE_SIGNS["1/1"] == [10, 3, 1] and "3/3" not in d.FILE_SIGNS  # ten Thai lines, three with a particle of speech, one possessive


def test_tier_2_sources_come_after_every_tier_1_source_that_is_taken():
    order = [(s.tier, s.pool == "candidate") for s in d.SOURCES]
    assert order == sorted(order)  # so no row is dropped as the duplicate of a tier 2 or candidate row that may later be removed


def test_reduction_keeps_one_of_each_look_alike_and_drops_what_should_not_be_translated(tmp_path, monkeypatch):
    passed = d.Source("passed", "P", "news", "CC0 1.0", "", "", "", None, None)
    unclear = d.Source("opensubtitles_th", "O", "film", "Unstated", "", "", "", None, None, tier=2, pool="candidate")
    held = d.Source("held", "H", "captions", "CC0 1.0", "", "", "", None, None, pool="candidate")
    monkeypatch.setattr(d, "OUT", tmp_path)
    monkeypatch.setattr(d, "SOURCES", (passed, held, unclear))

    def row(source, n, text, lang="th", **more):
        return {"id": f"{source.name}-{n}-0-{n}" if source is unclear else f"{source.name}-{n}", "text": text, "lang": lang, "source": source.name, "register": source.register,
                "licence": source.licence} | more | {"tier": source.tier}

    rows = [row(passed, 1, "วันนี้อากาศดีมากเลยนะครับ"),  # kept: tier 1 beats the tier 2 look-alike below
            row(passed, 2, "The meeting starts at 10 o'clock in room 12.", "en"),
            row(passed, 3, "The meeting starts at 11 o'clock in Room 34!", "en"),  # the same but for digits, case and punctuation
            row(passed, 4, "<i>เราไปกินข้าวด้วยกันไหม</i>"), row(passed, 5, "ตลกมาก 55555555 จริงๆ เลยนะเนี่ย"), row(passed, 6, "ใช่ค่ะ"),  # markup, repeated characters, too short
            row(held, 1, "แถวนี้มาจากแหล่งที่ยังไม่ผ่านเกณฑ์คุณภาพ"),
            row(unclear, 1, "วันนี้อากาศดีมากเลยครับ"),  # a look-alike of the first row, in the same politeness
            row(unclear, 2, "วันนี้อากาศดีมากเลยค่ะ"), row(unclear, 3, "วันนี้อากาศดีมาก"),  # the other two politenesses stay
            row(unclear, 4, "เราไปกินข้าวด้วยกันไหม"), row(unclear, 5, "เราไปกินข้าวด้วยกันไหม", context="หิวหรือยัง"),  # the one with context is kept
            row(unclear, 6, "ใช่ค่ะ", context="คุณมาจากกรุงเทพหรือ"), row(unclear, 7, "ใช่ครับ", context="พร้อมหรือยัง"),  # short replies stay with their context
            row(unclear, 999, "แถวนี้มาจากหนังที่คำบรรยายเสียเกือบทั้งเรื่อง"), row(unclear, 8, "ฉันขอให้เรือเดินสมุทรที่บันทึกไว้ทั้งหมดของคุณ")]
    write(tmp_path / "sources.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write(tmp_path / "manifest.json", json.dumps({"rows": len(rows), "sha256": "x", "sources": {"passed": {"in_mix": 6}, "held": {"in_mix": 1}, "opensubtitles_th": {"in_mix": 9}}}))
    write(tmp_path / "films.json", json.dumps({"999": {"read": 100, "in_mix": 1, "dropped_share": 0.9, "joined_share": 0, "mixed_register_lines": 0},
                                               "1": {"read": 100, "in_mix": 60, "dropped_share": 0.3, "joined_share": 0.1, "mixed_register_lines": 0,
                                                     "files": {"0": {"thai_lines": 90, "particle_rate": 0.34, "possessive_rate": 0.03}}},
                                               "8": {"read": 100, "in_mix": 60, "dropped_share": 0.3, "joined_share": 0.1, "mixed_register_lines": 0,  # no particles of speech, every possessive kept
                                                     "files": {"0": {"thai_lines": 90, "particle_rate": 0.1, "possessive_rate": 0.09}}}}))
    monkeypatch.setattr(d, "SHORT_KEEP", 1)
    d.reduce()
    first = (tmp_path / "distill_set.jsonl").read_bytes()
    kept = [json.loads(line)["id"] for line in first.decode().splitlines()]
    assert kept == ["passed-1", "passed-2", "opensubtitles_th-2-0-2", "opensubtitles_th-3-0-3", "opensubtitles_th-5-0-5", "opensubtitles_th-6-0-6"]
    record = json.loads((tmp_path / "distill_set.json").read_text())
    assert {rule: counts["all"] for rule, counts in record["removed"].items() if counts["all"]} == {
        "candidate source not admitted": 1, "film in the worst band": 1, "machine-translated subtitle file": 1, "leftover markup": 1, "repeated characters": 1, "too short": 1,
        "short reply over the cap": 1, "look-alike duplicate": 3}
    assert record["rows"] == 6 and record["sources"]["opensubtitles_th"] == {"pool": 9, "kept": 4} and record["subtitle_files_left_out_as_machine_translated"] == 1 and record["look_alikes"]["kept_as_politeness_variants"] == 2
    assert record["look_alikes"]["groups"] == 3 and record["films_left_out"] == 1 and record["pool"]["rows"] == len(rows)
    assert (tmp_path / "sources.jsonl").read_text().count("\n") == len(rows)  # the pool is not touched
    d.reduce()
    assert (tmp_path / "distill_set.jsonl").read_bytes() == first  # the same again
    d.reduce(tiers=(1,))
    assert [json.loads(line)["id"] for line in (tmp_path / "distill_set.jsonl").read_text().splitlines()] == ["passed-1", "passed-2"]
