"""The scrub: details become placeholders on both sides, and unsafe or lopsided pairs are dropped."""
from train.scrub import replace_details, scrub


def pair(src, dst, thread="t"):
    return {"thread": thread, "src": src, "dst": dst, "dir": "x"}


def test_each_kind_of_detail_gets_its_placeholder():
    cases = {
        "mail me at nok.s+1@example.co.th ok": "mail me at <email> ok",
        "see https://example.com/a?b=1 and www.x.org": "see <url> and <url>",
        "my site is nokbar.com/menu": "my site is <url>",
        "add @nok_99 please": "add <handle> please",
        "line id: nok99": "line id: <id>",
        "ไลน์ nok99 นะ": "ไลน์ <id> นะ",
        "call 081-234-5678 or +66 81 234 5678": "call <phone> or <phone>",
        "โทร ๐๘๑๒๓๔๕๖๗๘": "โทร <phone>",
        "account 12345678901234": "account <id>",
        "ref 1234567": "ref <id>",
        "room 1204, soi 8": "room <address>, soi <address>",
        "บ้านเลขที่ 99/12 ซอย 5": "บ้านเลขที่ <address> ซอย <address>",
    }
    for text, expected in cases.items():
        assert replace_details(text) == expected, text


def test_ordinary_numbers_dates_and_words_are_left_alone():
    for text in ("2,000 baht or 1500-2000", "see you 20/11/2026 at 10.30", "line up at 9", "I'd like 3 beers", "100000 baht", "ซอยนี้เงียบดี"):
        assert replace_details(text) == text, text


def test_a_name_someone_gives_is_replaced_everywhere_in_that_conversation():
    rows = [pair("my name is John", "ผมชื่อ John"), pair("John is here", "John มาแล้ว"), pair("hello again", "สวัสดีอีกครั้ง"),
            pair("John is here", "John มาแล้ว", thread="other")]
    kept, counts = scrub(rows)
    assert [r["src"] for r in kept] == ["my name is <name>", "<name> is here", "hello again", "John is here"]
    assert kept[1]["dst"] == "<name> มาแล้ว" and counts["<name>"] == 2


def test_names_from_the_export_are_replaced_in_either_script():
    rows = [pair("Nok went home", "นกกลับบ้านแล้ว"), pair("where is nok?", "นกอยู่ไหน")]
    kept, _ = scrub(rows, names={"t": ["Nok", "นก"]})
    assert [(r["src"], r["dst"]) for r in kept] == [("<name> went home", "<name>กลับบ้านแล้ว"), ("where is <name>?", "<name>อยู่ไหน")]


def test_a_pair_is_dropped_when_only_one_side_was_caught():
    rows = [pair("my name is John", "ผมชื่อ John"), pair("John is here", "จอห์นมาแล้ว"), pair("call 0812345678", "โทรมานะ"), pair("ok", "โอเค")]
    kept, counts = scrub(rows)
    assert [r["src"] for r in kept] == ["my name is <name>", "ok"] and counts["dropped_mismatch"] == 2


def test_a_thai_name_given_in_thai_is_caught_on_both_sides():
    kept, _ = scrub([pair("my name is John", "ผมชื่อจอห์น"), pair("John is here", "จอห์นมาแล้ว")])
    assert [(r["src"], r["dst"]) for r in kept] == [("my name is <name>", "ผมชื่อ<name>"), ("<name> is here", "<name>มาแล้ว")]


def test_explicit_messages_are_dropped_whichever_side_shows_it():
    rows = [pair("send nudes", "ส่งรูปมา"), pair("ok", "เงี่ยน"), pair("you look sexy", "คุณดูเซ็กซี่"), pair("ok", "โอเค", thread="u")]
    kept, counts = scrub(rows)
    assert [r["src"] for r in kept] == ["you look sexy", "ok"] and counts["dropped_explicit"] == 2


def test_nothing_is_left_for_a_second_pass_and_other_fields_survive():
    kept, counts = scrub([pair("mail a@b.com, call 0812345678", "เมล a@b.com โทร 0812345678")])
    assert counts["residual"] == 0 and counts["written"] == 1 and counts["<email>"] == 1 and counts["<phone>"] == 1
    assert kept[0]["dir"] == "x" and kept[0]["thread"] == "t"
