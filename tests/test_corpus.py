from train.corpus import SCB_PROFESSIONAL, THAI_GAP, assemble, clean, massive_faithful

STRONG = {"intent_score": 1, "slots_score": 1, "grammar_score": 4, "spelling_score": 2, "language_identification": "target"}
EN = {"annot_utt": "wake me at [time : five]"}
TH = {"annot_utt": "ปลุกฉัน [time : ตีห้า]", "slot_method": [{"slot": "time", "method": "translation"}], "judgments": [STRONG] * 3}


def pair(row_id, en, th, split, source="scb"):
    return {"id": row_id, "en": en, "th": th, "source": source, "subset": "s", "review": "r", "licence": "l", "split": split}


def test_both_scb_spellings_of_a_professional_subset_are_kept():
    assert SCB_PROFESSIONAL["task_master_corpus"] == SCB_PROFESSIONAL["task_master_1"]
    assert SCB_PROFESSIONAL["fake_reviews_hope"] == SCB_PROFESSIONAL["generated_reviews_translator"]
    assert "generated_reviews_yn" not in SCB_PROFESSIONAL and "fake_reviews_yn" not in SCB_PROFESSIONAL


def test_clean_normalises_thai():
    assert clean("นํา ​  ดี") == "นำ ดี"
    assert THAI_GAP.sub("", "ปิด ปลั๊ก at home ได้") == "ปิดปลั๊ก at home ได้"


def test_massive_needs_strong_unanimous_faithful_rows():
    assert massive_faithful(EN, TH)
    assert not massive_faithful(EN, {**TH, "slot_method": [{"slot": "time", "method": "localization"}]})
    assert not massive_faithful(EN, {**TH, "annot_utt": "ปลุกฉัน"})
    assert not massive_faithful(EN, {**TH, "judgments": [STRONG, STRONG, {**STRONG, "grammar_score": 3}]})


def test_training_rows_cannot_share_text_with_held_out_rows_or_suites():
    rows = [pair("a", "Hello there", "สวัสดี", "train"), pair("b", "Hello there", "หวัดดี", "test"),
            pair("c", "Good night", "ราตรีสวัสดิ์", "train"), pair("d", "Good night", "ราตรีสวัสดิ์", "train"),
            pair("e", "A suite line", "ประโยค", "train"), pair("f", "Fine", "ดีมาก", "train")]
    kept, dropped = assemble(rows, frozenset({"a suite line"}))
    assert sorted(r["id"] for r in kept) == ["b", "c", "f"]
    assert dropped["scb", "matches held-out text"] == 1 and dropped["scb", "duplicate"] == 1 and dropped["scb", "matches a suite"] == 1
