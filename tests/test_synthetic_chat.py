from train.synthetic_chat import row_of, usable

CASE = {"context": ["A: a", "B: b", "A: c", "B: d"], "source": "Are you coming?", "meaning": "Asks if they will come.",
        "translation": "มาไหม"}


def test_usable_refuses_suite_messages_wrong_language_and_bad_shape():
    assert usable(CASE, "en-th", set())
    assert not usable(CASE, "en-th", {"Are you coming?"})
    assert not usable(CASE, "th-en", set())  # a Thai target in the English direction
    assert not usable({**CASE, "context": ["A: a"]}, "en-th", set())
    assert not usable({**CASE, "translation": " "}, "en-th", set())


def test_row_names_the_target_language():
    row = row_of({**CASE, "direction": "en-th", "category": "idiom"})
    assert row["target_lang"] == "Thai" and row["target"] == "มาไหม" and row["context"] == CASE["context"]


def test_row_drops_a_speaker_label_from_the_message():
    assert row_of({**CASE, "source": "B: Are you coming?", "direction": "en-th", "category": "idiom"})["source"] == "Are you coming?"
