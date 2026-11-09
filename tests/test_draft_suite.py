from evaluation.draft_suite import valid

CASE = {"context": ["A: a", "B: b", "A: c", "B: d"], "irrelevant_context": ["A: e", "B: f", "A: g", "B: h"],
        "source": "Are you coming?", "meaning": "Asks if they will come."}


def test_valid_refuses_repeats_bad_shape_and_the_wrong_script():
    assert valid(CASE, "en-th", "idiom", set(), set())
    assert not valid(CASE, "en-th", "idiom", {"Are you coming?"}, set())
    assert not valid({**CASE, "context": ["A: a"]}, "en-th", "idiom", set(), set())
    assert not valid(CASE, "th-en", "idiom", set(), set())


def test_valid_allows_latin_script_where_the_style_expects_it():
    assert valid(CASE, "th-en", "romanised", set(), {"romanised"})
