from train.score import judge_prompt, summarise


def row(variant, direction, score, leak=False, jev=None):
    out = {"variant": variant, "direction": direction, "score": score, "translated_context": leak}
    if jev is not None:
        out["jev_same"] = jev
    return out


def good():
    rows = []
    for variant in ("none", "correct", "irrelevant"):
        for direction in ("en-th", "th-en"):
            rows += [row(variant, direction, 5) for _ in range(10)]
    return rows


def test_a_clean_run_passes_every_rule():
    result = summarise(good())
    assert result["go"] and result["all_rules_pass"] and result["scored"] == 60


def test_thai_to_english_over_the_bar_is_a_no_go():
    rows = good()
    for r in [r for r in rows if r["variant"] == "correct" and r["direction"] == "th-en"][:2]:
        r["score"] = 2  # 20%, over the 15% bar
    result = summarise(rows)
    assert result["rules"]["2_th_en"] is False and result["go"] is False


def test_translating_the_context_is_a_no_go():
    rows = good()
    next(r for r in rows if r["variant"] == "correct")["translated_context"] = True  # 1 in 20 is over 2%
    assert summarise(rows)["go"] is False


def test_context_may_not_make_things_worse():
    rows = good()
    next(r for r in rows if r["variant"] == "correct")["score"] = 1
    assert summarise(rows)["rules"]["4_context_not_worse"] is False


def test_irrelevant_context_has_a_five_point_allowance():
    rows = good()
    irrelevant = [r for r in rows if r["variant"] == "irrelevant"]
    irrelevant[0]["score"] = 2  # 1 in 20 is exactly 5 points
    assert summarise(rows)["rules"]["5_irrelevant_tolerated"] is True
    irrelevant[1]["score"] = 2
    assert summarise(rows)["rules"]["5_irrelevant_tolerated"] is False


def test_unscored_rows_are_reported():
    result = summarise(good() + [row("none", "en-th", None)])
    assert result["unscored"] == 1


def test_jev_agreement_compares_with_opus():
    rows = [row("none", "en-th", 5, jev=0.9), row("none", "en-th", 2, jev=0.9)]
    assert summarise(rows)["jev_agreement"] == 0.5


def test_the_judge_never_sees_which_system_wrote_it():
    case = {"direction": "en-th", "context": ["A: hi"], "source": "s", "meaning": "m"}
    prompt = judge_prompt(case, "output text")
    assert "Lumen" not in prompt and "A: hi" in prompt and "output text" in prompt
