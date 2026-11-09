from train.compare_adapter import compare, majority, one_sided


def test_majority_ignores_failed_calls_and_leaves_ties_unjudged():
    assert majority([True, True, False]) is True
    assert majority([False, None, False]) is False
    assert majority([True, False, None]) is None
    assert majority([None, None, None]) is None


def test_only_discordant_pairs_count():
    base = {"a": False, "b": True, "c": True, "d": None}
    adapter = {"a": True, "b": False, "c": True, "d": True}
    out = compare(base, adapter, list(base))
    assert out["pairs"] == 3  # the unjudged case is left out
    assert (out["fixed_by_adapter"], out["broken_by_adapter"]) == (1, 1)
    assert (out["base_wrong"], out["adapter_wrong"]) == (1, 1)


def test_one_sided_matches_the_rollout_gate():
    assert one_sided(7, 0) == 1 / 128
    assert one_sided(0, 0) == 1.0
