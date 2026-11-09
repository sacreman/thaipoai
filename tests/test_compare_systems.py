from evaluation.compare_systems import compare, majority, one_sided, upper_95


def test_majority_ignores_failed_calls_and_leaves_ties_unjudged():
    assert majority([True, True, False]) is True
    assert majority([False, None, False]) is False
    assert majority([True, False, None]) is None
    assert majority([None, None, None]) is None


def test_only_discordant_pairs_count():
    baseline = {"a": False, "b": True, "c": True, "d": None}
    candidate = {"a": True, "b": False, "c": True, "d": True}
    out = compare(baseline, candidate, list(baseline))
    assert out["pairs"] == 3  # the unjudged case is left out
    assert (out["fixed_by_candidate"], out["broken_by_candidate"]) == (1, 1)
    assert (out["baseline_wrong"], out["candidate_wrong"]) == (1, 1)


def test_one_sided_matches_the_rollout_gate():
    assert one_sided(7, 0) == 1 / 128
    assert one_sided(0, 0) == 1.0


def test_upper_bound_on_the_paired_gap():
    assert round(upper_95(10, 0, 100), 3) == 14.935
    assert upper_95(0, 0, 0) is None
