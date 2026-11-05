import math

import pytest

from evals import stats


def test_per_question_values():
    assert stats.per_question(1) == {"hit@1": 1.0, "hit@5": 1.0, "rr@10": 1.0}
    assert stats.per_question(4) == {"hit@1": 0.0, "hit@5": 1.0, "rr@10": 0.25}
    assert stats.per_question(11) == {"hit@1": 0.0, "hit@5": 0.0, "rr@10": 0.0}  # below the MRR cutoff
    assert stats.per_question(None) == {"hit@1": 0.0, "hit@5": 0.0, "rr@10": 0.0}


def test_bootstrap_ci_of_constant_values_is_the_value():
    assert stats.bootstrap_ci([1.0] * 20) == (1.0, 1.0)
    assert stats.bootstrap_ci([0.5]) == (0.5, 0.5)
    assert all(math.isnan(x) for x in stats.bootstrap_ci([]))


def test_bootstrap_ci_matches_the_normal_approximation_for_a_proportion():
    # 50 of 100: the standard error is 0.05, so a 95% interval is about 0.5 ± 0.098.
    lo, hi = stats.bootstrap_ci([1.0] * 50 + [0.0] * 50)
    assert 0.39 <= lo <= 0.41 and 0.59 <= hi <= 0.61


def test_bootstrap_ci_is_seeded_and_narrows_with_the_level():
    values = [1, 0, 1, 1, 0, 0, 1, 0.5, 0.25, 0]
    assert stats.bootstrap_ci(values, seed=7) == stats.bootstrap_ci(values, seed=7)
    lo95, hi95 = stats.bootstrap_ci(values)
    lo50, hi50 = stats.bootstrap_ci(values, level=0.5)
    assert lo95 < lo50 < sum(values) / len(values) < hi50 < hi95


def test_paired_bootstrap_with_no_difference():
    a = [1, 0, 1, 0.5]
    assert stats.paired_bootstrap(a, a) == {"diff": 0.0, "ci": (0.0, 0.0), "p": 1.0}


def test_paired_bootstrap_with_a_consistent_difference():
    r = stats.paired_bootstrap([0.0] * 30, [1.0] * 30)
    assert r == {"diff": 1.0, "ci": (1.0, 1.0), "p": 1 / (stats.N + 1)}  # (0 + 1) / (n + 1): never exactly 0


def test_paired_bootstrap_p_for_a_few_one_sided_wins():
    # 4 of 66 questions improve and none get worse. A resample has none of the 4 with probability
    # (62/66)^66 ≈ 0.016, and its shifted mean is then as extreme as the observed one; so are the resamples with
    # 8 or more of them. That puts p near 0.06, while the exact sign test gives 2 / 2^4 = 0.125.
    a, b = [0.0] * 66, [1.0] * 4 + [0.0] * 62
    r = stats.paired_bootstrap(a, b)
    assert 0.045 < r["p"] < 0.08 and r["ci"][0] > 0
    assert stats.sign_test(a, b) == {"wins": 4, "losses": 0, "p": 0.125}


def test_sign_test_matches_the_binomial():
    # 10 wins, 3 losses: 2 * (C(13,0) + C(13,1) + C(13,2) + C(13,3)) / 2^13 = 2 * 378 / 8192.
    a, b = [0] * 10 + [1] * 3 + [1] * 50, [1] * 10 + [0] * 3 + [1] * 50
    assert stats.sign_test(a, b) == {"wins": 10, "losses": 3, "p": pytest.approx(756 / 8192)}
    assert stats.sign_test([1, 0], [1, 0]) == {"wins": 0, "losses": 0, "p": 1.0}
    assert stats.sign_test([0, 1], [1, 0])["p"] == 1.0  # one each way: as even as it gets


def test_paired_bootstrap_is_antisymmetric():
    a, b = [1, 0, 0, 1, 1, 0, 0.5, 0], [1, 1, 0, 1, 0, 1, 1, 0]
    ab, ba = stats.paired_bootstrap(a, b), stats.paired_bootstrap(b, a)
    assert ab["diff"] == pytest.approx(-ba["diff"]) and ab["p"] == pytest.approx(ba["p"], abs=0.02)


def test_balanced_wins_and_losses_are_not_significant():
    a = [1] * 5 + [0] * 5 + [1] * 20
    b = [0] * 5 + [1] * 5 + [1] * 20
    r = stats.paired_bootstrap(a, b)
    assert r["diff"] == 0 and r["p"] > 0.9 and r["ci"][0] < 0 < r["ci"][1]


def test_pairing_finds_a_change_the_unpaired_intervals_hide():
    # 60 questions, half of them hard for both configs; B also gets 8 of A's misses right and loses none.
    a = [0.0] * 30 + [1.0] * 30
    b = [1.0] * 8 + [0.0] * 22 + [1.0] * 30
    (alo, ahi), (blo, bhi) = stats.bootstrap_ci(a), stats.bootstrap_ci(b)
    assert blo < ahi  # the separate intervals overlap
    assert stats.paired_bootstrap(a, b)["p"] < 0.01


def test_paired_bootstrap_needs_the_same_questions():
    with pytest.raises(ValueError, match="same questions"):
        stats.paired_bootstrap([1, 0], [1])
    with pytest.raises(ValueError, match="no questions"):
        stats.paired_bootstrap([], [])


def test_intervals_per_metric():
    ci = stats.intervals([1, 1, 3, None])
    assert set(ci) == {"recall@1", "recall@5", "mrr@10"}
    assert ci["recall@1"][0] <= 0.5 <= ci["recall@1"][1] and ci["mrr@10"][0] <= (1 + 1 + 1 / 3) / 4 <= ci["mrr@10"][1]
