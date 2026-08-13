import pytest

from argstrength.metrics import (
    descending_average_ranks,
    exact_order_accuracy,
    pairwise_accuracy,
    pairwise_tie_rate,
    strict_pairwise_accuracy,
    spearman_from_scores,
    top1_accuracy,
)


def test_perfect_order_metrics():
    ranks = [1, 2, 3, 4]
    scores = [90, 70, 40, 10]
    assert spearman_from_scores(ranks, scores) == pytest.approx(1.0)
    assert pairwise_accuracy(ranks, scores) == 1.0
    assert strict_pairwise_accuracy(ranks, scores, margin=10) == 1.0
    assert strict_pairwise_accuracy(ranks, scores, margin=25) == pytest.approx(5 / 6)
    assert top1_accuracy(ranks, scores) == 1.0
    assert exact_order_accuracy(ranks, scores) == 1.0


def test_ties_receive_average_ranks_and_half_pair_credit():
    scores = [90, 90, 40, 10]
    assert descending_average_ranks(scores) == [1.5, 1.5, 3.0, 4.0]
    assert pairwise_accuracy([1, 2, 3, 4], scores) == pytest.approx(5.5 / 6)
    assert pairwise_tie_rate(scores) == pytest.approx(1 / 6)
    assert exact_order_accuracy([1, 2, 3, 4], scores) == 0.0


def test_all_tied_spearman_is_undefined_not_zero():
    assert spearman_from_scores([1, 2, 3, 4], [50, 50, 50, 50]) is None
