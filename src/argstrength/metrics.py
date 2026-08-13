"""Tie-aware ranking, pairwise, and reliability metrics."""

from __future__ import annotations

from itertools import combinations
import math
import statistics
from typing import Sequence


def descending_average_ranks(scores: Sequence[float]) -> list[float]:
    """Rank high scores first and assign average ranks to ties."""

    if not scores:
        raise ValueError("scores cannot be empty")
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    ranks = [0.0] * len(scores)
    position = 1
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        average_rank = (position + (position + end - cursor - 1)) / 2
        for index in order[cursor:end]:
            ranks[index] = average_rank
        position += end - cursor
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("vectors must have equal length of at least two")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_dev = [value - left_mean for value in left]
    right_dev = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_dev)
        * sum(value * value for value in right_dev)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_dev, right_dev)) / denominator


def spearman_from_scores(
    reference_ranks: Sequence[int], scores: Sequence[float]
) -> float | None:
    if len(reference_ranks) != len(scores):
        raise ValueError("reference_ranks and scores must have equal length")
    return _pearson(
        [float(rank) for rank in reference_ranks], descending_average_ranks(scores)
    )


def _comparable_pairs(reference_ranks: Sequence[int], scores: Sequence[float]):
    if len(reference_ranks) != len(scores):
        raise ValueError("reference_ranks and scores must have equal length")
    for left, right in combinations(range(len(scores)), 2):
        if reference_ranks[left] == reference_ranks[right]:
            continue
        yield (
            (left, right)
            if reference_ranks[left] < reference_ranks[right]
            else (right, left)
        )


def pairwise_accuracy(
    reference_ranks: Sequence[int],
    scores: Sequence[float],
    *,
    tie_credit: float = 0.5,
) -> float:
    """Fraction of reference pair preferences recovered by scores."""

    if not 0 <= tie_credit <= 1:
        raise ValueError("tie_credit must be in [0, 1]")
    credits = []
    for preferred, other in _comparable_pairs(reference_ranks, scores):
        if scores[preferred] > scores[other]:
            credits.append(1.0)
        elif scores[preferred] == scores[other]:
            credits.append(tie_credit)
        else:
            credits.append(0.0)
    if not credits:
        raise ValueError("no comparable reference pairs")
    return statistics.fmean(credits)


def strict_pairwise_accuracy(
    reference_ranks: Sequence[int],
    scores: Sequence[float],
    *,
    margin: float = 10.0,
) -> float:
    """Correct-pair rate requiring the preferred score to lead by ``margin``."""

    if margin < 0:
        raise ValueError("margin must be non-negative")
    outcomes = [
        float(scores[preferred] - scores[other] >= margin)
        for preferred, other in _comparable_pairs(reference_ranks, scores)
    ]
    if not outcomes:
        raise ValueError("no comparable reference pairs")
    return statistics.fmean(outcomes)


def pairwise_tie_rate(scores: Sequence[float]) -> float:
    pairs = list(combinations(scores, 2))
    if not pairs:
        raise ValueError("at least two scores are required")
    return sum(left == right for left, right in pairs) / len(pairs)


def top1_accuracy(reference_ranks: Sequence[int], scores: Sequence[float]) -> float:
    if len(reference_ranks) != len(scores) or not scores:
        raise ValueError("reference_ranks and scores must be non-empty and equal")
    best_reference = min(reference_ranks)
    best_score = max(scores)
    return float(
        any(
            reference_ranks[index] == best_reference and score == best_score
            for index, score in enumerate(scores)
        )
    )


def exact_order_accuracy(reference_ranks: Sequence[int], scores: Sequence[float]) -> float:
    if len(reference_ranks) != len(scores) or not scores:
        raise ValueError("reference_ranks and scores must be non-empty and equal")
    predicted = descending_average_ranks(scores)
    return float(all(float(ref) == pred for ref, pred in zip(reference_ranks, predicted)))


def claim_metrics(
    reference_ranks: Sequence[int],
    scores: Sequence[float],
    *,
    strict_margin: float = 10.0,
) -> dict[str, float | None]:
    return {
        "spearman": spearman_from_scores(reference_ranks, scores),
        "pairwise_accuracy": pairwise_accuracy(reference_ranks, scores),
        "strict_pairwise_accuracy": strict_pairwise_accuracy(
            reference_ranks, scores, margin=strict_margin
        ),
        "tie_rate": pairwise_tie_rate(scores),
        "top1_accuracy": top1_accuracy(reference_ranks, scores),
        "exact_order_accuracy": exact_order_accuracy(reference_ranks, scores),
        "score_range": max(scores) - min(scores),
    }
