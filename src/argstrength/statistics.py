"""Claim-level uncertainty intervals and multiplicity correction."""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Sequence


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI; input units should be independent claims."""

    if not values:
        raise ValueError("values cannot be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        statistics.fmean(rng.choice(values) for _ in range(n))
        for _ in range(n_resamples)
    )
    alpha = 1 - confidence
    lower_index = max(0, int((alpha / 2) * n_resamples))
    upper_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return statistics.fmean(values), estimates[lower_index], estimates[upper_index]


def paired_permutation_test(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_resamples: int = 100_000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired sign-flip test over claim-level differences."""

    if len(left) != len(right) or not left:
        raise ValueError("left and right must be non-empty and equal length")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    differences = [a - b for a, b in zip(left, right)]
    observed = statistics.fmean(differences)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        permuted = statistics.fmean(
            difference if rng.random() < 0.5 else -difference
            for difference in differences
        )
        if abs(permuted) >= abs(observed) - 1e-15:
            extreme += 1
    p_value = (extreme + 1) / (n_resamples + 1)
    if len(differences) > 1:
        sd = statistics.stdev(differences)
        effect = observed / sd if sd else math.copysign(math.inf, observed) if observed else 0.0
    else:
        effect = math.nan
    return {
        "mean_difference": observed,
        "paired_standardised_effect": effect,
        "p_value": p_value,
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Holm family-wise-error adjusted p-values in original order."""

    values = list(p_values)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(values)
    running_max = 0.0
    total = len(values)
    for position, (original_index, value) in enumerate(ordered):
        candidate = min(1.0, (total - position) * value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted
