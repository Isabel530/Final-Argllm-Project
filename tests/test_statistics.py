import pytest

from argstrength.statistics import (
    bootstrap_mean_ci,
    holm_adjust,
    paired_permutation_test,
)


def test_bootstrap_is_reproducible_and_claim_level():
    first = bootstrap_mean_ci([0.0, 0.5, 1.0], n_resamples=200, seed=7)
    second = bootstrap_mean_ci([0.0, 0.5, 1.0], n_resamples=200, seed=7)
    assert first == second
    assert first[0] == pytest.approx(0.5)


def test_paired_permutation_uses_within_claim_differences():
    result = paired_permutation_test(
        [1.0, 0.8, 0.7], [0.0, 0.2, 0.1], n_resamples=500, seed=7
    )
    assert result["mean_difference"] == pytest.approx(0.7333333333)
    assert 0 < result["p_value"] <= 1


def test_holm_adjustment_preserves_order_and_monotonicity():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
