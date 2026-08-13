from __future__ import annotations

import math

from argstrength.adaptive_uq import (
    calibrate_claim_grouped_thresholds,
    complete_adaptive_cells,
    conformal_prediction_set,
    dirichlet_preference_probabilities,
    evaluate_adaptive_policy,
    split_conformal_quantile,
)


def _raw_rows(*, easy: bool = True):
    rows = []
    for seed in range(10):
        scores = (90, 70, 40, 10) if easy else (50 + seed % 2, 50, 50, 50)
        for rank, score in enumerate(scores, start=1):
            rows.append(
                {
                    "study": "uq_apo",
                    "model_key": "qwen3_8b",
                    "prompt_condition": "p_test",
                    "decoding": "hard",
                    "thinking": False,
                    "seed": seed,
                    "claim_id": 1 if easy else 2,
                    "claim": "easy" if easy else "uncertain",
                    "stance": "support",
                    "argument_id": f"rank_{rank}",
                    "argument": f"argument {rank}",
                    "reference_rank": rank,
                    "score": score,
                    "score_parse_ok": True,
                    "metadata": {"prompt_id": "p_test"},
                }
            )
    return rows


def test_dirichlet_posterior_is_smoothed_and_normalised():
    probabilities = dirichlet_preference_probabilities([5, 3, 1])
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert probabilities[1] > probabilities[0]
    assert probabilities[-1] > 0


def test_split_conformal_quantile_uses_finite_sample_correction():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert split_conformal_quantile(scores, 0.25) == 0.4
    assert split_conformal_quantile(scores, 0.10) == 1.0


def test_prediction_set_can_be_singleton_or_abstaining():
    probabilities = {-1: 0.05, 0: 0.05, 1: 0.90}
    assert conformal_prediction_set(probabilities, 0.15) == (1,)
    assert conformal_prediction_set(probabilities, 0.95) == (-1, 0, 1)


def test_claim_grouped_calibration_and_easy_early_stopping():
    rows = _raw_rows(easy=True)
    cells = complete_adaptive_cells(rows, prompt_id="p_test", seed_order=range(10))
    thresholds, scores = calibrate_claim_grouped_thresholds(
        cells,
        checkpoints=[3, 5, 10],
        miscoverage_levels=[0.5],
    )
    assert len(scores) == 1
    assert scores[0]["n_pairs_times_checkpoints"] == 18
    pair_rows, claim_rows = evaluate_adaptive_policy(
        cells,
        dataset="test",
        threshold=0.40,
        miscoverage=0.10,
        checkpoints=[3, 5, 10],
    )
    assert len(pair_rows) == 6
    assert claim_rows[0]["stopping_checkpoint"] == 3
    assert claim_rows[0]["all_pairs_answered"] is True
    assert thresholds[0]["n_calibration_claims"] == 1

