from __future__ import annotations

import math

from argstrength.uq import (
    auroc,
    build_pair_uq_rows,
    pair_uncertainty_signals,
    seed_partitions,
)


def _raw_rows():
    rows = []
    for seed in range(10):
        scores = (90 - seed % 2, 70 + seed % 3, 40, 10)
        for rank, score in enumerate(scores, start=1):
            rows.append(
                {
                    "study": "uq_apo",
                    "model_key": "qwen3_8b",
                    "prompt_condition": "p_test",
                    "decoding": "hard",
                    "thinking": False,
                    "seed": seed,
                    "claim_id": 7,
                    "claim": "claim",
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


def test_three_seed_uncertainty_distinguishes_unanimous_and_split_preferences():
    unanimous = pair_uncertainty_signals([20, 10, 5])
    split = pair_uncertainty_signals([20, -10, 5])
    assert unanimous["preference_entropy"] == 0
    assert unanimous["pair_disagreement_probability"] == 0
    assert split["preference_entropy"] > 0
    assert math.isclose(split["pair_disagreement_probability"], 2 / 3)


def test_ten_runs_create_252_disjoint_five_plus_five_partitions():
    partitions = seed_partitions(range(10), estimate_size=5, mode="all")
    assert len(partitions) == 252
    for estimate, heldout in partitions:
        assert len(estimate) == len(heldout) == 5
        assert set(estimate).isdisjoint(heldout)
        assert set(estimate) | set(heldout) == set(range(10))


def test_pair_rows_keep_estimate_and_future_targets_separate():
    partition = [((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))]
    rows = build_pair_uq_rows(_raw_rows(), partition)
    assert len(rows) == 6
    assert all(row["future_reference_error"] == 0 for row in rows)
    assert all(row["estimate_seeds"] == "0|1|2|3|4" for row in rows)
    assert all(row["heldout_seeds"] == "5|6|7|8|9" for row in rows)


def test_auroc_is_one_for_perfect_risk_order_and_half_for_ties():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.5, 0.5], [0, 1]) == 0.5
