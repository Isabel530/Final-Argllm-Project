from __future__ import annotations

import pytest

from argstrength.extended_analysis import (
    complete_score_cells,
    paired_tost,
    pair_observations,
    qwen_tie_transition_analysis,
    rank_distance_summary,
)


def _rows(
    *,
    study="main_study",
    model="qwen3_8b",
    prompt="direct",
    thinking=False,
    claim_id=1,
    scores_by_seed=((1, (90, 70, 50, 30)), (2, (80, 60, 40, 20))),
):
    output = []
    for seed, scores in scores_by_seed:
        for rank, score in enumerate(scores, start=1):
            output.append(
                {
                    "study": study,
                    "model_key": model,
                    "prompt_condition": prompt,
                    "decoding": "soft",
                    "thinking": thinking,
                    "seed": seed,
                    "claim_id": claim_id,
                    "claim": f"claim {claim_id}",
                    "stance": "support",
                    "argument_id": f"rank_{rank}",
                    "argument": f"argument {rank}",
                    "reference_rank": rank,
                    "score": score,
                    "score_parse_ok": True,
                    "format_ok": True,
                    "generated_tokens": 5,
                    "generation_seconds": 0.1,
                }
            )
    return output


def test_pair_observations_assign_rank_distance_and_reference_direction():
    cells = complete_score_cells(_rows())
    pairs = pair_observations(cells)
    assert len(cells) == 1
    assert len(pairs) == 6
    assert [row["rank_distance"] for row in pairs].count(1) == 3
    assert [row["rank_distance"] for row in pairs].count(2) == 2
    assert [row["rank_distance"] for row in pairs].count(3) == 1
    assert all(row["relation"] == 1 for row in pairs)
    assert all(row["pairwise_accuracy"] == 1 for row in pairs)
    assert all(row["seed_flip_rate"] == 0 for row in pairs)


def test_rank_distance_summary_keeps_claim_as_the_row_unit():
    rows = _rows(claim_id=1) + _rows(claim_id=2)
    pairs = pair_observations(complete_score_cells(rows))
    claim_rows, aggregate, _ = rank_distance_summary(
        pairs, bootstrap_resamples=100
    )
    assert len(claim_rows) == 6  # two claims × three distances
    distance_one = [
        row for row in aggregate if row["rank_distance"] == 1
    ][0]
    assert distance_one["n_claims"] == 2
    assert distance_one["n_pairs"] == 6
    assert distance_one["mean_pairwise_accuracy"] == 1


def test_qwen_tie_transition_tracks_correct_wrong_and_seed_disagreement():
    no_think = _rows(
        study="reasoning_study",
        thinking=False,
        scores_by_seed=((1, (50, 50, 30, 20)), (2, (50, 50, 30, 20))),
    )
    think = _rows(
        study="reasoning_study",
        thinking=True,
        scores_by_seed=((1, (60, 40, 30, 20)), (2, (40, 60, 30, 20))),
    )
    pairs = pair_observations(complete_score_cells(no_think + think))
    detail, summary = qwen_tie_transition_analysis(
        pairs, bootstrap_resamples=100
    )
    target = [
        row
        for row in detail
        if row["better_rank"] == 1 and row["worse_rank"] == 2
    ][0]
    # Think seed margins cancel in the seed average, so the pair stays tied,
    # while the individual seed directions disagree.
    assert target["transition"] == "Tie_to_Tie"
    assert target["think_seed_disagreement"] is True
    assert target["think_contains_correct_and_wrong_seeds"] is True
    overall = [row for row in summary if row["rank_distance"] == "all"][0]
    assert overall["n_no_think_tied_pairs"] >= 1


def test_normalized_margin_is_undefined_for_all_tied_claim():
    rows = _rows(
        scores_by_seed=((1, (50, 50, 50, 50)), (2, (50, 50, 50, 50)))
    )
    pairs = pair_observations(complete_score_cells(rows))
    assert all(row["normalized_margin"] is None for row in pairs)


def test_paired_tost_distinguishes_equivalence_from_imprecision():
    equivalent = paired_tost(
        [0.005, 0.006, 0.004, 0.005, 0.006] * 20,
        lower_bound=-0.02,
        upper_bound=0.02,
    )
    assert equivalent["equivalent_0_05"] is True
    assert equivalent["ci_low"] > -0.02
    assert equivalent["ci_high"] < 0.02

    imprecise = paired_tost(
        [-0.20, 0.20, -0.18, 0.18, 0.0],
        lower_bound=-0.02,
        upper_bound=0.02,
    )
    assert imprecise["equivalent_0_05"] is False


def test_incomplete_seed_cell_is_excluded():
    rows = _rows()
    rows.pop()
    assert complete_score_cells(rows) == []
