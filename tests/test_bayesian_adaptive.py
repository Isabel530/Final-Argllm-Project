from __future__ import annotations

from argstrength.adaptive_uq import complete_adaptive_cells
from argstrength.bayesian_adaptive import evaluate_bayesian_confidence_policy


def test_easy_claim_stops_after_three_runs():
    rows = []
    for seed in range(10):
        for rank, score in enumerate((90, 70, 40, 10), start=1):
            rows.append(
                {
                    "study": "uq_apo",
                    "model_key": "qwen3_8b",
                    "prompt_condition": "p_test",
                    "decoding": "hard",
                    "thinking": False,
                    "seed": seed,
                    "claim_id": 1,
                    "claim": "easy",
                    "stance": "support",
                    "argument_id": f"rank_{rank}",
                    "argument": f"argument {rank}",
                    "reference_rank": rank,
                    "score": score,
                    "score_parse_ok": True,
                    "metadata": {"prompt_id": "p_test"},
                }
            )
    cells = complete_adaptive_cells(rows, prompt_id="p_test", seed_order=range(10))
    pairs, claims = evaluate_bayesian_confidence_policy(
        cells,
        dataset="test",
        confidence_threshold=0.70,
        checkpoints=[3, 5, 10],
    )
    assert len(pairs) == 6
    assert claims[0]["stopping_checkpoint"] == 3
    assert claims[0]["all_pairs_answered"] is True

