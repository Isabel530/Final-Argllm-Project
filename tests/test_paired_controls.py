from __future__ import annotations

from argstrength.paired_controls import (
    build_claim_metrics,
    paired_prompt_contrasts,
    summarise_prompts,
)


def _rows(prompt_id, label, claim_id, margins):
    output = []
    for seed, margin in enumerate(margins, start=1):
        for rank, score in ((1, 50 + margin), (2, 50)):
            output.append(
                {
                    "study": "test",
                    "model_key": "model",
                    "prompt_condition": prompt_id,
                    "decoding": "hard",
                    "thinking": False,
                    "seed": seed,
                    "claim_id": claim_id,
                    "claim": f"claim {claim_id}",
                    "stance": "support" if claim_id % 2 else "attack",
                    "argument_id": f"rank_{rank}",
                    "argument": f"argument {rank}",
                    "reference_rank": rank,
                    "score": score,
                    "score_parse_ok": True,
                    "generated_tokens": 5,
                    "generation_seconds": 0.2,
                    "metadata": {"prompt_id": prompt_id, "prompt_label": label},
                }
            )
    return output


def test_claim_metrics_and_paired_holm_are_claim_level():
    raw = []
    for claim_id in (1, 2, 3, 4):
        raw += _rows("winner", None, claim_id, [10, 10, 10])
        raw += _rows("direct", "direct", claim_id, [-5, -5, -5])
        raw += _rows("rubric", "rubric", claim_id, [5, -5, 5])
    claims = build_claim_metrics(raw, winner_id="winner")
    assert len(claims) == 12
    summary = summarise_prompts(claims, bootstrap_resamples=100, seed=7)
    assert {row["prompt_label"] for row in summary} == {"winner", "direct", "rubric"}
    contrasts = paired_prompt_contrasts(
        claims,
        winner_id="winner",
        bootstrap_resamples=100,
        permutation_resamples=100,
        seed=7,
    )
    overall_accuracy = [
        row for row in contrasts if row["scope"] == "all" and row["metric"] == "pairwise_accuracy"
    ]
    assert len(overall_accuracy) == 2
    assert all(row["n_claims"] == 4 for row in overall_accuracy)
    assert all("p_value_holm" in row for row in overall_accuracy)
