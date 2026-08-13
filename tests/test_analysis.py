import pytest

from scripts.analyse import (
    compliance_table,
    factorial_effects,
    model_prompt_interactions,
    seed_average_claim_table,
    seed_claim_table,
)


def _row(*, argument, rank, score, seed, thinking=False, raw_output=None):
    raw = raw_output if raw_output is not None else f"Score: {score}"
    return {
        "study": "main",
        "model_key": "qwen",
        "prompt_condition": "direct",
        "decoding": "soft",
        "thinking": thinking,
        "seed": seed,
        "claim_id": 1,
        "stance": "support",
        "argument_id": argument,
        "reference_rank": rank,
        "score": score,
        "score_parse_ok": True,
        "format_ok": raw.startswith("Score:"),
        "raw_output": raw,
        "generated_tokens": 8,
        "hit_max_new_tokens": False,
        "generation_seconds": 0.1,
    }


def test_format_compliance_and_score_extraction_are_separate():
    rows = [
        _row(argument="a", rank=1, score=90, seed=1),
        _row(
            argument="b",
            rank=2,
            score=70,
            seed=1,
            raw_output="Reasoning.\nFinal score: 70/100",
        ),
    ]
    table = compliance_table(rows)
    assert table[0]["format_compliance_rate"] == 0.5
    assert table[0]["score_extraction_rate"] == 1.0


def test_seed_average_is_primary_claim_unit_and_keeps_modes_separate():
    rows = []
    for thinking in (False, True):
        for seed, scores in ((1, [90, 50]), (2, [70, 30])):
            rows.extend([
                _row(
                    argument="a", rank=1, score=scores[0],
                    seed=seed, thinking=thinking,
                ),
                _row(
                    argument="b", rank=2, score=scores[1],
                    seed=seed, thinking=thinking,
                ),
            ])
    averaged = seed_average_claim_table(rows, strict_margin=10)
    assert len(averaged) == 2
    assert {row["thinking"] for row in averaged} == {False, True}
    assert all(row["n_seeds"] == 2 for row in averaged)
    assert all(row["pairwise_accuracy"] == 1.0 for row in averaged)


def test_incomplete_claim_is_not_analysed():
    rows = [
        _row(argument="a", rank=1, score=90, seed=1),
        _row(argument="b", rank=2, score=70, seed=1),
        _row(argument="a", rank=1, score=80, seed=2),
    ]
    assert seed_average_claim_table(rows, strict_margin=10) == []


def test_configured_seed_set_catches_a_whole_missing_seed():
    rows = [
        _row(argument="a", rank=1, score=90, seed=1),
        _row(argument="b", rank=2, score=70, seed=1),
    ]
    assert seed_average_claim_table(
        rows,
        strict_margin=10,
        expected_study_seeds={"main": {1, 2}},
    ) == []


def test_per_seed_metrics_keep_modes_separate():
    rows = []
    for thinking in (False, True):
        rows.extend([
            _row(argument="a", rank=1, score=90, seed=1, thinking=thinking),
            _row(argument="b", rank=2, score=70, seed=1, thinking=thinking),
        ])
    output = seed_claim_table(rows, strict_margin=10)
    assert len(output) == 2
    assert {row["thinking"] for row in output} == {False, True}


def _claim_metric_row(model, prompt, claim_id, value):
    return {
        "study": "main",
        "model": model,
        "prompt": prompt,
        "decoding": "hard",
        "thinking": False,
        "claim_id": claim_id,
        "stance": "support",
        "spearman": value,
        "pairwise_accuracy": value,
        "strict_pairwise_accuracy": value,
        "tie_rate": value,
    }


def test_factorial_and_prompt_x_model_effects_are_claim_level_differences():
    prompt_values = {
        "model_a": {"direct": 0.0, "role": 1.0, "rubric": 2.0, "role_rubric": 4.0},
        "model_b": {"direct": 0.0, "role": 1.0, "rubric": 2.0, "role_rubric": 3.0},
    }
    rows = [
        _claim_metric_row(model, prompt, claim_id, value)
        for model, values in prompt_values.items()
        for prompt, value in values.items()
        for claim_id in (1, 2)
    ]

    within = factorial_effects(
        rows, bootstrap_resamples=100, permutation_resamples=100
    )
    within_spearman = {
        (row["model"], row["effect"]): row["mean_effect"]
        for row in within
        if row["stance"] == "all" and row["metric"] == "spearman"
    }
    assert within_spearman[("model_a", "role_main_effect")] == pytest.approx(1.5)
    assert within_spearman[("model_a", "rubric_main_effect")] == pytest.approx(2.5)
    assert within_spearman[("model_a", "role_x_rubric_interaction")] == pytest.approx(1.0)
    assert within_spearman[("model_b", "role_main_effect")] == pytest.approx(1.0)
    assert within_spearman[("model_b", "rubric_main_effect")] == pytest.approx(2.0)
    assert within_spearman[("model_b", "role_x_rubric_interaction")] == pytest.approx(0.0)

    between = model_prompt_interactions(
        rows, bootstrap_resamples=100, permutation_resamples=100
    )
    between_spearman = {
        row["effect"]: row["difference_in_effects_left_minus_right"]
        for row in between
        if row["stance"] == "all" and row["metric"] == "spearman"
    }
    assert between_spearman["role_main_effect"] == pytest.approx(0.5)
    assert between_spearman["rubric_main_effect"] == pytest.approx(0.5)
    assert between_spearman["role_x_rubric_interaction"] == pytest.approx(1.0)
