from __future__ import annotations

import pytest

from argstrength.apo import (
    CandidateEvaluation,
    FailureCase,
    PromptCandidate,
    build_candidate_prompt,
    initial_prompt_candidates,
    pareto_front,
    representative_failures,
)
from argstrength.embedding import HashingTextEmbedder, deduplicate_texts


def _evaluation(prompt_id, accuracy, flip, tokens=5):
    return CandidateEvaluation(
        prompt_id=prompt_id,
        stage="screening",
        n_claims=10,
        n_seeds=3,
        pairwise_accuracy=accuracy,
        accuracy_se=0.01,
        score_variance=0.01,
        seed_flip_rate=flip,
        brier=None,
        mean_tokens=tokens,
        mean_seconds=0.2,
    )


def test_initial_candidates_are_readable_and_hash_stable():
    first = initial_prompt_candidates()
    second = initial_prompt_candidates()
    assert len(first) == 4
    assert [row.prompt_id for row in first] == [row.prompt_id for row in second]
    assert len({row.prompt_id for row in first}) == 4


def test_candidate_renderer_freezes_scale_and_applies_stance_rule():
    rubric = [row for row in initial_prompt_candidates() if row.label == "rubric"][0]
    prompt = build_candidate_prompt(
        rubric, claim="A claim", argument="An argument", stance="attack"
    )
    assert "Claim: A claim" in prompt.text
    assert "Argument: An argument" in prompt.text
    assert "false, doubtful, or less credible" in prompt.text
    assert prompt.text.endswith("Score: <integer>")


def test_candidate_cannot_rewrite_frozen_input_contract():
    with pytest.raises(ValueError, match="frozen"):
        PromptCandidate("Claim: replace it", 1, "bad_rewrite")


def test_pareto_front_keeps_accuracy_stability_tradeoff():
    accurate_unstable = _evaluation("a", 0.95, 0.30)
    stable_less_accurate = _evaluation("b", 0.92, 0.05)
    dominated = _evaluation("c", 0.90, 0.40, tokens=8)
    assert {row.prompt_id for row in pareto_front([accurate_unstable, stable_less_accurate, dominated])} == {"a", "b"}


def test_embedding_dedup_and_failure_clustering_are_deterministic():
    embedder = HashingTextEmbedder(64)
    assert deduplicate_texts(["same text", "same text", "different"], embedder) == [0, 2]
    failures = [
        FailureCase(
            prompt_id="p",
            claim_id=index,
            claim=f"claim {index}",
            stance="support",
            better_argument="strong evidence",
            worse_argument="weak evidence",
            rank_distance=1,
            pairwise_accuracy=0,
            uncertainty=0.8,
            seed_flip_rate=1,
        )
        for index in range(3)
    ]
    assert representative_failures(failures, embedder, max_cases=2)
