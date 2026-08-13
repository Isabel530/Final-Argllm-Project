"""Practical Bayesian early stopping without a formal coverage guarantee.

This complements the deliberately conservative simultaneous conformal policy
in :mod:`argstrength.adaptive_uq`.  Fixed posterior-confidence thresholds show
the empirical accuracy--coverage--cost frontier and must not be described as
conformal risk guarantees.
"""

from __future__ import annotations

from itertools import combinations
import statistics
from typing import Any, Sequence

from .adaptive_uq import PREFERENCE_NAMES, _pair_state


def evaluate_bayesian_confidence_policy(
    cells: Sequence[dict[str, Any]],
    *,
    dataset: str,
    confidence_threshold: float,
    checkpoints: Sequence[int],
    prior_concentration: float = 0.5,
    tie_tolerance: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stop when all pairs exceed a fixed posterior-confidence threshold."""

    if not 1 / 3 < confidence_threshold < 1:
        raise ValueError("confidence_threshold must be between 1/3 and 1")
    ordered_checkpoints = tuple(sorted(set(int(value) for value in checkpoints)))
    pair_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for cell in cells:
        seed_order = cell["adaptive_seed_order"]
        ranks = sorted(cell["scores_by_rank"])
        selected_checkpoint = ordered_checkpoints[-1]
        selected_states: dict[tuple[int, int], dict[str, Any]] = {}
        for checkpoint in ordered_checkpoints:
            prefix = seed_order[:checkpoint]
            states = {}
            all_resolved = True
            for better_rank, worse_rank in combinations(ranks, 2):
                state = _pair_state(
                    cell,
                    better_rank,
                    worse_rank,
                    prefix,
                    prior_concentration=prior_concentration,
                    tie_tolerance=tie_tolerance,
                )
                maximum = max(state["probabilities"].values())
                winners = [
                    label
                    for label, probability in state["probabilities"].items()
                    if probability == maximum
                ]
                state["posterior_confidence"] = maximum
                state["decision_label"] = (
                    winners[0]
                    if len(winners) == 1 and maximum >= confidence_threshold
                    else None
                )
                states[(better_rank, worse_rank)] = state
                all_resolved &= state["decision_label"] is not None
            selected_checkpoint = checkpoint
            selected_states = states
            if all_resolved:
                break

        claim_pairs = []
        for (better_rank, worse_rank), state in selected_states.items():
            decision = state["decision_label"]
            row = {
                "dataset": dataset,
                "model": cell["model"],
                "prompt_id": cell["prompt_id"],
                "claim_id": cell["claim_id"],
                "claim": cell["claim"],
                "source_topic": cell["dataset_item_metadata"].get("source_topic"),
                "panel": cell["dataset_item_metadata"].get("panel"),
                "better_rank": better_rank,
                "worse_rank": worse_rank,
                "rank_distance": worse_rank - better_rank,
                "confidence_threshold": confidence_threshold,
                "stopping_checkpoint": selected_checkpoint,
                "posterior_confidence": state["posterior_confidence"],
                "decision": PREFERENCE_NAMES[decision] if decision is not None else None,
                "abstained": decision is None,
                "reference_correct": decision == 1 if decision is not None else None,
                "pairwise_credit": (
                    None
                    if decision is None
                    else 1.0
                    if decision == 1
                    else 0.5
                    if decision == 0
                    else 0.0
                ),
                "posterior_worse": state["probabilities"][-1],
                "posterior_tie": state["probabilities"][0],
                "posterior_better": state["probabilities"][1],
            }
            pair_rows.append(row)
            claim_pairs.append(row)

        answered = [row for row in claim_pairs if not row["abstained"]]
        claim_rows.append(
            {
                "dataset": dataset,
                "model": cell["model"],
                "prompt_id": cell["prompt_id"],
                "claim_id": cell["claim_id"],
                "claim": cell["claim"],
                "source_topic": cell["dataset_item_metadata"].get("source_topic"),
                "panel": cell["dataset_item_metadata"].get("panel"),
                "confidence_threshold": confidence_threshold,
                "stopping_checkpoint": selected_checkpoint,
                "score_generations": selected_checkpoint * len(ranks),
                "n_pairs": len(claim_pairs),
                "n_answered_pairs": len(answered),
                "all_pairs_answered": len(answered) == len(claim_pairs),
                "complete_reference_order": (
                    len(answered) == len(claim_pairs)
                    and all(row["reference_correct"] for row in answered)
                ),
            }
        )
    return pair_rows, claim_rows


def summarise_bayesian_confidence_policy(
    pair_rows: Sequence[dict[str, Any]],
    claim_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarise practical Bayesian early stopping by dataset and panel."""

    output = []
    datasets = sorted({str(row["dataset"]) for row in claim_rows})
    for dataset in datasets:
        dataset_claims = [row for row in claim_rows if row["dataset"] == dataset]
        panels = sorted({row.get("panel") for row in dataset_claims if row.get("panel")})
        thresholds = sorted(
            {float(row["confidence_threshold"]) for row in dataset_claims}
        )
        for threshold in thresholds:
            for scope in ["all", *panels]:
                claims = [
                    row
                    for row in dataset_claims
                    if float(row["confidence_threshold"]) == threshold
                    and (scope == "all" or row.get("panel") == scope)
                ]
                claim_ids = {row["claim_id"] for row in claims}
                pairs = [
                    row
                    for row in pair_rows
                    if row["dataset"] == dataset
                    and float(row["confidence_threshold"]) == threshold
                    and row["claim_id"] in claim_ids
                    and (scope == "all" or row.get("panel") == scope)
                ]
                answered = [row for row in pairs if not row["abstained"]]
                mean_runs = statistics.fmean(
                    row["stopping_checkpoint"] for row in claims
                )
                output.append(
                    {
                        "dataset": dataset,
                        "confidence_threshold": threshold,
                        "scope": scope,
                        "n_claims": len(claims),
                        "n_pairs": len(pairs),
                        "n_answered_pairs": len(answered),
                        "pair_coverage": len(answered) / len(pairs) if pairs else None,
                        "selective_reference_accuracy": (
                            statistics.fmean(
                                row["reference_correct"] for row in answered
                            )
                            if answered
                            else None
                        ),
                        "selective_pairwise_accuracy": (
                            statistics.fmean(row["pairwise_credit"] for row in answered)
                            if answered
                            else None
                        ),
                        "all_pairs_answered_rate": statistics.fmean(
                            row["all_pairs_answered"] for row in claims
                        ),
                        "complete_reference_order_rate": statistics.fmean(
                            row["complete_reference_order"] for row in claims
                        ),
                        "mean_runs_per_claim": mean_runs,
                        "mean_score_generations_per_claim": mean_runs * 4,
                        "generation_saving_vs_10_runs": 1.0 - mean_runs / 10.0,
                        "coverage_guarantee": "none_empirical_bayesian_policy",
                    }
                )
    return output

