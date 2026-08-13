"""Bayesian-conformal adaptive sampling for repeated LLM judge scores.

The policy treats the repeated score difference for an argument pair as one
of three preference outcomes: the reference-better argument wins, the scores
tie, or the reference-worse argument wins.  A symmetric Dirichlet posterior
smooths the empirical preference probabilities at every sampling checkpoint.

Split-conformal calibration is performed at the *claim* level.  Each
calibration claim contributes its worst true-label nonconformity across every
pair and every checkpoint.  The resulting quantile therefore protects the
simultaneous sequence of prediction sets under the usual exchangeability
assumption.  It does not guarantee conditional error among only the accepted
predictions, so selective error and coverage are always reported separately.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
import statistics
from typing import Any, Iterable, Sequence

from .uq import _complete_score_cube, _relation


PREFERENCE_LABELS = (-1, 0, 1)
PREFERENCE_NAMES = {-1: "worse_wins", 0: "tie", 1: "better_wins"}


def dirichlet_preference_probabilities(
    margins: Sequence[float],
    *,
    prior_concentration: float = 0.5,
    tie_tolerance: float = 0.0,
) -> dict[int, float]:
    """Return posterior-mean probabilities for worse/tie/better outcomes."""

    if not margins:
        raise ValueError("margins cannot be empty")
    if prior_concentration <= 0:
        raise ValueError("prior_concentration must be positive")
    counts = {label: prior_concentration for label in PREFERENCE_LABELS}
    for margin in margins:
        counts[_relation(float(margin), tie_tolerance)] += 1.0
    total = sum(counts.values())
    return {label: counts[label] / total for label in PREFERENCE_LABELS}


def conformal_prediction_set(
    probabilities: dict[int, float], threshold: float
) -> tuple[int, ...]:
    """Construct the label set ``{y: 1 - p(y) <= threshold}``."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if set(probabilities) != set(PREFERENCE_LABELS):
        raise ValueError("probabilities must contain worse, tie, and better labels")
    if any(not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("probabilities must be in [0, 1]")
    return tuple(
        label
        for label in PREFERENCE_LABELS
        if 1.0 - float(probabilities[label]) <= threshold
    )


def split_conformal_quantile(
    nonconformity_scores: Sequence[float], miscoverage: float
) -> float:
    """Finite-sample split-conformal quantile with the standard +1 correction."""

    if not nonconformity_scores:
        raise ValueError("nonconformity_scores cannot be empty")
    if not 0 < miscoverage < 1:
        raise ValueError("miscoverage must be in (0, 1)")
    ordered = sorted(float(value) for value in nonconformity_scores)
    if any(not 0 <= value <= 1 for value in ordered):
        raise ValueError("nonconformity scores must be in [0, 1]")
    rank = math.ceil((len(ordered) + 1) * (1.0 - miscoverage))
    # When the requested finite-sample coverage is unattainable, the conformal
    # quantile is +infinity.  Since these scores live in [0, 1], 1.0 is the
    # equivalent conservative threshold and returns the full label set.
    return 1.0 if rank > len(ordered) else ordered[rank - 1]


def complete_adaptive_cells(
    rows: Sequence[dict[str, Any]],
    *,
    prompt_id: str,
    seed_order: Sequence[int],
) -> list[dict[str, Any]]:
    """Return complete score cells for one prompt in a fixed sampling order."""

    requested = tuple(int(seed) for seed in seed_order)
    if len(requested) != len(set(requested)):
        raise ValueError("seed_order must contain unique seeds")
    cells = [
        cell
        for cell in _complete_score_cube(rows)
        if cell["prompt_id"] == prompt_id and set(requested) <= set(cell["seeds"])
    ]
    for cell in cells:
        cell["adaptive_seed_order"] = requested
    return cells


def _pair_state(
    cell: dict[str, Any],
    better_rank: int,
    worse_rank: int,
    seeds: Sequence[int],
    *,
    prior_concentration: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    margins = [
        float(cell["scores_by_rank"][better_rank][int(seed)])
        - float(cell["scores_by_rank"][worse_rank][int(seed)])
        for seed in seeds
    ]
    probabilities = dirichlet_preference_probabilities(
        margins,
        prior_concentration=prior_concentration,
        tie_tolerance=tie_tolerance,
    )
    return {
        "margins": margins,
        "probabilities": probabilities,
        "forced_relation": _relation(statistics.fmean(margins), tie_tolerance),
        "true_label_nonconformity": 1.0 - probabilities[1],
    }


def claim_nonconformity_scores(
    cells: Sequence[dict[str, Any]],
    *,
    checkpoints: Sequence[int],
    prior_concentration: float = 0.5,
    tie_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Build one worst-case simultaneous calibration score per claim."""

    ordered_checkpoints = tuple(sorted(set(int(value) for value in checkpoints)))
    if not ordered_checkpoints or ordered_checkpoints[0] < 2:
        raise ValueError("checkpoints must contain sample sizes of at least two")
    output: list[dict[str, Any]] = []
    for cell in cells:
        seeds = cell["adaptive_seed_order"]
        if ordered_checkpoints[-1] > len(seeds):
            raise ValueError("checkpoint exceeds available seeds")
        ranks = sorted(cell["scores_by_rank"])
        scores = []
        for checkpoint in ordered_checkpoints:
            prefix = seeds[:checkpoint]
            for better_rank, worse_rank in combinations(ranks, 2):
                state = _pair_state(
                    cell,
                    better_rank,
                    worse_rank,
                    prefix,
                    prior_concentration=prior_concentration,
                    tie_tolerance=tie_tolerance,
                )
                scores.append(state["true_label_nonconformity"])
        output.append(
            {
                "claim_id": cell["claim_id"],
                "claim": cell["claim"],
                "max_true_label_nonconformity": max(scores),
                "n_pairs_times_checkpoints": len(scores),
            }
        )
    return output


def calibrate_claim_grouped_thresholds(
    cells: Sequence[dict[str, Any]],
    *,
    checkpoints: Sequence[int],
    miscoverage_levels: Sequence[float],
    prior_concentration: float = 0.5,
    tie_tolerance: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit simultaneous claim-grouped split-conformal thresholds."""

    scores = claim_nonconformity_scores(
        cells,
        checkpoints=checkpoints,
        prior_concentration=prior_concentration,
        tie_tolerance=tie_tolerance,
    )
    values = [row["max_true_label_nonconformity"] for row in scores]
    thresholds = [
        {
            "miscoverage": float(level),
            "nominal_coverage": 1.0 - float(level),
            "threshold": split_conformal_quantile(values, float(level)),
            "n_calibration_claims": len(values),
            "calibration_unit": "claim_max_over_pairs_and_checkpoints",
        }
        for level in miscoverage_levels
    ]
    return thresholds, scores


def evaluate_adaptive_policy(
    cells: Sequence[dict[str, Any]],
    *,
    dataset: str,
    threshold: float,
    miscoverage: float,
    checkpoints: Sequence[int],
    prior_concentration: float = 0.5,
    tie_tolerance: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a claim-level adaptive policy and retain pair-level abstentions."""

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
            all_singleton = True
            for better_rank, worse_rank in combinations(ranks, 2):
                state = _pair_state(
                    cell,
                    better_rank,
                    worse_rank,
                    prefix,
                    prior_concentration=prior_concentration,
                    tie_tolerance=tie_tolerance,
                )
                prediction_set = conformal_prediction_set(
                    state["probabilities"], threshold
                )
                state["prediction_set"] = prediction_set
                states[(better_rank, worse_rank)] = state
                all_singleton &= len(prediction_set) == 1
            selected_checkpoint = checkpoint
            selected_states = states
            if all_singleton:
                break

        claim_pairs = []
        for (better_rank, worse_rank), state in selected_states.items():
            prediction_set = state["prediction_set"]
            decision = prediction_set[0] if len(prediction_set) == 1 else None
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
                "miscoverage": miscoverage,
                "conformal_threshold": threshold,
                "stopping_checkpoint": selected_checkpoint,
                "prediction_set": "|".join(
                    PREFERENCE_NAMES[label] for label in prediction_set
                ),
                "prediction_set_size": len(prediction_set),
                "decision": PREFERENCE_NAMES[decision] if decision is not None else None,
                "abstained": decision is None,
                "reference_correct": decision == 1 if decision is not None else None,
                "pairwise_credit": (
                    1.0 if decision == 1 else 0.5 if decision == 0 else 0.0
                    if decision is not None
                    else None
                ),
                "true_label_in_set": 1 in prediction_set,
                "posterior_worse": state["probabilities"][-1],
                "posterior_tie": state["probabilities"][0],
                "posterior_better": state["probabilities"][1],
                "forced_relation": PREFERENCE_NAMES[state["forced_relation"]],
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
                "miscoverage": miscoverage,
                "conformal_threshold": threshold,
                "stopping_checkpoint": selected_checkpoint,
                "score_generations": selected_checkpoint * len(ranks),
                "n_pairs": len(claim_pairs),
                "n_answered_pairs": len(answered),
                "all_pairs_answered": len(answered) == len(claim_pairs),
                "complete_reference_order": (
                    len(answered) == len(claim_pairs)
                    and all(row["reference_correct"] for row in answered)
                ),
                "simultaneous_true_label_coverage": all(
                    row["true_label_in_set"] for row in claim_pairs
                ),
            }
        )
    return pair_rows, claim_rows


def _scopes(cells: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    scopes = [("all", list(cells))]
    panels = sorted(
        {
            cell["dataset_item_metadata"].get("panel")
            for cell in cells
            if cell["dataset_item_metadata"].get("panel")
        }
    )
    scopes.extend(
        (
            str(panel),
            [
                cell
                for cell in cells
                if cell["dataset_item_metadata"].get("panel") == panel
            ],
        )
        for panel in panels
    )
    return scopes


def summarise_adaptive_policy(
    pair_rows: Sequence[dict[str, Any]],
    claim_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarise coverage, selective risk, and generation cost by panel."""

    grouped_pairs: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_claims: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    datasets = sorted({str(row["dataset"]) for row in claim_rows})
    for dataset in datasets:
        dataset_claims = [row for row in claim_rows if row["dataset"] == dataset]
        panels = sorted({row.get("panel") for row in dataset_claims if row.get("panel")})
        levels = sorted({float(row["miscoverage"]) for row in dataset_claims})
        for level in levels:
            for scope in ["all", *panels]:
                selected_claims = [
                    row
                    for row in dataset_claims
                    if float(row["miscoverage"]) == level
                    and (scope == "all" or row.get("panel") == scope)
                ]
                claim_ids = {row["claim_id"] for row in selected_claims}
                selected_pairs = [
                    row
                    for row in pair_rows
                    if row["dataset"] == dataset
                    and float(row["miscoverage"]) == level
                    and row["claim_id"] in claim_ids
                    and (scope == "all" or row.get("panel") == scope)
                ]
                grouped_claims[(dataset, level, scope)] = selected_claims
                grouped_pairs[(dataset, level, scope)] = selected_pairs

    output = []
    for key, claims in sorted(grouped_claims.items()):
        pairs = grouped_pairs[key]
        answered = [row for row in pairs if not row["abstained"]]
        output.append(
            {
                "dataset": key[0],
                "miscoverage": key[1],
                "scope": key[2],
                "n_claims": len(claims),
                "n_pairs": len(pairs),
                "n_answered_pairs": len(answered),
                "pair_coverage": len(answered) / len(pairs) if pairs else None,
                "selective_reference_accuracy": (
                    statistics.fmean(row["reference_correct"] for row in answered)
                    if answered
                    else None
                ),
                "selective_pairwise_accuracy": (
                    statistics.fmean(row["pairwise_credit"] for row in answered)
                    if answered
                    else None
                ),
                "marginal_pair_set_coverage": statistics.fmean(
                    row["true_label_in_set"] for row in pairs
                ),
                "simultaneous_claim_coverage": statistics.fmean(
                    row["simultaneous_true_label_coverage"] for row in claims
                ),
                "all_pairs_answered_rate": statistics.fmean(
                    row["all_pairs_answered"] for row in claims
                ),
                "complete_reference_order_rate": statistics.fmean(
                    row["complete_reference_order"] for row in claims
                ),
                "mean_runs_per_claim": statistics.fmean(
                    row["stopping_checkpoint"] for row in claims
                ),
                "mean_score_generations_per_claim": statistics.fmean(
                    row["score_generations"] for row in claims
                ),
                "generation_saving_vs_10_runs": 1.0
                - statistics.fmean(row["stopping_checkpoint"] for row in claims) / 10.0,
            }
        )
    return output


def fixed_budget_summaries(
    cells: Sequence[dict[str, Any]],
    *,
    dataset: str,
    budgets: Iterable[int],
    tie_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Fully answering baselines for fixed numbers of stochastic runs."""

    output = []
    for scope, scoped_cells in _scopes(cells):
        for budget in sorted(set(int(value) for value in budgets)):
            credits, strict = [], []
            for cell in scoped_cells:
                seeds = cell["adaptive_seed_order"][:budget]
                ranks = sorted(cell["scores_by_rank"])
                for better_rank, worse_rank in combinations(ranks, 2):
                    margins = [
                        cell["scores_by_rank"][better_rank][seed]
                        - cell["scores_by_rank"][worse_rank][seed]
                        for seed in seeds
                    ]
                    relation = _relation(statistics.fmean(margins), tie_tolerance)
                    credits.append(1.0 if relation == 1 else 0.5 if relation == 0 else 0.0)
                    strict.append(relation == 1)
            output.append(
                {
                    "dataset": dataset,
                    "scope": scope,
                    "policy": f"fixed_{budget}",
                    "runs_per_claim": budget,
                    "score_generations_per_claim": budget * 4,
                    "n_claims": len(scoped_cells),
                    "n_pairs": len(credits),
                    "pair_coverage": 1.0,
                    "reference_accuracy": statistics.fmean(strict),
                    "pairwise_accuracy": statistics.fmean(credits),
                    "generation_saving_vs_10_runs": 1.0 - budget / 10.0,
                }
            )
    return output
