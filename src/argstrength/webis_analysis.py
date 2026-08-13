"""Direct-human pair, UQ, and adaptive analyses for frozen Webis E11."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics
from typing import Any, Sequence

from .adaptive_uq import dirichlet_preference_probabilities
from .statistics import bootstrap_mean_ci, holm_adjust, paired_permutation_test
from .uq import (
    _fit_isotonic,
    _relation,
    auroc,
    average_precision,
    expected_calibration_error,
    pair_uncertainty_signals,
    spearman,
)


SCOPES = (
    "majority_decisive",
    "unanimous_decisive",
    "majority_all_tie_aware",
    "logical_decisive",
    "rhetorical_decisive",
    "dialectical_decisive",
    "combined_quality_decisive",
)
METRICS = (
    "pairwise_accuracy",
    "strict_accuracy",
    "seed_flip_rate",
    "normalised_margin_variance",
    "mean_tokens",
    "mean_seconds",
    "invalid_rate",
)
LABEL_TO_RELATION = {"A": 1, "Tie": 0, "B": -1}


def prompt_id(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("prompt_id", row["prompt_condition"]))


def prompt_label(row: dict[str, Any], winner_id: str) -> str:
    identifier = prompt_id(row)
    label = (row.get("metadata") or {}).get("prompt_label")
    return "winner" if identifier == winner_id else str(label or identifier)


def gold_label(pair: dict[str, Any], scope: str) -> str | None:
    """Return the prespecified E11 gold label for one sensitivity scope."""

    if scope == "majority_decisive":
        value = pair.get("majority_label")
        return value if value in {"A", "B"} else None
    if scope == "unanimous_decisive":
        value = pair.get("majority_label")
        return value if pair.get("unanimous") and value in {"A", "B"} else None
    if scope == "majority_all_tie_aware":
        return pair.get("majority_label")
    if scope.endswith("_decisive"):
        key = scope.removesuffix("_decisive")
        value = pair.get(key if key != "combined_quality" else "combined_label")
        return value if value in {"A", "B"} else None
    raise ValueError(f"unknown E11 scope {scope!r}")


def build_score_index(
    rows: Sequence[dict[str, Any]],
) -> tuple[
    dict[tuple[str, int, str, int], float],
    dict[tuple[str, int], list[dict[str, Any]]],
    dict[str, str],
]:
    """Index parsed scores and retain raw cost rows by prompt/group."""

    scores: dict[tuple[str, int, str, int], float] = {}
    raw_by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        identifier = prompt_id(row)
        claim_id = int(row["claim_id"])
        seed = int(row["seed"])
        argument_id = str(row["argument_id"])
        key = (identifier, claim_id, argument_id, seed)
        if key in scores:
            raise ValueError(f"duplicate E11 generation key {key}")
        if not row.get("score_parse_ok") or row.get("score") is None:
            scores[key] = math.nan
        else:
            scores[key] = float(row["score"])
        raw_by_group[(identifier, claim_id)].append(row)
        labels[identifier] = str((row.get("metadata") or {}).get("prompt_label") or identifier)
    return scores, raw_by_group, labels


def _direction_credit(direction: int, gold: str) -> tuple[float, float]:
    truth = LABEL_TO_RELATION[gold]
    strict = float(direction == truth)
    if truth == 0:
        return strict, strict
    credit = 1.0 if direction == truth else 0.5 if direction == 0 else 0.0
    return credit, strict


def build_pair_metrics(
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    *,
    prompt_ids: Sequence[str],
    seeds: Sequence[int],
    winner_id: str | None = None,
) -> list[dict[str, Any]]:
    """Score every directly annotated human pair under every frozen prompt."""

    scores, _, labels = build_score_index(rows)
    requested_seeds = tuple(int(seed) for seed in seeds)
    output = []
    for identifier in prompt_ids:
        if identifier not in labels:
            raise ValueError(f"prompt {identifier!r} absent from E11 raw rows")
        for pair in pairs:
            margins = []
            for seed in requested_seeds:
                left = scores.get(
                    (identifier, int(pair["external_claim_id"]), pair["argument_a_id"], seed)
                )
                right = scores.get(
                    (identifier, int(pair["external_claim_id"]), pair["argument_b_id"], seed)
                )
                if left is None or right is None or math.isnan(left) or math.isnan(right):
                    raise ValueError(
                        f"incomplete E11 pair scores for {identifier}/{pair['pair_id']}/{seed}"
                    )
                margins.append(left - right)
            mean_margin = statistics.fmean(margins)
            aggregate_direction = _relation(mean_margin)
            directions = [_relation(margin) for margin in margins]
            base = {
                "prompt_id": identifier,
                "prompt_label": "winner" if identifier == winner_id else labels[identifier],
                "pair_id": pair["pair_id"],
                "external_claim_id": int(pair["external_claim_id"]),
                "topic_id": int(pair["topic_id"]),
                "source_topic": pair["source_topic"],
                "stance": pair["stance"],
                "argument_a_id": pair["argument_a_id"],
                "argument_b_id": pair["argument_b_id"],
                "n_seeds": len(requested_seeds),
                "mean_margin_a_minus_b": mean_margin,
                "aggregate_direction": aggregate_direction,
                "seed_flip": float(len(set(directions)) > 1),
                "normalised_margin_variance": (
                    statistics.variance(margins) / 200.0**2 if len(margins) > 1 else 0.0
                ),
                "majority_label": pair.get("majority_label"),
                "unanimous": bool(pair.get("unanimous")),
                "agreement_count": int(pair["agreement_count"]),
            }
            for scope in SCOPES:
                gold = gold_label(pair, scope)
                if gold is None:
                    continue
                credit, strict = _direction_credit(aggregate_direction, gold)
                output.append(
                    {
                        **base,
                        "scope": scope,
                        "gold_label": gold,
                        "pairwise_accuracy": credit,
                        "strict_accuracy": strict,
                    }
                )
    return output


def build_topic_metrics(
    pair_rows: Sequence[dict[str, Any]],
    raw_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate pairs to the independent Webis topic unit."""

    costs: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    topic_by_claim: dict[int, int] = {}
    for row in raw_rows:
        identifier = prompt_id(row)
        claim_id = int(row["claim_id"])
        costs[(identifier, claim_id)].append(row)
        metadata = (row.get("metadata") or {}).get("dataset_item_metadata") or {}
        topic_by_claim[claim_id] = int(metadata["topic_id"])

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["scope"], row["prompt_id"], int(row["topic_id"]))].append(row)
    output = []
    for (scope, identifier, topic_id), pairs in sorted(grouped.items()):
        topic_cost_rows = [
            row
            for (candidate, claim_id), values in costs.items()
            if candidate == identifier and topic_by_claim[claim_id] == topic_id
            for row in values
        ]
        valid = [
            row
            for row in topic_cost_rows
            if row.get("score_parse_ok") and row.get("score") is not None
        ]
        output.append(
            {
                "scope": scope,
                "prompt_id": identifier,
                "prompt_label": pairs[0]["prompt_label"],
                "topic_id": topic_id,
                "source_topic": pairs[0]["source_topic"],
                "n_pairs": len(pairs),
                "n_topic_stance_groups": len(
                    {int(row["external_claim_id"]) for row in pairs}
                ),
                "pairwise_accuracy": statistics.fmean(
                    float(row["pairwise_accuracy"]) for row in pairs
                ),
                "strict_accuracy": statistics.fmean(
                    float(row["strict_accuracy"]) for row in pairs
                ),
                "seed_flip_rate": statistics.fmean(float(row["seed_flip"]) for row in pairs),
                "normalised_margin_variance": statistics.fmean(
                    float(row["normalised_margin_variance"]) for row in pairs
                ),
                "mean_tokens": statistics.fmean(
                    float(row.get("generated_tokens", 0)) for row in topic_cost_rows
                ),
                "mean_seconds": statistics.fmean(
                    float(row.get("generation_seconds", 0)) for row in topic_cost_rows
                ),
                "invalid_rate": 1.0 - len(valid) / len(topic_cost_rows),
            }
        )
    return output


def summarise_prompts_by_topic(
    topic_rows: Sequence[dict[str, Any]], *, bootstrap_resamples: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in topic_rows:
        grouped[(row["scope"], row["prompt_id"])].append(row)
    output = []
    for (scope, identifier), rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "scope": scope,
            "prompt_id": identifier,
            "prompt_label": rows[0]["prompt_label"],
            "n_topics": len(rows),
            "n_pairs": sum(int(row["n_pairs"]) for row in rows),
        }
        for metric in METRICS:
            mean, low, high = bootstrap_mean_ci(
                [float(row[metric]) for row in rows],
                n_resamples=bootstrap_resamples,
                seed=seed,
            )
            result[metric] = mean
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        output.append(result)
    return output


def paired_prompt_contrasts_by_topic(
    topic_rows: Sequence[dict[str, Any]],
    *,
    winner_id: str,
    baseline_ids: Sequence[str],
    bootstrap_resamples: int,
    permutation_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    index: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in topic_rows:
        index[(row["scope"], row["prompt_id"])][int(row["topic_id"])] = row
    output = []
    for scope in SCOPES:
        winner = index.get((scope, winner_id), {})
        if not winner:
            continue
        for baseline_id in baseline_ids:
            baseline = index.get((scope, baseline_id), {})
            common = sorted(set(winner) & set(baseline))
            for metric in METRICS:
                left = [float(winner[topic][metric]) for topic in common]
                right = [float(baseline[topic][metric]) for topic in common]
                differences = [a - b for a, b in zip(left, right)]
                mean, low, high = bootstrap_mean_ci(
                    differences, n_resamples=bootstrap_resamples, seed=seed
                )
                test = paired_permutation_test(
                    left, right, n_resamples=permutation_resamples, seed=seed
                )
                output.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "winner_id": winner_id,
                        "baseline_id": baseline_id,
                        "baseline_label": baseline[common[0]]["prompt_label"],
                        "n_topics": len(common),
                        "winner_mean": statistics.fmean(left),
                        "baseline_mean": statistics.fmean(right),
                        "difference_winner_minus_baseline": mean,
                        "difference_ci_low": low,
                        "difference_ci_high": high,
                        "paired_standardised_effect": test["paired_standardised_effect"],
                        "p_value": test["p_value"],
                        "higher_is_better": metric in {
                            "pairwise_accuracy",
                            "strict_accuracy",
                        },
                    }
                )
    families: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index_number, row in enumerate(output):
        families[(row["scope"], row["metric"])].append(index_number)
    for indices in families.values():
        adjusted = holm_adjust([output[index]["p_value"] for index in indices])
        for index_number, value in zip(indices, adjusted):
            output[index_number]["p_value_holm"] = value
            output[index_number]["significant_holm_0_05"] = value < 0.05
    return output


def build_fixed_partition_uq_rows(
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    *,
    prompt_ids: Sequence[str],
    estimate_seeds: Sequence[int],
    heldout_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Measure whether first-five dispersion predicts future human error."""

    scores, _, labels = build_score_index(rows)
    output = []
    for identifier in prompt_ids:
        for pair in pairs:
            gold = gold_label(pair, "majority_decisive")
            if gold is None:
                continue
            truth = LABEL_TO_RELATION[gold]
            estimate = [
                scores[(identifier, int(pair["external_claim_id"]), pair["argument_a_id"], int(seed))]
                - scores[(identifier, int(pair["external_claim_id"]), pair["argument_b_id"], int(seed))]
                for seed in estimate_seeds
            ]
            heldout = [
                scores[(identifier, int(pair["external_claim_id"]), pair["argument_a_id"], int(seed))]
                - scores[(identifier, int(pair["external_claim_id"]), pair["argument_b_id"], int(seed))]
                for seed in heldout_seeds
            ]
            signals = pair_uncertainty_signals(estimate)
            estimate_decision = _relation(statistics.fmean(estimate))
            heldout_decision = _relation(statistics.fmean(heldout))
            output.append(
                {
                    "prompt_id": identifier,
                    "prompt_label": labels[identifier],
                    "pair_id": pair["pair_id"],
                    "topic_id": int(pair["topic_id"]),
                    "source_topic": pair["source_topic"],
                    "stance": pair["stance"],
                    "gold_label": gold,
                    **signals,
                    "future_reference_error": int(heldout_decision != truth),
                    "future_model_flip": int(heldout_decision != estimate_decision),
                    "future_absolute_deviation": abs(
                        statistics.fmean(heldout) - statistics.fmean(estimate)
                    )
                    / 100.0,
                    "estimate_mean_margin": statistics.fmean(estimate),
                    "heldout_mean_margin": statistics.fmean(heldout),
                }
            )
    return output


def summarise_uq_discrimination(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["prompt_id"]].append(row)
    output = []
    for identifier, values in sorted(grouped.items()):
        variance = [float(row["sampling_variance"]) for row in values]
        deviation = [float(row["future_absolute_deviation"]) for row in values]
        margin = [float(row["margin_uncertainty"]) for row in values]
        error = [int(row["future_reference_error"]) for row in values]
        disagreement = [float(row["pair_disagreement_probability"]) for row in values]
        future_flip = [int(row["future_model_flip"]) for row in values]
        output.append(
            {
                "prompt_id": identifier,
                "prompt_label": values[0]["prompt_label"],
                "n_pairs": len(values),
                "n_topics": len({row["topic_id"] for row in values}),
                "future_reference_error_rate": statistics.fmean(error),
                "future_model_flip_rate": statistics.fmean(future_flip),
                "sampling_variance_spearman_future_deviation": spearman(
                    variance, deviation
                ),
                "margin_uncertainty_auroc_future_error": auroc(margin, error),
                "margin_uncertainty_auprc_future_error": average_precision(margin, error),
                "pair_disagreement_auroc_future_flip": auroc(
                    disagreement, future_flip
                ),
                "pair_disagreement_auprc_future_flip": average_precision(
                    disagreement, future_flip
                ),
            }
        )
    return output


def validation_fitted_calibration_transfer(
    validation_rows: Sequence[dict[str, Any]],
    external_rows: Sequence[dict[str, Any]],
    *,
    winner_id: str,
) -> list[dict[str, Any]]:
    """Apply the internal-validation isotonic error map to E11 without refit."""

    train = [row for row in validation_rows if row["prompt_id"] == winner_id]
    test = [row for row in external_rows if row["prompt_id"] == winner_id]
    if not train or not test:
        raise ValueError("winner rows missing for E11 calibration transfer")
    predictor = _fit_isotonic(
        [float(row["margin_uncertainty"]) for row in train],
        [int(row["future_reference_error"]) for row in train],
    )
    probabilities = [predictor(float(row["margin_uncertainty"])) for row in test]
    outcomes = [int(row["future_reference_error"]) for row in test]
    return [
        {
            "prompt_id": winner_id,
            "calibration_fit": "internal_validation_only",
            "external_refit": False,
            "n_validation_pairs": len(train),
            "n_external_pairs": len(test),
            "n_external_topics": len({row["topic_id"] for row in test}),
            "external_positive_rate": statistics.fmean(outcomes),
            "margin_uncertainty_auroc": auroc(
                [float(row["margin_uncertainty"]) for row in test], outcomes
            ),
            "margin_uncertainty_auprc": average_precision(
                [float(row["margin_uncertainty"]) for row in test], outcomes
            ),
            "brier_validation_fitted": statistics.fmean(
                (probability - outcome) ** 2
                for probability, outcome in zip(probabilities, outcomes)
            ),
            "ece_validation_fitted": expected_calibration_error(probabilities, outcomes),
        }
    ]


def evaluate_pairwise_bayesian_adaptive(
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    *,
    prompt_id_value: str,
    seed_order: Sequence[int],
    checkpoints: Sequence[int],
    confidence_thresholds: Sequence[float],
    prior_concentration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transfer E10's empirical Bayesian policy to independent human pairs.

    Cost is defined for pair-at-a-time serving. It is not the cost of scoring
    every argument in a large topic jointly, where pairs share generations.
    """

    scores, _, _ = build_score_index(rows)
    decisions = []
    for pair in pairs:
        gold = gold_label(pair, "majority_decisive")
        if gold is None:
            continue
        truth = LABEL_TO_RELATION[gold]
        all_margins = [
            scores[
                (
                    prompt_id_value,
                    int(pair["external_claim_id"]),
                    pair["argument_a_id"],
                    int(seed),
                )
            ]
            - scores[
                (
                    prompt_id_value,
                    int(pair["external_claim_id"]),
                    pair["argument_b_id"],
                    int(seed),
                )
            ]
            for seed in seed_order
        ]
        for threshold in confidence_thresholds:
            decision = None
            confidence = 0.0
            used = max(checkpoints)
            probabilities: dict[int, float] = {}
            for checkpoint in checkpoints:
                probabilities = dirichlet_preference_probabilities(
                    all_margins[:checkpoint],
                    prior_concentration=prior_concentration,
                )
                confidence = max(probabilities.values())
                winners = [
                    label for label, value in probabilities.items() if value == confidence
                ]
                used = int(checkpoint)
                if len(winners) == 1 and confidence >= float(threshold):
                    decision = winners[0]
                    break
            decisions.append(
                {
                    "prompt_id": prompt_id_value,
                    "pair_id": pair["pair_id"],
                    "topic_id": int(pair["topic_id"]),
                    "source_topic": pair["source_topic"],
                    "confidence_threshold": float(threshold),
                    "runs_used": used,
                    "posterior_confidence": confidence,
                    "decision": decision,
                    "abstained": decision is None,
                    "human_correct": decision == truth if decision is not None else None,
                    "pairwise_credit": (
                        None
                        if decision is None
                        else 1.0
                        if decision == truth
                        else 0.5
                        if decision == 0 and truth != 0
                        else 0.0
                    ),
                    "posterior_a": probabilities.get(1),
                    "posterior_tie": probabilities.get(0),
                    "posterior_b": probabilities.get(-1),
                }
            )
    summary = []
    for threshold in confidence_thresholds:
        values = [
            row for row in decisions if row["confidence_threshold"] == float(threshold)
        ]
        answered = [row for row in values if not row["abstained"]]
        mean_runs = statistics.fmean(float(row["runs_used"]) for row in values)
        summary.append(
            {
                "prompt_id": prompt_id_value,
                "scope": "majority_decisive",
                "confidence_threshold": float(threshold),
                "n_pairs": len(values),
                "n_topics": len({row["topic_id"] for row in values}),
                "n_answered_pairs": len(answered),
                "pair_coverage": len(answered) / len(values),
                "selective_human_accuracy": (
                    statistics.fmean(float(row["human_correct"]) for row in answered)
                    if answered
                    else None
                ),
                "selective_pairwise_accuracy": (
                    statistics.fmean(float(row["pairwise_credit"]) for row in answered)
                    if answered
                    else None
                ),
                "mean_runs_per_pair": mean_runs,
                "generation_saving_vs_10_runs": 1.0 - mean_runs / 10.0,
                "cost_interpretation": "pair_at_a_time_serving",
                "coverage_guarantee": "none_empirical_bayesian_policy",
            }
        )
    return decisions, summary
