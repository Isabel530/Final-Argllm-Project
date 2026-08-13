"""Extended reliability analyses for the argument-strength studies.

The functions in this module deliberately keep claims as the inferential unit.
Argument pairs, prompts, and seeds are nested observations and are never treated
as independent sample-size multipliers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import csv
import json
import math
from pathlib import Path
import statistics
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import numpy as np

from .metrics import spearman_from_scores
from .statistics import holm_adjust, paired_permutation_test


PRIMARY_PAIR_METRICS = (
    "pairwise_accuracy",
    "tie_rate",
    "seed_flip_rate",
)


def read_jsonl_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read JSON objects from one or more JSONL files."""

    rows: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    return rows


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write dictionaries to CSV with a stable union of columns."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _thinking(row: dict[str, Any]) -> bool:
    return bool(row.get("thinking", row.get("metadata", {}).get("thinking", False)))


def _score_ok(row: dict[str, Any]) -> bool:
    return bool(row.get("score_parse_ok")) and row.get("score") is not None


def _relation(value: float) -> int:
    """Direction relative to the reference order: correct/tie/wrong."""

    return 1 if value > 0 else -1 if value < 0 else 0


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Vectorised percentile bootstrap CI for independent claim-level values."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("values cannot be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    estimates = array[indices].mean(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(array.mean()), float(low), float(high)


def complete_score_cells(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build complete model/prompt/claim cells with every configured seed.

    Required seeds are inferred at the study/model/prompt/decoding/thinking
    setting level. A cell with one missing or unparseable argument score is
    excluded rather than silently averaged over fewer repetitions.
    """

    setting_seeds: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("reference_rank") is None:
            continue
        setting = (
            row["study"],
            row["model_key"],
            row["prompt_condition"],
            row["decoding"],
            _thinking(row),
        )
        setting_seeds[setting].add(int(row["seed"]))
        grouped[
            setting
            + (
                int(row["claim_id"]),
                row["stance"],
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        required_seeds = setting_seeds[key[:5]]
        by_rank: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in values:
            rank = int(row["reference_rank"])
            seed = int(row["seed"])
            if seed in by_rank[rank]:
                raise ValueError(f"Duplicate score for cell={key}, rank={rank}, seed={seed}")
            by_rank[rank][seed] = row
        if len(by_rank) < 2:
            continue
        complete = all(
            set(seed_rows) == required_seeds
            and all(_score_ok(row) for row in seed_rows.values())
            for seed_rows in by_rank.values()
        )
        if not complete:
            continue

        scores_by_rank = {
            rank: {seed: float(row["score"]) for seed, row in seed_rows.items()}
            for rank, seed_rows in by_rank.items()
        }
        arguments_by_rank = {
            rank: next(iter(seed_rows.values())).get("argument", "")
            for rank, seed_rows in by_rank.items()
        }
        claim = values[0].get("claim", "")
        averaged_scores = {
            rank: statistics.fmean(seed_scores.values())
            for rank, seed_scores in scores_by_rank.items()
        }
        argument_seed_stds = [
            statistics.stdev(seed_scores.values())
            for seed_scores in scores_by_rank.values()
            if len(seed_scores) > 1
        ]
        output.append(
            {
                "study": key[0],
                "model": key[1],
                "prompt": key[2],
                "decoding": key[3],
                "thinking": key[4],
                "claim_id": key[5],
                "stance": key[6],
                "claim": claim,
                "n_seeds": len(required_seeds),
                "seeds": tuple(sorted(required_seeds)),
                "scores_by_rank": scores_by_rank,
                "averaged_scores": averaged_scores,
                "arguments_by_rank": arguments_by_rank,
                "mean_argument_seed_std": (
                    statistics.fmean(argument_seed_stds)
                    if argument_seed_stds
                    else 0.0
                ),
            }
        )
    return output


def pair_observations(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Construct reference-oriented pair outcomes from pointwise scores."""

    output: list[dict[str, Any]] = []
    for cell in cells:
        averaged = cell["averaged_scores"]
        claim_range = max(averaged.values()) - min(averaged.values())
        for better_rank, worse_rank in combinations(sorted(averaged), 2):
            raw_margin = averaged[better_rank] - averaged[worse_rank]
            relation = _relation(raw_margin)
            seed_relations = []
            for seed in cell["seeds"]:
                seed_margin = (
                    cell["scores_by_rank"][better_rank][seed]
                    - cell["scores_by_rank"][worse_rank][seed]
                )
                seed_relations.append(_relation(seed_margin))
            agreement = [
                float(left == right)
                for left, right in combinations(seed_relations, 2)
            ]
            output.append(
                {
                    "study": cell["study"],
                    "model": cell["model"],
                    "prompt": cell["prompt"],
                    "decoding": cell["decoding"],
                    "thinking": cell["thinking"],
                    "claim_id": cell["claim_id"],
                    "stance": cell["stance"],
                    "claim": cell["claim"],
                    "better_rank": better_rank,
                    "worse_rank": worse_rank,
                    "pair_id": f"rank_{better_rank}_vs_rank_{worse_rank}",
                    "rank_distance": worse_rank - better_rank,
                    "better_argument": cell["arguments_by_rank"][better_rank],
                    "worse_argument": cell["arguments_by_rank"][worse_rank],
                    "better_score": averaged[better_rank],
                    "worse_score": averaged[worse_rank],
                    "raw_margin": raw_margin,
                    "normalized_margin": (
                        raw_margin / claim_range if claim_range > 0 else None
                    ),
                    "relation": relation,
                    "pairwise_accuracy": (
                        1.0 if relation == 1 else 0.5 if relation == 0 else 0.0
                    ),
                    "strict_pairwise_accuracy": float(raw_margin >= 10),
                    "tie_rate": float(relation == 0),
                    "wrong_rate": float(relation == -1),
                    "seed_relations": "|".join(map(str, seed_relations)),
                    "seed_consistency": statistics.fmean(agreement),
                    "seed_flip_rate": float(len(set(seed_relations)) > 1),
                }
            )
    return output


def _claim_distance_rows(
    pairs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[
            (
                row["study"],
                row["model"],
                row["prompt"],
                row["decoding"],
                row["thinking"],
                row["claim_id"],
                row["stance"],
                row["rank_distance"],
            )
        ].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        output.append(
            {
                "study": key[0],
                "model": key[1],
                "prompt": key[2],
                "decoding": key[3],
                "thinking": key[4],
                "claim_id": key[5],
                "stance": key[6],
                "rank_distance": key[7],
                "n_pairs": len(values),
                "pairwise_accuracy": statistics.fmean(
                    row["pairwise_accuracy"] for row in values
                ),
                "strict_pairwise_accuracy": statistics.fmean(
                    row["strict_pairwise_accuracy"] for row in values
                ),
                "tie_rate": statistics.fmean(row["tie_rate"] for row in values),
                "wrong_rate": statistics.fmean(row["wrong_rate"] for row in values),
                "seed_consistency": statistics.fmean(
                    row["seed_consistency"] for row in values
                ),
                "seed_flip_rate": statistics.fmean(
                    row["seed_flip_rate"] for row in values
                ),
                "raw_margin": statistics.fmean(row["raw_margin"] for row in values),
            }
        )
    return output


def rank_distance_summary(
    pairs: Sequence[dict[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return claim-distance rows, aggregate CIs, and paired effects."""

    claim_rows = _claim_distance_rows(pairs)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in claim_rows:
        grouped[
            (
                row["study"],
                row["model"],
                row["prompt"],
                row["decoding"],
                row["thinking"],
                row["rank_distance"],
            )
        ].append(row)

    aggregate: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        result = {
            "study": key[0],
            "model": key[1],
            "prompt": key[2],
            "decoding": key[3],
            "thinking": key[4],
            "rank_distance": key[5],
            "n_claims": len(values),
            "n_pairs": sum(row["n_pairs"] for row in values),
        }
        for metric in (
            "pairwise_accuracy",
            "strict_pairwise_accuracy",
            "tie_rate",
            "wrong_rate",
            "seed_consistency",
            "seed_flip_rate",
            "raw_margin",
        ):
            metric_values = [float(row[metric]) for row in values]
            mean, low, high = bootstrap_mean_ci(
                metric_values,
                n_resamples=bootstrap_resamples,
                seed=42 + int(key[5]),
            )
            result[f"mean_{metric}"] = mean
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        aggregate.append(result)

    effects: list[dict[str, Any]] = []
    effect_metrics = PRIMARY_PAIR_METRICS

    # Each non-direct prompt is paired with Direct on the same claims.
    prompt_groups: dict[tuple[Any, ...], dict[str, dict[int, dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for row in claim_rows:
        setting = (
            row["study"],
            row["model"],
            row["decoding"],
            row["thinking"],
            row["rank_distance"],
        )
        prompt_groups[setting][row["prompt"]][row["claim_id"]] = row
    for setting, prompts in sorted(prompt_groups.items()):
        if "direct" not in prompts:
            continue
        for prompt in sorted(set(prompts) - {"direct"}):
            common = sorted(set(prompts["direct"]) & set(prompts[prompt]))
            for metric in effect_metrics:
                left = [prompts[prompt][claim][metric] for claim in common]
                right = [prompts["direct"][claim][metric] for claim in common]
                if not left:
                    continue
                differences = [a - b for a, b in zip(left, right)]
                mean, low, high = bootstrap_mean_ci(
                    differences,
                    n_resamples=bootstrap_resamples,
                    seed=73 + int(setting[4]),
                )
                test = paired_permutation_test(
                    left, right, n_resamples=10_000, seed=42
                )
                effects.append(
                    {
                        "effect_type": "prompt_vs_direct",
                        "study": setting[0],
                        "model": setting[1],
                        "decoding": setting[2],
                        "thinking": setting[3],
                        "rank_distance": setting[4],
                        "contrast": f"{prompt}_minus_direct",
                        "metric": metric,
                        "n_claims": len(common),
                        "mean_difference": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "paired_standardised_effect": test[
                            "paired_standardised_effect"
                        ],
                        "p_value": test["p_value"],
                    }
                )

    # Think versus No-Think within the same model/prompt/claim.
    thinking_groups: dict[
        tuple[Any, ...], dict[bool, dict[int, dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in claim_rows:
        setting = (
            row["study"],
            row["model"],
            row["prompt"],
            row["decoding"],
            row["rank_distance"],
        )
        thinking_groups[setting][row["thinking"]][row["claim_id"]] = row
    for setting, modes in sorted(thinking_groups.items()):
        if False not in modes or True not in modes:
            continue
        common = sorted(set(modes[False]) & set(modes[True]))
        for metric in effect_metrics:
            left = [modes[True][claim][metric] for claim in common]
            right = [modes[False][claim][metric] for claim in common]
            differences = [a - b for a, b in zip(left, right)]
            mean, low, high = bootstrap_mean_ci(
                differences,
                n_resamples=bootstrap_resamples,
                seed=107 + int(setting[4]),
            )
            test = paired_permutation_test(
                left, right, n_resamples=10_000, seed=42
            )
            effects.append(
                {
                    "effect_type": "thinking",
                    "study": setting[0],
                    "model": setting[1],
                    "prompt": setting[2],
                    "decoding": setting[3],
                    "rank_distance": setting[4],
                    "contrast": "think_minus_no_think",
                    "metric": metric,
                    "n_claims": len(common),
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "paired_standardised_effect": test[
                        "paired_standardised_effect"
                    ],
                    "p_value": test["p_value"],
                }
            )

    families: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(effects):
        families[
            (
                row["effect_type"],
                row["study"],
                row["model"],
                row["rank_distance"],
            )
        ].append(index)
    for indices in families.values():
        adjusted = holm_adjust(effects[index]["p_value"] for index in indices)
        for index, value in zip(indices, adjusted):
            effects[index]["holm_p_value"] = value
            effects[index]["significant_0_05_holm"] = value < 0.05

    return claim_rows, aggregate, effects


def qwen_tie_transition_analysis(
    pairs: Sequence[dict[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Trace Qwen pairs tied under No-Think into their Think outcomes."""

    selected = [
        row
        for row in pairs
        if row["study"] == "reasoning_study" and row["model"] == "qwen3_8b"
    ]
    grouped: dict[tuple[Any, ...], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        key = (
            row["prompt"],
            row["decoding"],
            row["claim_id"],
            row["rank_distance"],
            row["better_rank"],
            row["worse_rank"],
        )
        grouped[key][row["thinking"]] = row

    detail: list[dict[str, Any]] = []
    for key, modes in sorted(grouped.items()):
        if False not in modes or True not in modes:
            continue
        no_think = modes[False]
        think = modes[True]
        if no_think["relation"] != 0:
            continue
        category = {
            1: "Tie_to_Correct",
            0: "Tie_to_Tie",
            -1: "Tie_to_Wrong",
        }[think["relation"]]
        seed_relations = [int(value) for value in think["seed_relations"].split("|")]
        detail.append(
            {
                "model": "qwen3_8b",
                "prompt": key[0],
                "decoding": key[1],
                "claim_id": key[2],
                "stance": think["stance"],
                "claim": think["claim"],
                "rank_distance": key[3],
                "better_rank": key[4],
                "worse_rank": key[5],
                "pair_id": think["pair_id"],
                "transition": category,
                "think_relation": think["relation"],
                "think_raw_margin": think["raw_margin"],
                "think_seed_relations": think["seed_relations"],
                "think_seed_disagreement": len(set(seed_relations)) > 1,
                "think_contains_correct_and_wrong_seeds": (
                    1 in seed_relations and -1 in seed_relations
                ),
            }
        )

    summary: list[dict[str, Any]] = []
    categories = ("Tie_to_Correct", "Tie_to_Wrong", "Tie_to_Tie")
    for distance in ("all", 1, 2, 3):
        available = [
            row
            for row in detail
            if distance == "all" or row["rank_distance"] == distance
        ]
        if not available:
            continue
        by_claim: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in available:
            by_claim[row["claim_id"]].append(row)
        result: dict[str, Any] = {
            "rank_distance": distance,
            "n_claims_with_no_think_ties": len(by_claim),
            "n_no_think_tied_pairs": len(available),
        }
        for category in categories:
            values = [
                sum(row["transition"] == category for row in rows) / len(rows)
                for rows in by_claim.values()
            ]
            mean, low, high = bootstrap_mean_ci(
                values,
                n_resamples=bootstrap_resamples,
                seed=131 + (0 if distance == "all" else int(distance)),
            )
            prefix = category.lower()
            result[f"{prefix}_rate"] = mean
            result[f"{prefix}_ci_low"] = low
            result[f"{prefix}_ci_high"] = high
        for field in (
            "think_seed_disagreement",
            "think_contains_correct_and_wrong_seeds",
        ):
            values = [
                sum(bool(row[field]) for row in rows) / len(rows)
                for rows in by_claim.values()
            ]
            mean, low, high = bootstrap_mean_ci(
                values,
                n_resamples=bootstrap_resamples,
                seed=149 + (0 if distance == "all" else int(distance)),
            )
            result[f"{field}_rate"] = mean
            result[f"{field}_ci_low"] = low
            result[f"{field}_ci_high"] = high
        summary.append(result)
    return detail, summary


def score_scale_analysis(
    rows: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Describe score use, digit heaping, claim ranges, and pair margins."""

    score_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if not _score_ok(row):
            continue
        score_groups[
            (
                row["study"],
                row["model_key"],
                row["prompt_condition"],
                row["decoding"],
                _thinking(row),
            )
        ].append(float(row["score"]))
    range_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for cell in cells:
        key = (
            cell["study"],
            cell["model"],
            cell["prompt"],
            cell["decoding"],
            cell["thinking"],
        )
        scores = list(cell["averaged_scores"].values())
        range_groups[key].append(max(scores) - min(scores))

    summary: list[dict[str, Any]] = []
    histogram: list[dict[str, Any]] = []
    for key, scores in sorted(score_groups.items()):
        array = np.asarray(scores)
        counts = Counter(int(score) for score in scores)
        probabilities = np.asarray(list(counts.values()), dtype=float) / len(scores)
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        normalized_entropy = entropy / math.log(101)
        ranges = range_groups.get(key, [])
        summary.append(
            {
                "study": key[0],
                "model": key[1],
                "prompt": key[2],
                "decoding": key[3],
                "thinking": key[4],
                "n_scores": len(scores),
                "mean_score": float(array.mean()),
                "score_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "median_score": float(np.median(array)),
                "score_q1": float(np.quantile(array, 0.25)),
                "score_q3": float(np.quantile(array, 0.75)),
                "min_score": float(array.min()),
                "max_score": float(array.max()),
                "n_unique_scores": len(counts),
                "multiple_of_10_rate": float(np.mean(array % 10 == 0)),
                "multiple_of_5_rate": float(np.mean(array % 5 == 0)),
                "ends_in_5_rate": float(np.mean(array % 10 == 5)),
                "score_use_entropy": entropy,
                "score_use_entropy_normalized_101": normalized_entropy,
                "n_complete_claim_cells": len(ranges),
                "mean_within_claim_score_range": (
                    statistics.fmean(ranges) if ranges else None
                ),
                "median_within_claim_score_range": (
                    statistics.median(ranges) if ranges else None
                ),
            }
        )
        for score in range(101):
            histogram.append(
                {
                    "study": key[0],
                    "model": key[1],
                    "prompt": key[2],
                    "decoding": key[3],
                    "thinking": key[4],
                    "score": score,
                    "count": counts.get(score, 0),
                    "rate": counts.get(score, 0) / len(scores),
                }
            )

    margin_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        for distance in ("all", row["rank_distance"]):
            margin_groups[
                (
                    row["study"],
                    row["model"],
                    row["prompt"],
                    row["decoding"],
                    row["thinking"],
                    distance,
                )
            ].append(row)
    margin_rows: list[dict[str, Any]] = []
    for key, values in sorted(margin_groups.items(), key=lambda item: str(item[0])):
        raw = [row["raw_margin"] for row in values]
        normalized = [
            row["normalized_margin"]
            for row in values
            if row["normalized_margin"] is not None
        ]
        margin_rows.append(
            {
                "study": key[0],
                "model": key[1],
                "prompt": key[2],
                "decoding": key[3],
                "thinking": key[4],
                "rank_distance": key[5],
                "n_pairs": len(values),
                "mean_raw_margin": statistics.fmean(raw),
                "median_raw_margin": statistics.median(raw),
                "raw_margin_sd": statistics.stdev(raw) if len(raw) > 1 else 0.0,
                "n_normalized_pairs": len(normalized),
                "mean_normalized_margin": (
                    statistics.fmean(normalized) if normalized else None
                ),
                "median_normalized_margin": (
                    statistics.median(normalized) if normalized else None
                ),
                "strict_pairwise_accuracy_raw_10": statistics.fmean(
                    row["strict_pairwise_accuracy"] for row in values
                ),
            }
        )
    return summary, histogram, margin_rows


def claim_difficulty_analysis(
    cells: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    *,
    difficult_n: int = 20,
    disagreement_n: int = 10,
    reference_review_n: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank Main Study claims by error, disagreement, and reference tension."""

    main_pairs = [row for row in pairs if row["study"] == "main_study"]
    main_cells = [cell for cell in cells if cell["study"] == "main_study"]
    by_claim: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in main_pairs:
        by_claim[row["claim_id"]].append(row)

    # Model disagreement is evaluated within prompt and pair.
    model_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    prompt_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    pair_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in main_pairs:
        model_groups[
            (
                row["claim_id"],
                row["prompt"],
                row["better_rank"],
                row["worse_rank"],
            )
        ].append(row)
        prompt_groups[
            (
                row["claim_id"],
                row["model"],
                row["better_rank"],
                row["worse_rank"],
            )
        ].append(row)
        pair_groups[
            (row["claim_id"], row["better_rank"], row["worse_rank"])
        ].append(row)

    model_disagreement_by_claim: dict[int, list[float]] = defaultdict(list)
    model_opposition_by_claim: dict[int, list[float]] = defaultdict(list)
    for key, values in model_groups.items():
        relations = {row["relation"] for row in values}
        model_disagreement_by_claim[key[0]].append(float(len(relations) > 1))
        model_opposition_by_claim[key[0]].append(
            float(1 in relations and -1 in relations)
        )
    prompt_disagreement_by_claim: dict[int, list[float]] = defaultdict(list)
    for key, values in prompt_groups.items():
        prompt_disagreement_by_claim[key[0]].append(
            float(len({row["relation"] for row in values}) > 1)
        )

    pair_detail: list[dict[str, Any]] = []
    reference_candidates_by_claim: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key, values in sorted(pair_groups.items()):
        wrong_rate = statistics.fmean(row["wrong_rate"] for row in values)
        tie_rate = statistics.fmean(row["tie_rate"] for row in values)
        correct_rate = statistics.fmean(
            float(row["relation"] == 1) for row in values
        )
        candidate = wrong_rate >= 2 / 3
        result = {
            "claim_id": key[0],
            "stance": values[0]["stance"],
            "claim": values[0]["claim"],
            "better_rank": key[1],
            "worse_rank": key[2],
            "pair_id": values[0]["pair_id"],
            "rank_distance": values[0]["rank_distance"],
            "better_argument": values[0]["better_argument"],
            "worse_argument": values[0]["worse_argument"],
            "n_model_prompt_cells": len(values),
            "mean_pairwise_accuracy": statistics.fmean(
                row["pairwise_accuracy"] for row in values
            ),
            "correct_rate": correct_rate,
            "tie_rate": tie_rate,
            "wrong_rate": wrong_rate,
            "mean_seed_flip_rate": statistics.fmean(
                row["seed_flip_rate"] for row in values
            ),
            "reference_review_candidate": candidate,
        }
        pair_detail.append(result)
        if candidate:
            reference_candidates_by_claim[key[0]].append(result)

    cell_std_by_claim: dict[int, list[float]] = defaultdict(list)
    metadata_by_claim: dict[int, dict[str, Any]] = {}
    for cell in main_cells:
        cell_std_by_claim[cell["claim_id"]].append(cell["mean_argument_seed_std"])
        metadata_by_claim[cell["claim_id"]] = cell

    ranking: list[dict[str, Any]] = []
    for claim_id, values in sorted(by_claim.items()):
        adjacent = [row for row in values if row["rank_distance"] == 1]
        metadata = metadata_by_claim[claim_id]
        candidates = reference_candidates_by_claim.get(claim_id, [])
        result: dict[str, Any] = {
            "claim_id": claim_id,
            "stance": metadata["stance"],
            "claim": metadata["claim"],
            "n_pair_cells": len(values),
            "mean_pairwise_accuracy": statistics.fmean(
                row["pairwise_accuracy"] for row in values
            ),
            "difficulty_score": 1
            - statistics.fmean(row["pairwise_accuracy"] for row in values),
            "adjacent_pairwise_accuracy": statistics.fmean(
                row["pairwise_accuracy"] for row in adjacent
            ),
            "wrong_rate": statistics.fmean(row["wrong_rate"] for row in values),
            "tie_rate": statistics.fmean(row["tie_rate"] for row in values),
            "seed_flip_rate": statistics.fmean(
                row["seed_flip_rate"] for row in values
            ),
            "mean_argument_seed_std": statistics.fmean(
                cell_std_by_claim[claim_id]
            ),
            "model_disagreement_rate": statistics.fmean(
                model_disagreement_by_claim[claim_id]
            ),
            "model_direct_opposition_rate": statistics.fmean(
                model_opposition_by_claim[claim_id]
            ),
            "prompt_disagreement_rate": statistics.fmean(
                prompt_disagreement_by_claim[claim_id]
            ),
            "reference_review_candidate_pairs": len(candidates),
            "max_reference_wrong_rate": (
                max(row["wrong_rate"] for row in candidates) if candidates else 0.0
            ),
        }
        for rank, argument in sorted(metadata["arguments_by_rank"].items()):
            result[f"argument_rank_{rank}"] = argument
        ranking.append(result)

    for rank, row in enumerate(
        sorted(
            ranking,
            key=lambda value: (
                -value["difficulty_score"],
                value["claim_id"],
            ),
        ),
        start=1,
    ):
        row["difficulty_rank"] = rank
    for rank, row in enumerate(
        sorted(
            ranking,
            key=lambda value: (
                -value["model_disagreement_rate"],
                -value["prompt_disagreement_rate"],
                value["claim_id"],
            ),
        ),
        start=1,
    ):
        row["disagreement_rank"] = rank
    for rank, row in enumerate(
        sorted(
            ranking,
            key=lambda value: (
                -value["reference_review_candidate_pairs"],
                -value["max_reference_wrong_rate"],
                value["claim_id"],
            ),
        ),
        start=1,
    ):
        row["reference_review_rank"] = rank

    audit_ids = {
        row["claim_id"]
        for row in ranking
        if row["difficulty_rank"] <= difficult_n
        or row["disagreement_rank"] <= disagreement_n
        or row["reference_review_rank"] <= reference_review_n
    }
    audit: list[dict[str, Any]] = []
    for row in sorted(
        (row for row in ranking if row["claim_id"] in audit_ids),
        key=lambda value: (value["difficulty_rank"], value["claim_id"]),
    ):
        audit_row = dict(row)
        audit_row["selected_as_difficult"] = row["difficulty_rank"] <= difficult_n
        audit_row["selected_as_model_disagreement"] = (
            row["disagreement_rank"] <= disagreement_n
        )
        audit_row["selected_for_reference_review"] = (
            row["reference_review_rank"] <= reference_review_n
        )
        audit_row["manual_error_category"] = ""
        audit_row["manual_notes"] = ""
        audit.append(audit_row)
    return ranking, pair_detail, audit


def accuracy_stability_cost_table(
    rows: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine semantic accuracy, seed stability, compliance, and cost."""

    claim_pair_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        claim_pair_groups[
            (
                row["study"],
                row["model"],
                row["prompt"],
                row["decoding"],
                row["thinking"],
                row["claim_id"],
            )
        ].append(row)
    setting_claims: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    for key, values in claim_pair_groups.items():
        setting_claims[key[:5]].append(
            {
                "pairwise_accuracy": statistics.fmean(
                    row["pairwise_accuracy"] for row in values
                ),
                "seed_consistency": statistics.fmean(
                    row["seed_consistency"] for row in values
                ),
                "seed_flip_rate": statistics.fmean(
                    row["seed_flip_rate"] for row in values
                ),
            }
        )
    setting_stds: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for cell in cells:
        setting_stds[
            (
                cell["study"],
                cell["model"],
                cell["prompt"],
                cell["decoding"],
                cell["thinking"],
            )
        ].append(cell["mean_argument_seed_std"])
    raw_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_groups[
            (
                row["study"],
                row["model_key"],
                row["prompt_condition"],
                row["decoding"],
                _thinking(row),
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for key, claim_values in sorted(setting_claims.items()):
        raw_values = raw_groups[key]
        output.append(
            {
                "study": key[0],
                "model": key[1],
                "prompt": key[2],
                "decoding": key[3],
                "thinking": key[4],
                "n_claims": len(claim_values),
                "mean_pairwise_accuracy": statistics.fmean(
                    row["pairwise_accuracy"] for row in claim_values
                ),
                "mean_seed_consistency": statistics.fmean(
                    row["seed_consistency"] for row in claim_values
                ),
                "mean_seed_flip_rate": statistics.fmean(
                    row["seed_flip_rate"] for row in claim_values
                ),
                "mean_argument_score_std": statistics.fmean(setting_stds[key]),
                "n_generations": len(raw_values),
                "format_compliance_rate": statistics.fmean(
                    float(bool(row.get("format_ok"))) for row in raw_values
                ),
                "score_extraction_rate": statistics.fmean(
                    float(_score_ok(row)) for row in raw_values
                ),
                "mean_generated_tokens": statistics.fmean(
                    float(row.get("generated_tokens", 0)) for row in raw_values
                ),
                "mean_generation_seconds": statistics.fmean(
                    float(row.get("generation_seconds", 0)) for row in raw_values
                ),
            }
        )

    # A cell is dominated only if another cell for the same study/model is no
    # worse in accuracy and consistency and no more expensive in tokens/time.
    for row in output:
        peers = [
            other
            for other in output
            if other["study"] == row["study"]
            and other["model"] == row["model"]
            and other is not row
        ]
        dominated_by = []
        for other in peers:
            comparisons = (
                other["mean_pairwise_accuracy"] >= row["mean_pairwise_accuracy"],
                other["mean_seed_consistency"] >= row["mean_seed_consistency"],
                other["mean_generated_tokens"] <= row["mean_generated_tokens"],
                other["mean_generation_seconds"] <= row["mean_generation_seconds"],
            )
            strict = (
                other["mean_pairwise_accuracy"] > row["mean_pairwise_accuracy"]
                or other["mean_seed_consistency"] > row["mean_seed_consistency"]
                or other["mean_generated_tokens"] < row["mean_generated_tokens"]
                or other["mean_generation_seconds"]
                < row["mean_generation_seconds"]
            )
            if all(comparisons) and strict:
                dominated_by.append(
                    f"{other['prompt']}|{other['decoding']}|think={other['thinking']}"
                )
        row["pareto_status_within_study_model"] = (
            "dominated" if dominated_by else "non_dominated"
        )
        row["dominated_by"] = ";".join(dominated_by)
    return output


def _student_t_distribution(df: int):
    """Return scipy's t distribution if available, otherwise normal fallback."""

    try:
        from scipy.stats import t  # type: ignore

        return (
            lambda probability: float(t.ppf(probability, df)),
            lambda value: float(t.cdf(value, df)),
            lambda value: float(t.sf(value, df)),
            "student_t",
        )
    except ImportError:
        normal = NormalDist()
        return (
            normal.inv_cdf,
            normal.cdf,
            lambda value: 1 - normal.cdf(value),
            "normal_approximation",
        )


def paired_tost(
    differences: Sequence[float],
    *,
    lower_bound: float,
    upper_bound: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired TOST over already paired claim-level differences."""

    if not differences:
        raise ValueError("differences cannot be empty")
    if not lower_bound < upper_bound:
        raise ValueError("lower_bound must be below upper_bound")
    values = [float(value) for value in differences]
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    if sd == 0:
        inside = lower_bound < mean < upper_bound
        return {
            "n_claims": n,
            "mean_difference": mean,
            "difference_sd": sd,
            "ci_level": 1 - 2 * alpha,
            "ci_low": mean,
            "ci_high": mean,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "p_lower": 0.0 if mean > lower_bound else 1.0,
            "p_upper": 0.0 if mean < upper_bound else 1.0,
            "equivalence_p_value": 0.0 if inside else 1.0,
            "equivalent_0_05": inside,
            "distribution": "degenerate",
        }
    standard_error = sd / math.sqrt(n)
    quantile, cdf, sf, distribution_name = _student_t_distribution(n - 1)
    critical = quantile(1 - alpha)
    ci_low = mean - critical * standard_error
    ci_high = mean + critical * standard_error
    t_lower = (mean - lower_bound) / standard_error
    t_upper = (mean - upper_bound) / standard_error
    p_lower = sf(t_lower)
    p_upper = cdf(t_upper)
    equivalence_p = max(p_lower, p_upper)
    return {
        "n_claims": n,
        "mean_difference": mean,
        "difference_sd": sd,
        "ci_level": 1 - 2 * alpha,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "equivalence_p_value": equivalence_p,
        "equivalent_0_05": equivalence_p < alpha,
        "distribution": distribution_name,
    }


def equivalence_tests(
    cells: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    *,
    pairwise_bound: float = 0.02,
    spearman_bound: float = 0.03,
) -> list[dict[str, Any]]:
    """Secondary prompt-equivalence analysis for Main Study primary metrics."""

    pair_by_cell_claim: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        if row["study"] != "main_study":
            continue
        pair_by_cell_claim[
            (
                row["model"],
                row["prompt"],
                row["claim_id"],
            )
        ].append(row)
    metric_rows: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, values in pair_by_cell_claim.items():
        metric_rows[key] = {
            "pairwise_accuracy": statistics.fmean(
                row["pairwise_accuracy"] for row in values
            )
        }
    for cell in cells:
        if cell["study"] != "main_study":
            continue
        ranks = sorted(cell["averaged_scores"])
        scores = [cell["averaged_scores"][rank] for rank in ranks]
        spearman = spearman_from_scores(ranks, scores)
        if spearman is not None:
            metric_rows[
                (cell["model"], cell["prompt"], cell["claim_id"])
            ]["spearman"] = spearman

    output: list[dict[str, Any]] = []
    models = sorted({key[0] for key in metric_rows})
    prompts = ("role", "rubric", "role_rubric")
    for model in models:
        for prompt in prompts:
            for metric, bound in (
                ("pairwise_accuracy", pairwise_bound),
                ("spearman", spearman_bound),
            ):
                direct = {
                    claim: values[metric]
                    for (row_model, row_prompt, claim), values in metric_rows.items()
                    if row_model == model
                    and row_prompt == "direct"
                    and metric in values
                }
                comparison = {
                    claim: values[metric]
                    for (row_model, row_prompt, claim), values in metric_rows.items()
                    if row_model == model
                    and row_prompt == prompt
                    and metric in values
                }
                common = sorted(set(direct) & set(comparison))
                differences = [
                    comparison[claim] - direct[claim] for claim in common
                ]
                result = paired_tost(
                    differences,
                    lower_bound=-bound,
                    upper_bound=bound,
                )
                output.append(
                    {
                        "study": "main_study",
                        "model": model,
                        "contrast": f"{prompt}_minus_direct",
                        "metric": metric,
                        **result,
                    }
                )

    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        families[row["model"]].append(index)
    for indices in families.values():
        adjusted = holm_adjust(
            output[index]["equivalence_p_value"] for index in indices
        )
        for index, value in zip(indices, adjusted):
            output[index]["holm_equivalence_p_value"] = value
            output[index]["equivalent_0_05_holm"] = value < 0.05
    return output
