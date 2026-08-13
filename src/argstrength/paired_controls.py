"""Claim-level paired comparisons for frozen APO prompts and controls."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import statistics
from typing import Any, Sequence

from .statistics import bootstrap_mean_ci, holm_adjust, paired_permutation_test
from .uq import _complete_score_cube


METRICS = (
    "pairwise_accuracy",
    "seed_flip_rate",
    "score_variance",
    "mean_tokens",
    "mean_seconds",
)


def _prompt_id(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("prompt_id", row["prompt_condition"]))


def build_claim_metrics(
    rows: Sequence[dict[str, Any]], *, winner_id: str
) -> list[dict[str, Any]]:
    """Reproduce APO aggregate metrics while retaining one row per claim."""

    costs: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        prompt_id = _prompt_id(row)
        costs[(prompt_id, int(row["claim_id"]))].append(row)
        prompt_label = (row.get("metadata") or {}).get("prompt_label")
        labels[prompt_id] = (
            "winner" if prompt_id == winner_id else str(prompt_label or prompt_id)
        )

    output = []
    for cell in _complete_score_cube(rows):
        prompt_id = cell["prompt_id"]
        ranks = sorted(cell["scores_by_rank"])
        accuracies, flips = [], []
        variances = []
        for seed_scores in cell["scores_by_rank"].values():
            variances.append(statistics.variance(seed_scores.values()) / (100.0**2))
        for better_rank, worse_rank in combinations(ranks, 2):
            margins = [
                cell["scores_by_rank"][better_rank][seed]
                - cell["scores_by_rank"][worse_rank][seed]
                for seed in cell["seeds"]
            ]
            mean_margin = statistics.fmean(margins)
            accuracies.append(1.0 if mean_margin > 0 else 0.5 if mean_margin == 0 else 0.0)
            directions = {1 if margin > 0 else -1 if margin < 0 else 0 for margin in margins}
            flips.append(float(len(directions) > 1))
        raw = costs[(prompt_id, int(cell["claim_id"]))]
        item_metadata = dict((raw[0].get("metadata") or {}).get("dataset_item_metadata") or {})
        valid = [row for row in raw if row.get("score_parse_ok") and row.get("score") is not None]
        output.append(
            {
                "model": cell["model"],
                "prompt_id": prompt_id,
                "prompt_label": labels[prompt_id],
                "claim_id": int(cell["claim_id"]),
                "stance": cell["stance"],
                "source_topic": item_metadata.get("source_topic"),
                "source_group_id": item_metadata.get("source_group_id"),
                "panel": item_metadata.get("panel"),
                "n_seeds": len(cell["seeds"]),
                "pairwise_accuracy": statistics.fmean(accuracies),
                "seed_flip_rate": statistics.fmean(flips),
                "score_variance": statistics.fmean(variances),
                "mean_tokens": statistics.fmean(float(row.get("generated_tokens", 0)) for row in raw),
                "mean_seconds": statistics.fmean(float(row.get("generation_seconds", 0)) for row in raw),
                "invalid_rate": 1.0 - len(valid) / len(raw),
            }
        )
    return output


def summarise_prompts(
    claim_rows: Sequence[dict[str, Any]], *, bootstrap_resamples: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in claim_rows:
        grouped[row["prompt_id"]].append(row)
    output = []
    for prompt_id, rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "model": rows[0]["model"],
            "prompt_id": prompt_id,
            "prompt_label": rows[0]["prompt_label"],
            "n_claims": len(rows),
            "n_seeds": rows[0]["n_seeds"],
            "invalid_rate": statistics.fmean(row["invalid_rate"] for row in rows),
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


def paired_prompt_contrasts(
    claim_rows: Sequence[dict[str, Any]],
    *,
    winner_id: str,
    baseline_ids: Sequence[str] | None = None,
    bootstrap_resamples: int = 10_000,
    permutation_resamples: int = 100_000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compare winner minus each baseline, correcting within metric and scope."""

    index: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in claim_rows:
        index[row["prompt_id"]][int(row["claim_id"])] = row
    if winner_id not in index:
        raise ValueError(f"winner {winner_id!r} is absent")
    baselines = list(baseline_ids or sorted(set(index) - {winner_id}))
    missing = set(baselines) - set(index)
    if missing:
        raise ValueError(f"baseline prompt IDs are absent: {sorted(missing)}")

    output = []
    scopes = ("all", "support", "attack")
    for scope in scopes:
        for baseline_id in baselines:
            common = sorted(set(index[winner_id]) & set(index[baseline_id]))
            if scope != "all":
                common = [
                    claim_id
                    for claim_id in common
                    if index[winner_id][claim_id]["stance"] == scope
                ]
            for metric in METRICS:
                left = [float(index[winner_id][claim_id][metric]) for claim_id in common]
                right = [float(index[baseline_id][claim_id][metric]) for claim_id in common]
                differences = [a - b for a, b in zip(left, right)]
                mean, low, high = bootstrap_mean_ci(
                    differences, n_resamples=bootstrap_resamples, seed=seed
                )
                test = paired_permutation_test(
                    left, right, n_resamples=permutation_resamples, seed=seed
                )
                output.append(
                    {
                        "model": index[winner_id][common[0]]["model"],
                        "scope": scope,
                        "metric": metric,
                        "winner_id": winner_id,
                        "baseline_id": baseline_id,
                        "baseline_label": index[baseline_id][common[0]]["prompt_label"],
                        "n_claims": len(common),
                        "winner_mean": statistics.fmean(left),
                        "baseline_mean": statistics.fmean(right),
                        "difference_winner_minus_baseline": mean,
                        "difference_ci_low": low,
                        "difference_ci_high": high,
                        "paired_standardised_effect": test["paired_standardised_effect"],
                        "p_value": test["p_value"],
                        "higher_is_better": metric == "pairwise_accuracy",
                    }
                )

    families: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index_number, row in enumerate(output):
        families[(row["scope"], row["metric"])].append(index_number)
    for indices in families.values():
        adjusted = holm_adjust([output[index_number]["p_value"] for index_number in indices])
        for index_number, p_holm in zip(indices, adjusted):
            output[index_number]["p_value_holm"] = p_holm
            output[index_number]["significant_holm_0_05"] = p_holm < 0.05
    return output
