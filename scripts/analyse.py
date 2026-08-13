#!/usr/bin/env python3
"""Create reproducible descriptive and inferential tables from raw JSONL."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.metrics import claim_metrics, spearman_from_scores  # noqa: E402
from argstrength.parsing import parse_score  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.statistics import (  # noqa: E402
    bootstrap_mean_ci,
    holm_adjust,
    paired_permutation_test,
)


METRICS = [
    "spearman",
    "pairwise_accuracy",
    "strict_pairwise_accuracy",
    "tie_rate",
    "top1_accuracy",
    "exact_order_accuracy",
    "score_range",
]
INFERENTIAL_METRICS = [
    "spearman",
    "pairwise_accuracy",
    "strict_pairwise_accuracy",
    "tie_rate",
]
PROMPT_PAIRS = [
    ("role", "direct", "role_effect_without_rubric"),
    ("rubric", "direct", "rubric_effect_without_role"),
    ("role_rubric", "role", "rubric_effect_with_role"),
    ("role_rubric", "rubric", "role_effect_with_rubric"),
    ("role_rubric", "direct", "combined_vs_direct"),
]


def thinking_mode(row: dict[str, Any]) -> bool:
    return bool(row.get("thinking", row.get("metadata", {}).get("thinking", False)))


def format_ok(row: dict[str, Any]) -> bool:
    if "format_ok" in row:
        return bool(row["format_ok"])
    return parse_score(row.get("raw_output", ""), strict=True).ok


def score_parse_ok(row: dict[str, Any]) -> bool:
    if "score_parse_ok" in row:
        return bool(row["score_parse_ok"])
    # Backward compatibility for raw files generated before the field rename.
    if "parse_ok" in row:
        return bool(row["parse_ok"])
    return parse_score(row.get("raw_output", ""), strict=False).ok


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: corrupt JSONL") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"[warning] no rows for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compliance_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["study"], row["model_key"], row["prompt_condition"],
            row["decoding"], thinking_mode(row),
        )].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        exact = sum(format_ok(value) for value in values)
        extractable = sum(score_parse_ok(value) for value in values)
        token_counts = [value["generated_tokens"] for value in values if "generated_tokens" in value]
        truncation = [bool(value["hit_max_new_tokens"]) for value in values if "hit_max_new_tokens" in value]
        output.append({
            "study": key[0], "model": key[1], "prompt": key[2],
            "decoding": key[3], "thinking": key[4],
            "n_generations": len(values),
            "n_exact_format": exact,
            "format_compliance_rate": exact / len(values),
            "n_extractable_scores": extractable,
            "score_extraction_rate": extractable / len(values),
            "n_hit_max_new_tokens": sum(truncation) if truncation else None,
            "truncation_rate": statistics.fmean(truncation) if truncation else None,
            "mean_generated_tokens": statistics.fmean(token_counts) if token_counts else None,
            "max_generated_tokens": max(token_counts) if token_counts else None,
            "mean_generation_seconds": statistics.fmean(
                value["generation_seconds"] for value in values
                if "generation_seconds" in value
            ) if any("generation_seconds" in value for value in values) else None,
        })
    return output


def seed_claim_table(
    rows: list[dict[str, Any]], *, strict_margin: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("reference_rank") is not None:
            grouped[(
                row["study"], row["model_key"], row["prompt_condition"],
                row["decoding"], thinking_mode(row), row["seed"],
                row["claim_id"], row["stance"],
            )].append(row)

    output = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: row["reference_rank"])
        argument_ids = [row["argument_id"] for row in values]
        ranks = [row["reference_rank"] for row in values]
        if (
            len(set(argument_ids)) != len(argument_ids)
            or len(set(ranks)) != len(ranks)
            or len(values) < 2
        ):
            continue
        if not all(score_parse_ok(row) for row in values):
            continue
        metrics = claim_metrics(
            [row["reference_rank"] for row in values],
            [row["score"] for row in values],
            strict_margin=strict_margin,
        )
        output.append({
            "study": key[0], "model": key[1], "prompt": key[2],
            "decoding": key[3], "thinking": key[4], "seed": key[5],
            "claim_id": key[6], "stance": key[7],
            "n_arguments": len(values), **metrics,
        })
    return output


def seed_average_claim_table(
    rows: list[dict[str, Any]],
    *,
    strict_margin: float,
    expected_study_seeds: dict[str, set[int]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    inferred_seeds: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        if row.get("reference_rank") is not None:
            setting = (
                row["study"], row["model_key"], row["prompt_condition"],
                row["decoding"], thinking_mode(row),
            )
            inferred_seeds[setting].add(row["seed"])
            grouped[(
                row["study"], row["model_key"], row["prompt_condition"],
                row["decoding"], thinking_mode(row), row["claim_id"], row["stance"],
            )].append(row)

    output = []
    for key, values in sorted(grouped.items()):
        configured = (
            expected_study_seeds.get(key[0])
            if expected_study_seeds is not None
            else None
        )
        required_seeds = configured or inferred_seeds[key[:5]]
        by_argument: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            by_argument[row["argument_id"]].append(row)
        reference_ranks = {
            next(iter({row["reference_rank"] for row in argument_rows}))
            for argument_rows in by_argument.values()
            if len({row["reference_rank"] for row in argument_rows}) == 1
        }
        complete = all(
            len(argument_rows) == len(required_seeds)
            and {row["seed"] for row in argument_rows} == required_seeds
            and all(score_parse_ok(row) for row in argument_rows)
            and len({row["reference_rank"] for row in argument_rows}) == 1
            for argument_rows in by_argument.values()
        )
        if not complete or len(by_argument) < 2 or len(reference_ranks) != len(by_argument):
            continue
        averaged = []
        for argument_rows in by_argument.values():
            ranks = {row["reference_rank"] for row in argument_rows}
            if len(ranks) != 1:
                raise ValueError(f"Inconsistent reference ranks for {key}")
            averaged.append((next(iter(ranks)), statistics.fmean(row["score"] for row in argument_rows)))
        averaged.sort()
        metrics = claim_metrics(
            [rank for rank, _ in averaged],
            [score for _, score in averaged],
            strict_margin=strict_margin,
        )
        output.append({
            "study": key[0], "model": key[1], "prompt": key[2],
            "decoding": key[3], "thinking": key[4], "claim_id": key[5],
            "stance": key[6], "n_seeds": len(required_seeds),
            "n_arguments": len(averaged), **metrics,
        })
    return output


def _aggregate_claim_rows(
    rows: list[dict[str, Any]], *, by_stance: bool, bootstrap_resamples: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["study"], row["model"], row["prompt"], row["decoding"], row["thinking"])
        if by_stance:
            key += (row["stance"],)
        grouped[key].append(row)

    output = []
    for key, values in sorted(grouped.items()):
        result = {
            "study": key[0], "model": key[1], "prompt": key[2],
            "decoding": key[3], "thinking": key[4],
            "stance": key[5] if by_stance else "all",
            "n_claims": len({row["claim_id"] for row in values}),
        }
        for metric in METRICS:
            available = [row[metric] for row in values if row.get(metric) is not None]
            if not available:
                result[f"mean_{metric}"] = None
                result[f"{metric}_ci_low"] = None
                result[f"{metric}_ci_high"] = None
                continue
            mean, low, high = bootstrap_mean_ci(
                available, n_resamples=bootstrap_resamples, seed=42
            )
            result[f"mean_{metric}"] = mean
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        output.append(result)
    return output


def _relation(left: float, right: float) -> int:
    return 1 if left > right else -1 if left < right else 0


def repeat_stability_table(
    rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("reference_rank") is not None:
            raw_groups[(
                row["study"], row["model_key"], row["prompt_condition"],
                row["decoding"], thinking_mode(row), row["claim_id"], row["stance"],
            )].append(row)
    seed_metric_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        seed_metric_groups[(
            row["study"], row["model"], row["prompt"], row["decoding"],
            row["thinking"], row["claim_id"], row["stance"],
        )].append(row)

    output = []
    for key, values in sorted(raw_groups.items()):
        by_seed: dict[int, dict[str, float]] = defaultdict(dict)
        by_argument: dict[str, list[float]] = defaultdict(list)
        for row in values:
            if score_parse_ok(row):
                by_seed[row["seed"]][row["argument_id"]] = row["score"]
                by_argument[row["argument_id"]].append(row["score"])
        if len(by_seed) < 2:
            continue
        argument_sets = [set(scores) for scores in by_seed.values()]
        if any(argument_set != argument_sets[0] for argument_set in argument_sets[1:]):
            continue
        argument_ids = sorted(argument_sets[0])
        if len(argument_ids) < 2:
            continue
        agreement_values = []
        flip_flags = []
        for left_id, right_id in combinations(argument_ids, 2):
            relations = [
                _relation(scores[left_id], scores[right_id]) for scores in by_seed.values()
            ]
            flip_flags.append(float(len(set(relations)) > 1))
            agreement_values.extend(
                float(a == b) for a, b in combinations(relations, 2)
            )
        argument_stds = [
            statistics.stdev(scores) for scores in by_argument.values() if len(scores) > 1
        ]
        seed_metrics = seed_metric_groups.get(key, [])
        spearmans = [row["spearman"] for row in seed_metrics if row["spearman"] is not None]
        output.append({
            "study": key[0], "model": key[1], "prompt": key[2],
            "decoding": key[3], "thinking": key[4], "claim_id": key[5],
            "stance": key[6], "n_seeds": len(by_seed),
            "mean_argument_score_std": statistics.fmean(argument_stds) if argument_stds else 0.0,
            "pairwise_seed_consistency": statistics.fmean(agreement_values),
            "pairwise_flip_rate": statistics.fmean(flip_flags),
            "mean_seed_spearman": statistics.fmean(spearmans) if spearmans else None,
            "std_seed_spearman": statistics.stdev(spearmans) if len(spearmans) > 1 else 0.0,
        })
    return output


def repeat_stability_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["study"], row["model"], row["prompt"], row["decoding"], row["thinking"]
        )].append(row)
    metric_names = [
        "mean_argument_score_std", "pairwise_seed_consistency", "pairwise_flip_rate",
        "mean_seed_spearman", "std_seed_spearman",
    ]
    output = []
    for key, values in sorted(grouped.items()):
        result = {
            "study": key[0], "model": key[1], "prompt": key[2],
            "decoding": key[3], "thinking": key[4], "n_claims": len(values),
        }
        for metric in metric_names:
            available = [row[metric] for row in values if row.get(metric) is not None]
            result[f"mean_{metric}"] = statistics.fmean(available) if available else None
        output.append(result)
    return output


def _comparison_rows_for_scope(
    rows: list[dict[str, Any]],
    *,
    stance: str,
    bootstrap_resamples: int,
    permutation_resamples: int,
) -> list[dict[str, Any]]:
    scoped = rows if stance == "all" else [row for row in rows if row["stance"] == stance]
    by_setting: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in scoped:
        by_setting[(row["study"], row["model"], row["decoding"], row["thinking"])].append(row)
    output = []
    for setting, values in sorted(by_setting.items()):
        indices: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in values:
            indices[row["prompt"]][row["claim_id"]] = row
        for left_prompt, right_prompt, contrast in PROMPT_PAIRS:
            common = sorted(set(indices[left_prompt]) & set(indices[right_prompt]))
            if not common:
                continue
            for metric in INFERENTIAL_METRICS:
                left = [indices[left_prompt][claim][metric] for claim in common]
                right = [indices[right_prompt][claim][metric] for claim in common]
                paired = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
                if not paired:
                    continue
                left_values = [a for a, _ in paired]
                right_values = [b for _, b in paired]
                differences = [a - b for a, b in paired]
                mean, low, high = bootstrap_mean_ci(
                    differences, n_resamples=bootstrap_resamples, seed=42
                )
                test = paired_permutation_test(
                    left_values, right_values,
                    n_resamples=permutation_resamples, seed=42,
                )
                output.append({
                    "study": setting[0], "model": setting[1],
                    "decoding": setting[2], "thinking": setting[3],
                    "stance": stance, "contrast": contrast,
                    "left_prompt": left_prompt, "right_prompt": right_prompt,
                    "metric": metric, "n_claims": len(paired),
                    "left_mean": statistics.fmean(left_values),
                    "right_mean": statistics.fmean(right_values),
                    "mean_difference_left_minus_right": mean,
                    "difference_ci_low": low, "difference_ci_high": high,
                    "paired_standardised_effect": test["paired_standardised_effect"],
                    "p_value": test["p_value"],
                })

        required = {"direct", "role", "rubric", "role_rubric"}
        if required.issubset(indices):
            common = sorted(set.intersection(*(set(indices[prompt]) for prompt in required)))
            for metric in INFERENTIAL_METRICS:
                interactions = []
                for claim in common:
                    values_for_claim = [indices[prompt][claim][metric] for prompt in required]
                    if any(value is None for value in values_for_claim):
                        continue
                    direct = indices["direct"][claim][metric]
                    role = indices["role"][claim][metric]
                    rubric = indices["rubric"][claim][metric]
                    combined = indices["role_rubric"][claim][metric]
                    interactions.append(combined - rubric - role + direct)
                if interactions:
                    mean, low, high = bootstrap_mean_ci(
                        interactions, n_resamples=bootstrap_resamples, seed=42
                    )
                    test = paired_permutation_test(
                        interactions, [0.0] * len(interactions),
                        n_resamples=permutation_resamples, seed=42,
                    )
                    output.append({
                        "study": setting[0], "model": setting[1],
                        "decoding": setting[2], "thinking": setting[3],
                        "stance": stance, "contrast": "role_x_rubric_interaction",
                        "left_prompt": "factorial_interaction", "right_prompt": "zero",
                        "metric": metric, "n_claims": len(interactions),
                        "left_mean": mean, "right_mean": 0.0,
                        "mean_difference_left_minus_right": mean,
                        "difference_ci_low": low, "difference_ci_high": high,
                        "paired_standardised_effect": test["paired_standardised_effect"],
                        "p_value": test["p_value"],
                    })
    return output


def _factorial_claim_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return claim-level Role, Rubric, and Role×Rubric factorial effects."""

    output = []
    required = {"direct", "role", "rubric", "role_rubric"}
    for stance in ("all", "support", "attack"):
        scoped = rows if stance == "all" else [row for row in rows if row["stance"] == stance]
        by_setting: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in scoped:
            by_setting[(
                row["study"], row["model"], row["decoding"], row["thinking"]
            )].append(row)

        for setting, values in sorted(by_setting.items()):
            indices: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
            for row in values:
                indices[row["prompt"]][row["claim_id"]] = row
            if not required.issubset(indices):
                continue
            common = sorted(set.intersection(*(set(indices[prompt]) for prompt in required)))
            for claim_id in common:
                for metric in INFERENTIAL_METRICS:
                    prompt_values = {
                        prompt: indices[prompt][claim_id][metric] for prompt in required
                    }
                    if any(value is None for value in prompt_values.values()):
                        continue
                    direct = prompt_values["direct"]
                    role = prompt_values["role"]
                    rubric = prompt_values["rubric"]
                    combined = prompt_values["role_rubric"]
                    effects = {
                        "role_main_effect": 0.5 * (
                            (role - direct) + (combined - rubric)
                        ),
                        "rubric_main_effect": 0.5 * (
                            (rubric - direct) + (combined - role)
                        ),
                        "role_x_rubric_interaction": combined - role - rubric + direct,
                    }
                    for effect, value in effects.items():
                        output.append({
                            "study": setting[0], "model": setting[1],
                            "decoding": setting[2], "thinking": setting[3],
                            "stance": stance, "claim_id": claim_id,
                            "effect": effect, "metric": metric, "value": value,
                        })
    return output


def factorial_effects(
    rows: list[dict[str, Any]], *, bootstrap_resamples: int, permutation_resamples: int
) -> list[dict[str, Any]]:
    """Estimate factorial prompt effects within each model at claim level."""

    claim_effects = _factorial_claim_effect_rows(rows)
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in claim_effects:
        grouped[(
            row["study"], row["model"], row["decoding"], row["thinking"],
            row["stance"], row["effect"], row["metric"],
        )].append(row["value"])

    output = []
    for key, values in sorted(grouped.items()):
        mean, low, high = bootstrap_mean_ci(
            values, n_resamples=bootstrap_resamples, seed=42
        )
        test = paired_permutation_test(
            values, [0.0] * len(values),
            n_resamples=permutation_resamples, seed=42,
        )
        output.append({
            "study": key[0], "model": key[1], "decoding": key[2],
            "thinking": key[3], "stance": key[4], "effect": key[5],
            "metric": key[6], "n_claims": len(values),
            "mean_effect": mean, "effect_ci_low": low, "effect_ci_high": high,
            "paired_standardised_effect": test["paired_standardised_effect"],
            "p_value": test["p_value"],
        })

    families: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        families[(
            row["study"], row["model"], row["decoding"],
            row["thinking"], row["stance"],
        )].append(index)
    for indices in families.values():
        adjusted = holm_adjust(output[index]["p_value"] for index in indices)
        for index, value in zip(indices, adjusted):
            output[index]["holm_p_value"] = value
            output[index]["significant_0_05_holm"] = value < 0.05
    return output


def model_prompt_interactions(
    rows: list[dict[str, Any]], *, bootstrap_resamples: int, permutation_resamples: int
) -> list[dict[str, Any]]:
    """Compare claim-level factorial prompt effects between model pairs."""

    claim_effects = _factorial_claim_effect_rows(rows)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in claim_effects:
        grouped[(
            row["study"], row["decoding"], row["thinking"], row["stance"],
            row["effect"], row["metric"],
        )].append(row)

    output = []
    for key, values in sorted(grouped.items()):
        by_model: dict[str, dict[int, float]] = defaultdict(dict)
        for row in values:
            by_model[row["model"]][row["claim_id"]] = row["value"]
        for left_model, right_model in combinations(sorted(by_model), 2):
            common = sorted(set(by_model[left_model]) & set(by_model[right_model]))
            if not common:
                continue
            left = [by_model[left_model][claim] for claim in common]
            right = [by_model[right_model][claim] for claim in common]
            differences = [a - b for a, b in zip(left, right)]
            mean, low, high = bootstrap_mean_ci(
                differences, n_resamples=bootstrap_resamples, seed=42
            )
            test = paired_permutation_test(
                left, right, n_resamples=permutation_resamples, seed=42
            )
            output.append({
                "study": key[0], "decoding": key[1], "thinking": key[2],
                "stance": key[3], "effect": key[4], "metric": key[5],
                "left_model": left_model, "right_model": right_model,
                "n_claims": len(common),
                "left_model_mean_effect": statistics.fmean(left),
                "right_model_mean_effect": statistics.fmean(right),
                "difference_in_effects_left_minus_right": mean,
                "difference_ci_low": low, "difference_ci_high": high,
                "paired_standardised_effect": test["paired_standardised_effect"],
                "p_value": test["p_value"],
            })

    families: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        families[(
            row["study"], row["decoding"], row["thinking"], row["stance"],
        )].append(index)
    for indices in families.values():
        adjusted = holm_adjust(output[index]["p_value"] for index in indices)
        for index, value in zip(indices, adjusted):
            output[index]["holm_p_value"] = value
            output[index]["significant_0_05_holm"] = value < 0.05
    return output


def prompt_comparisons(
    rows: list[dict[str, Any]], *, bootstrap_resamples: int, permutation_resamples: int
) -> list[dict[str, Any]]:
    output = []
    for stance in ("all", "support", "attack"):
        output.extend(_comparison_rows_for_scope(
            rows, stance=stance, bootstrap_resamples=bootstrap_resamples,
            permutation_resamples=permutation_resamples,
        ))
    families: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        families[(
            row["study"], row["model"], row["decoding"],
            row["thinking"], row["stance"],
        )].append(index)
    for indices in families.values():
        adjusted = holm_adjust(output[index]["p_value"] for index in indices)
        for index, value in zip(indices, adjusted):
            output[index]["holm_p_value"] = value
            output[index]["significant_0_05_holm"] = value < 0.05
    return output


def representation_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("metadata", {}).get("group_id")
        if group_id is not None:
            grouped[(
                row["study"], row["model_key"], row["prompt_condition"],
                row["decoding"], thinking_mode(row), row["seed"],
                group_id, row["stance"],
            )].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        if not all(score_parse_ok(row) for row in values):
            continue
        scores = [row["score"] for row in values]
        summary = [row["score"] for row in values if row["metadata"]["representation"] == "summary"]
        expanded = [row["score"] for row in values if row["metadata"]["representation"].startswith("expanded_")]
        if len(summary) != 1 or not expanded:
            continue
        output.append({
            "study": key[0], "model": key[1], "prompt": key[2],
            "decoding": key[3], "thinking": key[4], "seed": key[5],
            "group_id": key[6], "stance": key[7], "n_variants": len(scores),
            "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "score_range": max(scores) - min(scores),
            "summary_minus_expanded_mean": summary[0] - statistics.fmean(expanded),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path, nargs="+", help="One or more raw JSONL files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/processed")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=ROOT / "configs/experiments.yaml",
        help="Study config used to enforce the declared seed set.",
    )
    parser.add_argument("--strict-margin", type=float, default=10.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--permutation-resamples", type=int, default=10_000)
    args = parser.parse_args()
    rows = read_jsonl(args.raw)
    studies = load_yaml(args.experiments).get("studies", {})
    expected_study_seeds = {
        name: set(config["seeds"]) for name, config in studies.items()
    }

    compliance = compliance_table(rows)
    seed_claims = seed_claim_table(rows, strict_margin=args.strict_margin)
    averaged_claims = seed_average_claim_table(
        rows,
        strict_margin=args.strict_margin,
        expected_study_seeds=expected_study_seeds,
    )
    stability = repeat_stability_table(rows, seed_claims)

    write_csv(args.output_dir / "compliance.csv", compliance)
    write_csv(args.output_dir / "claim_metrics_per_seed.csv", seed_claims)
    write_csv(args.output_dir / "claim_metrics_seed_average.csv", averaged_claims)
    write_csv(args.output_dir / "aggregate_metrics.csv", _aggregate_claim_rows(
        averaged_claims, by_stance=False, bootstrap_resamples=args.bootstrap_resamples
    ))
    write_csv(args.output_dir / "aggregate_by_stance.csv", _aggregate_claim_rows(
        averaged_claims, by_stance=True, bootstrap_resamples=args.bootstrap_resamples
    ))
    write_csv(args.output_dir / "repeat_stability.csv", stability)
    write_csv(args.output_dir / "repeat_stability_aggregate.csv", repeat_stability_aggregate(stability))
    write_csv(args.output_dir / "prompt_comparisons.csv", prompt_comparisons(
        averaged_claims,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
    ))
    write_csv(args.output_dir / "factorial_effects.csv", factorial_effects(
        averaged_claims,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
    ))
    write_csv(args.output_dir / "prompt_x_model_interactions.csv", model_prompt_interactions(
        averaged_claims,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
    ))
    write_csv(args.output_dir / "representation_sensitivity.csv", representation_table(rows))


if __name__ == "__main__":
    main()
