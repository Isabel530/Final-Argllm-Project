"""Topic-clustered performance and UQ-transfer analysis for external E9."""

from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Any, Sequence

from .paired_controls import METRICS
from .statistics import bootstrap_mean_ci, holm_adjust, paired_permutation_test
from .uq import (
    _fit_isotonic,
    aurc,
    auroc,
    average_precision,
    expected_calibration_error,
    spearman,
)


PANELS = ("all", "broad", "near_tie")


def aggregate_topic_metrics(
    claim_rows: Sequence[dict[str, Any]], *, panel: str
) -> list[dict[str, Any]]:
    """Aggregate repeated stance/panel records to independent source topics."""

    if panel not in PANELS:
        raise ValueError(f"unknown panel {panel!r}")
    filtered = [
        row for row in claim_rows if panel == "all" or row.get("panel") == panel
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        topic = row.get("source_topic")
        if not isinstance(topic, str) or not topic:
            raise ValueError("external claim metrics require source_topic")
        grouped[(row["prompt_id"], topic)].append(row)
    output = []
    for (prompt_id, topic), rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "model": rows[0]["model"],
            "panel": panel,
            "prompt_id": prompt_id,
            "prompt_label": rows[0]["prompt_label"],
            "source_topic": topic,
            "n_claim_records": len(rows),
            "n_seeds": rows[0]["n_seeds"],
            "invalid_rate": statistics.fmean(row["invalid_rate"] for row in rows),
        }
        for metric in METRICS:
            result[metric] = statistics.fmean(float(row[metric]) for row in rows)
        output.append(result)
    return output


def summarise_external_prompts(
    topic_rows: Sequence[dict[str, Any]], *, bootstrap_resamples: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in topic_rows:
        grouped[(row["panel"], row["prompt_id"])].append(row)
    output = []
    for (panel, prompt_id), rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "model": rows[0]["model"],
            "panel": panel,
            "prompt_id": prompt_id,
            "prompt_label": rows[0]["prompt_label"],
            "n_topics": len(rows),
            "n_claim_records": sum(row["n_claim_records"] for row in rows),
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


def external_prompt_contrasts(
    topic_rows: Sequence[dict[str, Any]],
    *,
    winner_id: str,
    baseline_ids: Sequence[str],
    bootstrap_resamples: int,
    permutation_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in topic_rows:
        index[(row["panel"], row["prompt_id"])][row["source_topic"]] = row
    output = []
    for panel in PANELS:
        winner = index.get((panel, winner_id), {})
        if not winner:
            raise ValueError(f"winner absent from panel {panel}")
        for baseline_id in baseline_ids:
            baseline = index.get((panel, baseline_id), {})
            common = sorted(set(winner) & set(baseline))
            if not common:
                raise ValueError(f"no paired topics for {panel}/{baseline_id}")
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
                        "model": winner[common[0]]["model"],
                        "panel": panel,
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
                        "higher_is_better": metric == "pairwise_accuracy",
                    }
                )
    families: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index_number, row in enumerate(output):
        families[(row["panel"], row["metric"])].append(index_number)
    for indices in families.values():
        adjusted = holm_adjust([output[index]["p_value"] for index in indices])
        for index_number, p_holm in zip(indices, adjusted):
            output[index_number]["p_value_holm"] = p_holm
            output[index_number]["significant_holm_0_05"] = p_holm < 0.05
    return output


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def external_uq_discrimination(
    pair_rows: Sequence[dict[str, Any]],
    *,
    continuous_signal: str,
    continuous_target: str,
    binary_signal: str,
    binary_target: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Raw external discrimination for every prompt; no external calibrator."""

    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        for panel in ("all", row.get("panel")):
            if panel in PANELS:
                grouped[(int(row["split_id"]), row["prompt_id"], panel)].append(row)
    per_partition = []
    for (split_id, prompt_id, panel), rows in sorted(grouped.items()):
        continuous_risk = [float(row[continuous_signal]) for row in rows]
        continuous = [float(row[continuous_target]) for row in rows]
        binary_risk = [float(row[binary_signal]) for row in rows]
        binary = [int(row[binary_target]) for row in rows]
        per_partition.append(
            {
                "split_id": split_id,
                "prompt_id": prompt_id,
                "panel": panel,
                "n_pairs": len(rows),
                "n_topics": len({row["source_topic"] for row in rows}),
                "continuous_signal": continuous_signal,
                "continuous_spearman": spearman(continuous_risk, continuous),
                "continuous_aurc": aurc(continuous_risk, continuous),
                "binary_signal": binary_signal,
                "binary_auroc": auroc(binary_risk, binary),
                "binary_auprc": average_precision(binary_risk, binary),
                "positive_rate": statistics.fmean(binary),
            }
        )
    aggregate = []
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_partition:
        by_cell[(row["prompt_id"], row["panel"])].append(row)
    for (prompt_id, panel), rows in sorted(by_cell.items()):
        aggregate.append(
            {
                "prompt_id": prompt_id,
                "panel": panel,
                "n_partitions": len(rows),
                "n_pairs_per_partition": rows[0]["n_pairs"],
                "n_topics": rows[0]["n_topics"],
                "continuous_signal": continuous_signal,
                "mean_continuous_spearman": _mean_or_none(
                    [row["continuous_spearman"] for row in rows]
                ),
                "mean_continuous_aurc": statistics.fmean(
                    row["continuous_aurc"] for row in rows
                ),
                "binary_signal": binary_signal,
                "mean_binary_auroc": _mean_or_none(
                    [row["binary_auroc"] for row in rows]
                ),
                "mean_binary_auprc": _mean_or_none(
                    [row["binary_auprc"] for row in rows]
                ),
                "mean_positive_rate": statistics.fmean(
                    row["positive_rate"] for row in rows
                ),
            }
        )
    return per_partition, aggregate


def validation_fitted_uq_transfer(
    validation_pair_rows: Sequence[dict[str, Any]],
    external_pair_rows: Sequence[dict[str, Any]],
    *,
    winner_id: str,
    continuous_signal: str,
    continuous_target: str,
    binary_signal: str,
    binary_target: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit isotonic risk only on validation and apply it to external E9."""

    validation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    external: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validation_pair_rows:
        if row["prompt_id"] == winner_id:
            validation[int(row["split_id"])].append(row)
    for row in external_pair_rows:
        if row["prompt_id"] != winner_id:
            continue
        for panel in ("all", row.get("panel")):
            if panel in PANELS:
                external[(int(row["split_id"]), panel)].append(row)

    output = []
    for (split_id, panel), test_rows in sorted(external.items()):
        train_rows = validation.get(split_id, [])
        if not train_rows:
            raise ValueError(f"validation partition {split_id} is absent")
        predictor = _fit_isotonic(
            [float(row[binary_signal]) for row in train_rows],
            [int(row[binary_target]) for row in train_rows],
        )
        binary_risk = [float(row[binary_signal]) for row in test_rows]
        binary = [int(row[binary_target]) for row in test_rows]
        probabilities = [predictor(value) for value in binary_risk]
        continuous_risk = [float(row[continuous_signal]) for row in test_rows]
        continuous = [float(row[continuous_target]) for row in test_rows]
        output.append(
            {
                "split_id": split_id,
                "prompt_id": winner_id,
                "panel": panel,
                "calibration_fit": "internal_validation_only",
                "external_refit": False,
                "n_validation_pairs": len(train_rows),
                "n_external_pairs": len(test_rows),
                "n_external_topics": len({row["source_topic"] for row in test_rows}),
                "continuous_signal": continuous_signal,
                "continuous_spearman": spearman(continuous_risk, continuous),
                "continuous_aurc": aurc(continuous_risk, continuous),
                "binary_signal": binary_signal,
                "binary_auroc": auroc(binary_risk, binary),
                "binary_auprc": average_precision(binary_risk, binary),
                "binary_brier_validation_fitted": statistics.fmean(
                    (probability - outcome) ** 2
                    for probability, outcome in zip(probabilities, binary)
                ),
                "binary_ece_validation_fitted": expected_calibration_error(
                    probabilities, binary
                ),
                "positive_rate": statistics.fmean(binary),
            }
        )
    aggregate = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        grouped[row["panel"]].append(row)
    for panel, rows in sorted(grouped.items()):
        aggregate.append(
            {
                "prompt_id": winner_id,
                "panel": panel,
                "calibration_fit": "internal_validation_only",
                "external_refit": False,
                "n_partitions": len(rows),
                "n_validation_pairs_per_partition": rows[0]["n_validation_pairs"],
                "n_external_pairs_per_partition": rows[0]["n_external_pairs"],
                "n_external_topics": rows[0]["n_external_topics"],
                "continuous_signal": continuous_signal,
                "mean_continuous_spearman": _mean_or_none(
                    [row["continuous_spearman"] for row in rows]
                ),
                "mean_continuous_aurc": statistics.fmean(
                    row["continuous_aurc"] for row in rows
                ),
                "binary_signal": binary_signal,
                "mean_binary_auroc": _mean_or_none(
                    [row["binary_auroc"] for row in rows]
                ),
                "mean_binary_auprc": _mean_or_none(
                    [row["binary_auprc"] for row in rows]
                ),
                "mean_binary_brier_validation_fitted": statistics.fmean(
                    row["binary_brier_validation_fitted"] for row in rows
                ),
                "mean_binary_ece_validation_fitted": statistics.fmean(
                    row["binary_ece_validation_fitted"] for row in rows
                ),
                "mean_positive_rate": statistics.fmean(
                    row["positive_rate"] for row in rows
                ),
            }
        )
    return output, aggregate
