#!/usr/bin/env python3
"""Generate auditable CSV inputs for the dissertation's synthesis figures.

The report renders the figures in TikZ so that it can be compiled without a
Python plotting environment.  This script keeps every plotted number traceable
to the machine-readable analysis outputs.  It does not rerun any statistical
test or model generation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "dissertation"
WINNER = "p_53a4acbe525c"
DIRECT = "p_83adf7318fd5"
RUBRIC = "p_0086c8edb3b1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def uq_signal_ranking() -> None:
    source = ROOT / "results/uq_apo/processed/validation_uq/pair_uq_summary.csv"
    rows = [row for row in read_csv(source) if row["prompt_id"] == WINNER]
    fields = [
        "signal",
        "mean_spearman",
        "mean_auroc",
        "mean_auprc",
        "mean_aurc",
        "mean_brier",
        "mean_ece",
    ]
    write_csv(OUT / "uq_signal_ranking.csv", [{key: row[key] for key in fields} for row in rows], fields)


def risk_coverage() -> None:
    source = ROOT / "results/uq_apo/processed/validation_uq/pair_uq_rows.csv"
    rows = [row for row in read_csv(source) if row["prompt_id"] == WINNER]
    by_split: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split[int(row["split_id"])].append(row)

    specifications = [
        ("margin_uncertainty", "future_reference_error", "future error"),
        ("sampling_variance", "future_absolute_deviation", "future deviation"),
    ]
    coverages = (0.10, 0.25, 0.50, 0.75, 1.00)
    output: list[dict[str, object]] = []
    for signal, target, target_label in specifications:
        per_coverage: dict[float, list[float]] = defaultdict(list)
        for split_rows in by_split.values():
            ordered = sorted(split_rows, key=lambda row: float(row[signal]))
            for coverage in coverages:
                retained = ordered[: max(1, round(len(ordered) * coverage))]
                per_coverage[coverage].append(
                    sum(float(row[target]) for row in retained) / len(retained)
                )
        for coverage in coverages:
            risks = per_coverage[coverage]
            output.append(
                {
                    "signal": signal,
                    "target": target_label,
                    "coverage": coverage,
                    "mean_risk": sum(risks) / len(risks),
                    "n_partitions": len(risks),
                }
            )
    write_csv(
        OUT / "uq_risk_coverage.csv",
        output,
        ["signal", "target", "coverage", "mean_risk", "n_partitions"],
    )


def apo_trajectory() -> None:
    output: list[dict[str, object]] = []
    for iteration in range(3):
        directory = ROOT / f"results/uq_apo/search/round_{iteration}"
        summary = read_csv(directory / "candidate_summary.csv")
        with (directory / "round_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        indexed = {row["prompt_id"]: row for row in summary}
        output.append(
            {
                "round": iteration,
                "evaluated": len(manifest["evaluated_prompt_ids"]),
                "survivors": len(manifest["survivor_prompt_ids"]),
                "proposed": len(manifest["proposal_prompt_ids"]),
                "next_population": len(manifest["next_prompt_ids"]),
                "best_accuracy": max(float(row["pairwise_accuracy"]) for row in summary),
                "best_flip": min(
                    float(row["seed_flip_rate"])
                    for row in summary
                    if float(row["pairwise_accuracy"])
                    == max(float(item["pairwise_accuracy"]) for item in summary)
                ),
                "direct_accuracy": float(indexed[DIRECT]["pairwise_accuracy"]),
                "direct_flip": float(indexed[DIRECT]["seed_flip_rate"]),
                "winner_accuracy": float(indexed[WINNER]["pairwise_accuracy"])
                if WINNER in indexed
                else "",
                "winner_flip": float(indexed[WINNER]["seed_flip_rate"])
                if WINNER in indexed
                else "",
            }
        )
    fields = [
        "round",
        "evaluated",
        "survivors",
        "proposed",
        "next_population",
        "best_accuracy",
        "best_flip",
        "direct_accuracy",
        "direct_flip",
        "winner_accuracy",
        "winner_flip",
    ]
    write_csv(OUT / "apo_search_trajectory.csv", output, fields)


def transfer_summary() -> None:
    sources = [
        (
            "Qwen internal",
            ROOT / "results/uq_apo/processed/qwen_paired_controls/prompt_summary.csv",
            None,
        ),
        (
            "Mistral internal",
            ROOT / "results/uq_apo/processed/mistral_paired/prompt_summary.csv",
            None,
        ),
        (
            "IBM external",
            ROOT / "results/uq_apo/processed/e9_external/prompt_summary.csv",
            ("panel", "all"),
        ),
        (
            "Webis external",
            ROOT / "results/uq_apo/processed/e11_webis_external/prompt_summary.csv",
            ("scope", "majority_decisive"),
        ),
    ]
    wanted = {DIRECT: "Direct", RUBRIC: "Rubric", WINNER: "Winner"}
    output: list[dict[str, object]] = []
    for domain, path, row_filter in sources:
        rows = read_csv(path)
        for row in rows:
            if row_filter and row[row_filter[0]] != row_filter[1]:
                continue
            if row["prompt_id"] not in wanted:
                continue
            output.append(
                {
                    "domain": domain,
                    "prompt": wanted[row["prompt_id"]],
                    "accuracy": row["pairwise_accuracy"],
                    "flip_rate": row["seed_flip_rate"],
                }
            )
    write_csv(OUT / "transfer_summary.csv", output, ["domain", "prompt", "accuracy", "flip_rate"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    uq_signal_ranking()
    risk_coverage()
    apo_trajectory()
    transfer_summary()
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "winner_prompt_id": WINNER,
        "note": (
            "Derived only from processed result tables. Risk--coverage retains the "
            "lowest-uncertainty fraction within each of 252 complementary 5+5 "
            "partitions and then averages risk; partitions are not inferential units."
        ),
    }
    with (OUT / "figure_data_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
