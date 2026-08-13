#!/usr/bin/env python3
"""Analyse E9 with topic-clustered inference and validation-fitted UQ transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import read_jsonl  # noqa: E402
from argstrength.external_analysis import (  # noqa: E402
    PANELS,
    aggregate_topic_metrics,
    external_prompt_contrasts,
    external_uq_discrimination,
    summarise_external_prompts,
    validation_fitted_uq_transfer,
)
from argstrength.paired_controls import build_claim_metrics  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.tabular import write_csv, write_json  # noqa: E402
from argstrength.uq import build_pair_uq_rows, seed_partitions  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path, nargs="?", default=ROOT / "results/uq_apo/raw/e9_external.jsonl")
    parser.add_argument("--validation-raw", type=Path, default=ROOT / "results/uq_apo/raw/validation.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e9_external.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/uq_apo/processed/e9_external")
    args = parser.parse_args()
    config = load_yaml(args.config)
    analysis = config["analysis"]
    winner_id = config["prompts"]["winner"]
    baseline_ids = list(config["prompts"]["manual"].values())
    rows = read_jsonl(args.raw)
    validation_rows = read_jsonl(args.validation_raw)

    expected_rows = 52 * 4 * 5 * 10
    if len(rows) != expected_rows:
        raise ValueError(f"formal E9 must contain {expected_rows} rows, found {len(rows)}")
    claim_rows = build_claim_metrics(rows, winner_id=winner_id)
    topic_rows = [
        row
        for panel in PANELS
        for row in aggregate_topic_metrics(claim_rows, panel=panel)
    ]
    prompt_summary = summarise_external_prompts(
        topic_rows,
        bootstrap_resamples=int(analysis["bootstrap_resamples"]),
        seed=int(analysis["random_seed"]),
    )
    contrasts = external_prompt_contrasts(
        topic_rows,
        winner_id=winner_id,
        baseline_ids=baseline_ids,
        bootstrap_resamples=int(analysis["bootstrap_resamples"]),
        permutation_resamples=int(analysis["permutation_resamples"]),
        seed=int(analysis["random_seed"]),
    )

    seeds = config["stages"]["formal"]["seeds"]
    uq_config = analysis["uq_transfer"]
    partitions = seed_partitions(
        seeds,
        estimate_size=int(uq_config["estimate_runs"]),
        mode=uq_config["partition_mode"],
        random_seed=int(analysis["random_seed"]),
    )
    external_pair_rows = build_pair_uq_rows(rows, partitions)
    validation_pair_rows = build_pair_uq_rows(validation_rows, partitions)
    discrimination_rows, discrimination_summary = external_uq_discrimination(
        external_pair_rows,
        continuous_signal=uq_config["continuous_signal"],
        continuous_target=uq_config["continuous_target"],
        binary_signal=uq_config["binary_signal"],
        binary_target=uq_config["binary_target"],
    )
    transfer_rows, transfer_summary = validation_fitted_uq_transfer(
        validation_pair_rows,
        external_pair_rows,
        winner_id=winner_id,
        continuous_signal=uq_config["continuous_signal"],
        continuous_target=uq_config["continuous_target"],
        binary_signal=uq_config["binary_signal"],
        binary_target=uq_config["binary_target"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "claim_metrics.csv", claim_rows)
    write_csv(args.output_dir / "topic_cluster_metrics.csv", topic_rows)
    write_csv(args.output_dir / "prompt_summary.csv", prompt_summary)
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    write_csv(
        args.output_dir / "pair_uq_fixed_partition.csv",
        [row for row in external_pair_rows if int(row["split_id"]) == 0],
    )
    write_csv(
        args.output_dir / "uq_discrimination_by_partition.csv", discrimination_rows
    )
    write_csv(
        args.output_dir / "uq_discrimination_summary.csv", discrimination_summary
    )
    write_csv(
        args.output_dir / "uq_validation_transfer_by_partition.csv", transfer_rows
    )
    write_csv(
        args.output_dir / "uq_validation_transfer_summary.csv", transfer_summary
    )
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "raw": {"path": str(args.raw), "sha256": _sha256(args.raw)},
            "validation_raw": {
                "path": str(args.validation_raw),
                "sha256": _sha256(args.validation_raw),
            },
            "winner_id": winner_id,
            "baseline_ids": baseline_ids,
            "independent_inference_unit": "source_topic",
            "n_external_topics": 13,
            "n_external_records": 52,
            "primary_panel": analysis["primary_panel"],
            "near_tie_interpretation": (
                "secondary uncertainty challenge; WA gaps may be too small for a strong gold-order claim"
            ),
            "uq_partitions": len(partitions),
            "uq_calibration": "fit on internal validation and applied externally without refit",
            "warning": (
                "The 252 seed partitions are repeated views, not independent datasets. "
                "External results cannot change the frozen winner."
            ),
        },
    )
    print(f"E9 prompt summary: {args.output_dir / 'prompt_summary.csv'}")
    print(f"E9 paired contrasts: {args.output_dir / 'paired_contrasts.csv'}")
    print(
        "E9 validation-fitted UQ transfer: "
        f"{args.output_dir / 'uq_validation_transfer_summary.csv'}"
    )


if __name__ == "__main__":
    main()
