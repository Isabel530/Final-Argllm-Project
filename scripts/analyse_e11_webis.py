#!/usr/bin/env python3
"""Analyse frozen Webis E11 performance, UQ transfer, and E10 policy transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import read_jsonl  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.tabular import write_csv, write_json  # noqa: E402
from argstrength.uq import build_pair_uq_rows  # noqa: E402
from argstrength.webis_analysis import (  # noqa: E402
    build_fixed_partition_uq_rows,
    build_pair_metrics,
    build_topic_metrics,
    evaluate_pairwise_bayesian_adaptive,
    paired_prompt_contrasts_by_topic,
    summarise_prompts_by_topic,
    summarise_uq_discrimination,
    validation_fitted_calibration_transfer,
)
from argstrength.webis_external import (  # noqa: E402
    load_webis_generation_dataset,
    load_webis_pair_labels,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw",
        type=Path,
        nargs="?",
        default=ROOT / "results/uq_apo/raw/e11_webis_external.jsonl",
    )
    parser.add_argument(
        "--validation-raw",
        type=Path,
        default=ROOT / "results/uq_apo/raw/validation.jsonl",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/e11_webis_external.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/uq_apo/processed/e11_webis_external",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    analysis = config["analysis"]
    stage = config["stages"]["formal"]
    manifest_path = ROOT / config["preparation"]["manifest"]
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    dataset_path = ROOT / stage["dataset"]
    pair_path = ROOT / stage["pairs"]
    items = load_webis_generation_dataset(dataset_path)
    pairs = load_webis_pair_labels(pair_path)
    raw_rows = read_jsonl(args.raw)
    validation_raw = read_jsonl(args.validation_raw)
    prompt_ids = [
        config["prompts"]["manual"]["direct"],
        config["prompts"]["manual"]["rubric"],
        config["prompts"]["winner"],
    ]
    expected_rows = (
        sum(len(item.arguments) for item in items)
        * len(prompt_ids)
        * len(stage["seeds"])
    )
    if len(raw_rows) != expected_rows:
        raise ValueError(
            f"formal E11 must contain {expected_rows} rows, found {len(raw_rows)}"
        )
    if any(not row.get("score_parse_ok") for row in raw_rows):
        raise ValueError("hard-decoded E11 contains invalid score rows")

    pair_metrics = build_pair_metrics(
        raw_rows,
        pairs,
        prompt_ids=prompt_ids,
        seeds=stage["seeds"],
        winner_id=config["prompts"]["winner"],
    )
    topic_metrics = build_topic_metrics(pair_metrics, raw_rows)
    prompt_summary = summarise_prompts_by_topic(
        topic_metrics,
        bootstrap_resamples=int(analysis["bootstrap_resamples"]),
        seed=int(analysis["random_seed"]),
    )
    contrasts = paired_prompt_contrasts_by_topic(
        topic_metrics,
        winner_id=config["prompts"]["winner"],
        baseline_ids=list(config["prompts"]["manual"].values()),
        bootstrap_resamples=int(analysis["bootstrap_resamples"]),
        permutation_resamples=int(analysis["permutation_resamples"]),
        seed=int(analysis["random_seed"]),
    )

    estimate_seeds = [int(value) for value in analysis["estimate_seeds"]]
    heldout_seeds = [int(value) for value in analysis["heldout_seeds"]]
    uq_rows = build_fixed_partition_uq_rows(
        raw_rows,
        pairs,
        prompt_ids=prompt_ids,
        estimate_seeds=estimate_seeds,
        heldout_seeds=heldout_seeds,
    )
    uq_summary = summarise_uq_discrimination(uq_rows)
    validation_uq = build_pair_uq_rows(
        validation_raw, [(tuple(estimate_seeds), tuple(heldout_seeds))]
    )
    calibration_transfer = validation_fitted_calibration_transfer(
        validation_uq,
        uq_rows,
        winner_id=config["prompts"]["winner"],
    )

    adaptive_rows, adaptive_summary = evaluate_pairwise_bayesian_adaptive(
        raw_rows,
        pairs,
        prompt_id_value=config["prompts"]["winner"],
        seed_order=stage["seeds"],
        checkpoints=[int(value) for value in analysis["adaptive_checkpoints"]],
        confidence_thresholds=[
            float(value) for value in analysis["bayesian_confidence_thresholds"]
        ],
        prior_concentration=float(analysis["dirichlet_prior_concentration"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pair_metrics.csv", pair_metrics)
    write_csv(args.output_dir / "topic_cluster_metrics.csv", topic_metrics)
    write_csv(args.output_dir / "prompt_summary.csv", prompt_summary)
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    write_csv(args.output_dir / "uq_fixed_partition_pairs.csv", uq_rows)
    write_csv(args.output_dir / "uq_discrimination_summary.csv", uq_summary)
    write_csv(args.output_dir / "uq_calibration_transfer.csv", calibration_transfer)
    write_csv(args.output_dir / "bayesian_adaptive_pair_decisions.csv", adaptive_rows)
    write_csv(args.output_dir / "bayesian_adaptive_summary.csv", adaptive_summary)
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "raw": {"path": str(args.raw), "sha256": _sha256(args.raw)},
            "validation_raw": {
                "path": str(args.validation_raw),
                "sha256": _sha256(args.validation_raw),
            },
            "locked_data_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "winner_id": config["prompts"]["winner"],
            "baseline_ids": list(config["prompts"]["manual"].values()),
            "primary_scope": analysis["primary_scope"],
            "independent_inference_unit": analysis["independent_cluster"],
            "n_formal_topics": manifest["formal"]["topics"],
            "n_formal_groups": manifest["formal"]["topic_stance_groups"],
            "n_human_pairs": manifest["formal"]["eligible_dimension_pairs"],
            "n_primary_pairs": manifest["formal"]["majority_decisive_pairs"],
            "uq_partition": {
                "estimate_seeds": estimate_seeds,
                "heldout_seeds": heldout_seeds,
            },
            "calibration": "fit on internal validation; applied to E11 without refit",
            "adaptive_policy": (
                "empirical Bayesian pair-at-a-time transfer; no conformal guarantee"
            ),
            "warning": (
                "E11 cannot change or rewrite the frozen prompt. Selective adaptive "
                "accuracy must be reported with coverage and cost."
            ),
        },
    )
    print(f"E11 prompt summary: {args.output_dir / 'prompt_summary.csv'}")
    print(f"E11 paired contrasts: {args.output_dir / 'paired_contrasts.csv'}")
    print(f"E11 UQ summary: {args.output_dir / 'uq_discrimination_summary.csv'}")
    print(f"E11 adaptive summary: {args.output_dir / 'bayesian_adaptive_summary.csv'}")


if __name__ == "__main__":
    main()
