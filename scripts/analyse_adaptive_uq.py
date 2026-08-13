#!/usr/bin/env python3
"""Run post-freeze Bayesian/conformal adaptive-sampling analyses."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.adaptive_uq import (  # noqa: E402
    calibrate_claim_grouped_thresholds,
    complete_adaptive_cells,
    evaluate_adaptive_policy,
    fixed_budget_summaries,
    summarise_adaptive_policy,
)
from argstrength.apo_experiment import read_jsonl  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.tabular import write_csv, write_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/adaptive_uq.yaml"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    output_dir = args.output_dir or _resolve(config["output_dir"])
    prompt_id = str(config["winner_prompt_id"])
    seeds = [int(seed) for seed in config["seed_order"]]
    checkpoints = [int(value) for value in config["checkpoints"]]
    fixed_budgets = [int(value) for value in config["fixed_budgets"]]
    prior = float(config["bayesian"]["prior_concentration"])
    tie_tolerance = float(config["bayesian"]["tie_tolerance"])
    levels = [float(value) for value in config["conformal"]["miscoverage_levels"]]

    calibration_path = _resolve(config["calibration"]["raw"])
    calibration_cells = complete_adaptive_cells(
        read_jsonl(calibration_path), prompt_id=prompt_id, seed_order=seeds
    )
    thresholds, calibration_scores = calibrate_claim_grouped_thresholds(
        calibration_cells,
        checkpoints=checkpoints,
        miscoverage_levels=levels,
        prior_concentration=prior,
        tie_tolerance=tie_tolerance,
    )

    all_pair_rows = []
    all_claim_rows = []
    fixed_rows = []
    input_files = {
        "calibration": {
            "path": str(calibration_path),
            "sha256": _sha256(calibration_path),
        }
    }
    for dataset, dataset_config in config["evaluation"].items():
        raw_path = _resolve(dataset_config["raw"])
        cells = complete_adaptive_cells(
            read_jsonl(raw_path), prompt_id=prompt_id, seed_order=seeds
        )
        if not cells:
            raise ValueError(f"No complete winner cells found for {dataset}")
        input_files[dataset] = {"path": str(raw_path), "sha256": _sha256(raw_path)}
        fixed_rows.extend(
            fixed_budget_summaries(
                cells,
                dataset=dataset,
                budgets=fixed_budgets,
                tie_tolerance=tie_tolerance,
            )
        )
        for threshold_row in thresholds:
            pair_rows, claim_rows = evaluate_adaptive_policy(
                cells,
                dataset=dataset,
                threshold=float(threshold_row["threshold"]),
                miscoverage=float(threshold_row["miscoverage"]),
                checkpoints=checkpoints,
                prior_concentration=prior,
                tie_tolerance=tie_tolerance,
            )
            all_pair_rows.extend(pair_rows)
            all_claim_rows.extend(claim_rows)

    summaries = summarise_adaptive_policy(all_pair_rows, all_claim_rows)
    write_csv(output_dir / "conformal_thresholds.csv", thresholds)
    write_csv(output_dir / "calibration_claim_scores.csv", calibration_scores)
    write_csv(output_dir / "adaptive_pair_decisions.csv", all_pair_rows)
    write_csv(output_dir / "adaptive_claim_decisions.csv", all_claim_rows)
    write_csv(output_dir / "adaptive_summary.csv", summaries)
    write_csv(output_dir / "fixed_budget_summary.csv", fixed_rows)
    write_json(
        output_dir / "analysis_manifest.json",
        {
            "status": config["status"],
            "winner_prompt_id": prompt_id,
            "inputs": input_files,
            "seed_order": seeds,
            "checkpoints": checkpoints,
            "fixed_budgets": fixed_budgets,
            "prior": config["bayesian"],
            "conformal": config["conformal"],
            "n_calibration_claims": len(calibration_cells),
            "warnings": [
                "This analysis was designed after E5--E9 results were inspected and is exploratory.",
                "Conformal coverage relies on claim exchangeability; E9 domain shift can violate it.",
                "Claim-max calibration targets simultaneous label-set coverage, not conditional selective risk.",
                "The frozen APO winner cannot be changed by this analysis.",
            ],
        },
    )
    print(f"Adaptive summary: {output_dir / 'adaptive_summary.csv'}")
    print(f"Fixed-budget summary: {output_dir / 'fixed_budget_summary.csv'}")


if __name__ == "__main__":
    main()

