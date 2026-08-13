#!/usr/bin/env python3
"""Evaluate fixed Bayesian posterior-confidence early-stopping policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.adaptive_uq import complete_adaptive_cells  # noqa: E402
from argstrength.apo_experiment import read_jsonl  # noqa: E402
from argstrength.bayesian_adaptive import (  # noqa: E402
    evaluate_bayesian_confidence_policy,
    summarise_bayesian_confidence_policy,
)
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.tabular import write_csv, write_json  # noqa: E402


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bayesian_adaptive.yaml"
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    output_dir = _resolve(config["output_dir"])
    prompt_id = str(config["winner_prompt_id"])
    seed_order = [int(value) for value in config["seed_order"]]
    checkpoints = [int(value) for value in config["checkpoints"]]
    prior = float(config["prior_concentration"])
    tie_tolerance = float(config["tie_tolerance"])

    pair_rows, claim_rows = [], []
    for dataset, raw in config["evaluation"].items():
        cells = complete_adaptive_cells(
            read_jsonl(_resolve(raw)), prompt_id=prompt_id, seed_order=seed_order
        )
        for threshold in config["confidence_thresholds"]:
            pairs, claims = evaluate_bayesian_confidence_policy(
                cells,
                dataset=dataset,
                confidence_threshold=float(threshold),
                checkpoints=checkpoints,
                prior_concentration=prior,
                tie_tolerance=tie_tolerance,
            )
            pair_rows.extend(pairs)
            claim_rows.extend(claims)

    summary = summarise_bayesian_confidence_policy(pair_rows, claim_rows)
    write_csv(output_dir / "bayesian_pair_decisions.csv", pair_rows)
    write_csv(output_dir / "bayesian_claim_decisions.csv", claim_rows)
    write_csv(output_dir / "bayesian_adaptive_summary.csv", summary)
    write_json(
        output_dir / "bayesian_policy_manifest.json",
        {
            **config,
            "warnings": [
                "Fixed posterior-confidence thresholds have no conformal coverage guarantee.",
                "This policy was added after E5--E9 results were inspected and is exploratory.",
                "The frozen winner cannot be changed by this analysis.",
            ],
        },
    )
    print(f"Bayesian adaptive summary: {output_dir / 'bayesian_adaptive_summary.csv'}")


if __name__ == "__main__":
    main()

