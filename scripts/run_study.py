#!/usr/bin/env python3
"""Run any study declared in configs/experiments.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.runner import load_yaml, run_experiment  # noqa: E402


def main() -> None:
    studies = load_yaml(ROOT / "configs/experiments.yaml").get("studies", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, choices=sorted(studies))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mistral_7b", "qwen3_8b", "gpt_oss_20b"],
        help="Run a subset; omitted means all models configured for the study.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Raw JSONL checkpoint. Cluster jobs must use separate files.",
    )
    args = parser.parse_args()
    output = run_experiment(
        models_path=ROOT / "configs/models.yaml",
        experiments_path=ROOT / "configs/experiments.yaml",
        study_name=args.study,
        output_path=args.output,
        model_keys=args.models,
    )
    print(f"Raw generations: {output}")


if __name__ == "__main__":
    main()
