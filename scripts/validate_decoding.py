#!/usr/bin/env python3
"""Run the small hard-vs-soft decoding validation across selected models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.runner import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mistral_7b", "qwen3_8b", "gpt_oss_20b"],
        help="Default: all models in the YAML study.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = run_experiment(
        models_path=ROOT / "configs/models.yaml",
        experiments_path=ROOT / "configs/experiments.yaml",
        study_name="decoding_validation",
        output_path=args.output,
        model_keys=args.models,
    )
    print(f"Raw generations: {output}")


if __name__ == "__main__":
    main()
