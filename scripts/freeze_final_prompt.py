#!/usr/bin/env python3
"""Freeze the validation winner and prepare the cross-model registry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.registry_freeze import freeze_validation_winner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, default=ROOT / "results/uq_apo/candidates/iteration_0.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/uq_apo/frozen")
    args = parser.parse_args()
    manifest = freeze_validation_winner(
        candidate_registry=args.candidates,
        selection_path=args.selection,
        baseline_registry=args.baselines,
        winner_output=args.output_dir / "qwen_frozen_winner.jsonl",
        transfer_output=args.output_dir / "mistral_transfer_prompts.jsonl",
        manifest_output=args.output_dir / "freeze_manifest.json",
    )
    print(f"Frozen winner: {manifest['winner_prompt_id']}")
    print(f"Freeze manifest: {args.output_dir / 'freeze_manifest.json'}")
    print(f"Mistral transfer registry: {args.output_dir / 'mistral_transfer_prompts.jsonl'}")


if __name__ == "__main__":
    main()
