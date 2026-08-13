#!/usr/bin/env python3
"""Freeze claim splits and initialise the readable APO candidate registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo import initial_prompt_candidates  # noqa: E402
from argstrength.apo_experiment import write_candidate_registry  # noqa: E402
from argstrength.data import load_main_dataset  # noqa: E402
from argstrength.experimental_design import make_claim_splits  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/uq_apo.yaml")
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=ROOT / "results/uq_apo/candidates/iteration_0.jsonl",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing split/registry. Omit to protect frozen design files.",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    dataset_path = ROOT / config["dataset"]
    split_path = ROOT / config["locked_split"]
    if not args.force and (split_path.exists() or args.candidate_registry.exists()):
        raise FileExistsError(
            "A locked split or candidate registry already exists; use --force only "
            "if you intentionally want a new prospectively locked design."
        )

    items = load_main_dataset(dataset_path)
    sizes = config["split_sizes"]
    document = make_claim_splits(
        items,
        train_size=int(sizes["apo_train"]),
        validation_size=int(sizes["apo_validation"]),
        test_size=int(sizes["internal_test"]),
        seed=int(config["split_seed"]),
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_candidate_registry(
        args.candidate_registry, initial_prompt_candidates(), append=False
    )
    print(f"Locked split: {split_path}")
    print(f"Initial candidates: {args.candidate_registry}")
    print(json.dumps(document["stance_counts"], indent=2))


if __name__ == "__main__":
    main()
