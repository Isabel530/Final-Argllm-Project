#!/usr/bin/env python3
"""Run a post-validation evaluation only from a hash-locked registry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import (  # noqa: E402
    read_candidate_registry,
    run_candidate_evaluation,
)
from argstrength.data import load_main_dataset  # noqa: E402
from argstrength.experimental_design import load_claim_splits  # noqa: E402
from argstrength.registry_freeze import verify_frozen_registry  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402


PURPOSES = {
    "qwen_internal_test": {"model": "qwen3_8b", "stage": "internal_test"},
    "cross_model_transfer": {"model": "mistral_7b", "stage": "cross_model_transfer"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/uq_apo.yaml")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--purpose", choices=tuple(PURPOSES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = PURPOSES[args.purpose]
    config = load_yaml(args.config)
    candidates = read_candidate_registry(args.registry)
    authorised_purpose = verify_frozen_registry(
        args.registry, candidates, args.freeze_manifest
    )
    if authorised_purpose != args.purpose:
        raise ValueError(
            f"registry is authorised for {authorised_purpose}, not {args.purpose}"
        )

    # Both confirmatory evaluations use the untouched 50-claim internal split
    # and the same ten seeds. The transfer changes only the model and registry.
    stage_config = config["stages"]["internal_test"]
    dataset_path = ROOT / config["dataset"]
    items = load_main_dataset(dataset_path)
    split_document = load_claim_splits(ROOT / config["locked_split"], items)
    claim_ids = split_document["splits"][stage_config["split"]]
    if len(claim_ids) != int(stage_config["claims"]):
        raise ValueError("locked internal-test claim count differs from config")

    generation = config["generation"]
    path = run_candidate_evaluation(
        models_path=ROOT / "configs/models.yaml",
        dataset_path=dataset_path,
        model_key=protocol["model"],
        candidates=candidates,
        claim_ids=claim_ids,
        seeds=stage_config["seeds"],
        output_path=args.output,
        stage=protocol["stage"],
        decoding=generation["decoding"],
        thinking=bool(generation["thinking"]),
        do_sample=bool(generation["do_sample"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        repetition_penalty=float(generation["repetition_penalty"]),
        max_new_tokens=int(generation["max_new_tokens"]),
    )
    print(f"Authorised purpose: {args.purpose}")
    print(f"Model: {protocol['model']}")
    print(f"Raw candidate evaluations: {path}")
    print(f"Claims: {len(claim_ids)}; seeds: {len(stage_config['seeds'])}")


if __name__ == "__main__":
    main()
