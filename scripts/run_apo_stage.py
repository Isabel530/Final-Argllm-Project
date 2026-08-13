#!/usr/bin/env python3
"""Run one resumable candidate-evaluation stage (screen/promote/validate/test)."""

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
from argstrength.experimental_design import (  # noqa: E402
    load_claim_splits,
    screening_batch,
)
from argstrength.runner import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/uq_apo.yaml")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "search_screening",
            "promotion",
            "train_full",
            "validation",
            "internal_test",
        ),
        required=True,
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Controls the rotating 10-claim training batch.",
    )
    parser.add_argument("--model", help="Override config.task_model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_yaml(args.config)
    stage = config["stages"][args.stage]
    dataset_path = ROOT / config["dataset"]
    items = load_main_dataset(dataset_path)
    split_document = load_claim_splits(ROOT / config["locked_split"], items)
    split_claim_ids = split_document["splits"][stage["split"]]
    requested_claims = int(stage["claims"])
    if requested_claims > len(split_claim_ids):
        raise ValueError("stage requests more claims than its locked split")
    if requested_claims == len(split_claim_ids):
        claim_ids = list(split_claim_ids)
    elif stage.get("balanced_by_stance", False):
        claim_ids = screening_batch(
            items,
            split_claim_ids,
            iteration=args.iteration if stage.get("rotate_claims_each_iteration") else 0,
            batch_size=requested_claims,
            seed=int(config["split_seed"]),
        )
    else:
        claim_ids = list(split_claim_ids[:requested_claims])

    candidates = read_candidate_registry(args.candidates)
    output = args.output or (
        ROOT
        / "results/uq_apo/raw"
        / f"{args.stage}_iteration_{args.iteration}.jsonl"
    )
    generation = config["generation"]
    path = run_candidate_evaluation(
        models_path=ROOT / "configs/models.yaml",
        dataset_path=dataset_path,
        model_key=args.model or config["task_model"],
        candidates=candidates,
        claim_ids=claim_ids,
        seeds=stage["seeds"],
        output_path=output,
        stage=args.stage,
        decoding=generation["decoding"],
        thinking=bool(generation["thinking"]),
        do_sample=bool(generation["do_sample"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        repetition_penalty=float(generation["repetition_penalty"]),
        max_new_tokens=int(generation["max_new_tokens"]),
    )
    print(f"Raw candidate evaluations: {path}")
    print(f"Claims: {len(claim_ids)}; seeds: {len(stage['seeds'])}")


if __name__ == "__main__":
    main()
