#!/usr/bin/env python3
"""Run the excluded E9 pilot or the once-only frozen formal external test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import read_candidate_registry, run_candidate_evaluation  # noqa: E402
from argstrength.external_data import load_external_rank_dataset  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e9_external.yaml")
    parser.add_argument("--stage", choices=("pilot", "formal"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    stage = config["stages"][args.stage]
    manifest_path = ROOT / config["preparation"]["manifest"]
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    dataset_path = ROOT / stage["dataset"]
    registry_path = ROOT / stage["candidates"]
    locked = manifest[args.stage]
    for path, expected, label in (
        (dataset_path, locked["sha256"], "dataset"),
        (registry_path, locked["candidate_registry_sha256"], "candidate registry"),
    ):
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"{label} fingerprint mismatch: {observed} != {expected}")

    items = load_external_rank_dataset(dataset_path)
    candidates = read_candidate_registry(registry_path)
    expected_ids = list(config["prompts"]["manual"].values()) + [config["prompts"]["winner"]]
    if args.stage == "pilot":
        expected_ids = [config["prompts"]["manual"]["direct"], config["prompts"]["winner"]]
    if [candidate.prompt_id for candidate in candidates] != expected_ids:
        raise ValueError("candidate registry IDs/order differ from locked E9 config")

    output = args.output or ROOT / stage["output"]
    generation = config["generation"]
    run_candidate_evaluation(
        models_path=ROOT / "configs/models.yaml",
        dataset_path=dataset_path,
        dataset_items=items,
        model_key=config["model"],
        candidates=candidates,
        claim_ids=[item.claim_id for item in items],
        seeds=stage["seeds"],
        output_path=output,
        stage=f"external_{args.stage}",
        decoding=generation["decoding"],
        thinking=bool(generation["thinking"]),
        do_sample=bool(generation["do_sample"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        repetition_penalty=float(generation["repetition_penalty"]),
        max_new_tokens=int(generation["max_new_tokens"]),
    )
    expected_rows = len(items) * 4 * len(candidates) * len(stage["seeds"])
    print(f"E9 {args.stage} raw: {output}")
    print(
        f"Locked records={len(items)}, prompts={len(candidates)}, "
        f"seeds={len(stage['seeds'])}, expected rows={expected_rows}"
    )


if __name__ == "__main__":
    main()
