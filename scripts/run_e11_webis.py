#!/usr/bin/env python3
"""Run the excluded E11 pilot or the once-only frozen Webis formal test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import (  # noqa: E402
    read_candidate_registry,
    run_candidate_evaluation,
)
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.webis_external import load_webis_generation_dataset  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/e11_webis_external.yaml"
    )
    parser.add_argument("--stage", choices=("pilot", "formal"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    stage = config["stages"][args.stage]
    manifest_path = ROOT / config["preparation"]["manifest"]
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "locked_before_generation":
        raise ValueError("E11 manifest is not locked_before_generation")

    dataset_path = ROOT / stage["dataset"]
    pair_path = ROOT / stage["pairs"]
    registry_path = ROOT / stage["candidates"]
    locked = manifest[args.stage]
    checks = (
        (dataset_path, locked["generation_sha256"], "generation dataset"),
        (pair_path, locked["pairs_sha256"], "pair labels"),
        (registry_path, locked["candidate_registry_sha256"], "candidate registry"),
    )
    for path, expected, label in checks:
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"{label} fingerprint mismatch: {observed} != {expected}")

    items = load_webis_generation_dataset(dataset_path)
    candidates = read_candidate_registry(registry_path)
    direct = config["prompts"]["manual"]["direct"]
    rubric = config["prompts"]["manual"]["rubric"]
    winner = config["prompts"]["winner"]
    expected_ids = [direct, winner] if args.stage == "pilot" else [direct, rubric, winner]
    if [candidate.prompt_id for candidate in candidates] != expected_ids:
        raise ValueError("candidate registry IDs/order differ from locked E11 config")

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
        stage=f"e11_webis_{args.stage}",
        decoding=generation["decoding"],
        thinking=bool(generation["thinking"]),
        do_sample=bool(generation["do_sample"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        repetition_penalty=float(generation["repetition_penalty"]),
        max_new_tokens=int(generation["max_new_tokens"]),
    )
    arguments_per_prompt_seed = sum(len(item.arguments) for item in items)
    expected_rows = arguments_per_prompt_seed * len(candidates) * len(stage["seeds"])
    print(f"E11 {args.stage} raw: {output}")
    print(
        f"Locked topics={locked['topics']}, groups={len(items)}, "
        f"human pairs={locked['eligible_dimension_pairs']}, prompts={len(candidates)}, "
        f"seeds={len(stage['seeds'])}, expected rows={expected_rows}"
    )


if __name__ == "__main__":
    main()
