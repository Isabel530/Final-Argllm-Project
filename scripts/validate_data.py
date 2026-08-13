#!/usr/bin/env python3
"""Validate both input datasets before any GPU job is submitted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.data import (  # noqa: E402
    load_main_dataset,
    load_representation_dataset,
    summarise_main_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main",
        type=Path,
        default=ROOT
        / "datasets/5.15Dataset/true_false_uncertain_gen_gpt-5_ver_sonnet-4-5.jsonl",
    )
    parser.add_argument(
        "--representation",
        type=Path,
        default=ROOT
        / "datasets/5.15Dataset/expanded_synonym_arguments_evaluation_tfu_150.jsonl",
    )
    args = parser.parse_args()

    main_items = load_main_dataset(args.main)
    representation = load_representation_dataset(args.representation)
    report = {
        "main": summarise_main_dataset(main_items),
        "representation": {
            "groups": len(representation),
            "claims": len({group.claim_id for group in representation}),
            "support_groups": sum(group.stance.value == "support" for group in representation),
            "attack_groups": sum(group.stance.value == "attack" for group in representation),
            "variants": sum(len(group.variants) for group in representation),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
