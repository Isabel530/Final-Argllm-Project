#!/usr/bin/env python3
"""Create the next immutable registry from selected prompt IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import (  # noqa: E402
    read_candidate_registry,
    write_candidate_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.selection.open(encoding="utf-8") as handle:
        selected_ids = set(json.load(handle)["prompt_ids"])
    candidates = [
        candidate
        for candidate in read_candidate_registry(args.candidates)
        if candidate.prompt_id in selected_ids
    ]
    if {candidate.prompt_id for candidate in candidates} != selected_ids:
        missing = sorted(selected_ids - {candidate.prompt_id for candidate in candidates})
        raise ValueError(f"selected prompt IDs absent from registry: {missing}")
    write_candidate_registry(args.output, candidates)
    print(f"Selected candidate registry: {args.output}")


if __name__ == "__main__":
    main()
