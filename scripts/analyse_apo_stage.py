#!/usr/bin/env python3
"""Summarise one APO stage and rank candidates conservatively."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo import (  # noqa: E402
    conservative_utility,
    pareto_front,
    select_survivors,
)
from argstrength.apo_experiment import (  # noqa: E402
    read_jsonl,
    summarise_candidate_evaluations,
)
from argstrength.tabular import write_csv, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=4)
    args = parser.parse_args()
    rows = [row for path in args.raw for row in read_jsonl(path)]
    evaluations, failures, pair_details = summarise_candidate_evaluations(rows)
    front = {row.prompt_id for row in pareto_front(evaluations)}
    selected = {row.prompt_id for row in select_survivors(evaluations, keep=args.keep)}
    summary = []
    for evaluation in evaluations:
        row = evaluation.to_dict()
        row["conservative_utility"] = conservative_utility(evaluation)
        row["pareto_optimal"] = evaluation.prompt_id in front
        row["selected"] = evaluation.prompt_id in selected
        row.pop("per_claim_accuracy", None)
        summary.append(row)
    summary.sort(key=lambda row: row["conservative_utility"], reverse=True)

    failure_rows = []
    for prompt_failures in failures.values():
        failure_rows.extend(asdict(failure) for failure in prompt_failures)
    write_csv(args.output_dir / "candidate_summary.csv", summary)
    write_csv(args.output_dir / "pair_diagnostics.csv", pair_details)
    write_csv(args.output_dir / "failure_cases.csv", failure_rows)
    write_json(
        args.output_dir / "selected_prompt_ids.json",
        {"keep": args.keep, "prompt_ids": list(selected)},
    )
    print(f"Candidate summary: {args.output_dir / 'candidate_summary.csv'}")
    print(f"Selected prompt IDs: {sorted(selected)}")


if __name__ == "__main__":
    main()
