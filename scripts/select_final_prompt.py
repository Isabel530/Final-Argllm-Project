#!/usr/bin/env python3
"""Select one validation winner using accuracy, stability, UQ calibration, and cost."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo import (  # noqa: E402
    CandidateEvaluation,
    conservative_utility,
    pareto_front,
)
from argstrength.tabular import write_csv, write_json  # noqa: E402


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row, key, default=0.0):
    value = row.get(key)
    return default if value in (None, "", "None") else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--pair-uq-summary", type=Path, required=True)
    parser.add_argument(
        "--uq-signal",
        default="sampling_variance",
        help="Must be selected before looking at internal-test results.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = _read_csv(args.candidate_summary)
    uq_rows = [
        row
        for row in _read_csv(args.pair_uq_summary)
        if row["signal"] == args.uq_signal
    ]
    brier_by_prompt = {row["prompt_id"]: float(row["mean_brier"]) for row in uq_rows}
    missing = {row["prompt_id"] for row in candidates} - set(brier_by_prompt)
    if missing:
        raise ValueError(f"UQ summary lacks prompts: {sorted(missing)}")

    evaluations = []
    for row in candidates:
        evaluations.append(
            CandidateEvaluation(
                prompt_id=row["prompt_id"],
                stage="validation",
                n_claims=int(row["n_claims"]),
                n_seeds=int(row["n_seeds"]),
                pairwise_accuracy=float(row["pairwise_accuracy"]),
                accuracy_se=float(row["accuracy_se"]),
                score_variance=float(row["score_variance"]),
                seed_flip_rate=float(row["seed_flip_rate"]),
                brier=brier_by_prompt[row["prompt_id"]],
                mean_tokens=float(row["mean_tokens"]),
                mean_seconds=float(row["mean_seconds"]),
                invalid_rate=_number(row, "invalid_rate"),
            )
        )
    front = {row.prompt_id for row in pareto_front(evaluations)}
    ranked = sorted(evaluations, key=conservative_utility, reverse=True)
    winner = ranked[0]
    table = []
    for rank, evaluation in enumerate(ranked, start=1):
        row = evaluation.to_dict()
        row.pop("per_claim_accuracy", None)
        row.update(
            rank=rank,
            uq_signal=args.uq_signal,
            pareto_optimal=evaluation.prompt_id in front,
            conservative_utility=conservative_utility(evaluation),
            selected=evaluation.prompt_id == winner.prompt_id,
        )
        table.append(row)
    write_csv(args.output_dir / "final_prompt_selection.csv", table)
    write_json(
        args.output_dir / "selected_prompt_ids.json",
        {
            "prompt_ids": [winner.prompt_id],
            "uq_signal": args.uq_signal,
            "rule": "maximum pre-specified conservative utility on validation",
            "warning": "Freeze this prompt before running internal_test.",
        },
    )
    print(f"Frozen validation winner: {winner.prompt_id}")


if __name__ == "__main__":
    main()
