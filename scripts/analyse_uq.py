#!/usr/bin/env python3
"""Evaluate 5-run UQ signals against five disjoint held-out generations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import read_jsonl  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.tabular import write_csv, write_json  # noqa: E402
from argstrength.uq import (  # noqa: E402
    aurc,
    build_argument_uq_rows,
    build_pair_uq_rows,
    evaluate_uq_by_partition,
    seed_partitions,
    spearman,
)


def _argument_results(rows, signals, target):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["split_id"])].append(row)
    per_partition = []
    for split_id, values in sorted(grouped.items()):
        for signal in signals:
            usable = [
                row
                for row in values
                if row.get(signal) is not None and row.get(target) is not None
            ]
            risks = [float(row[signal]) for row in usable]
            targets = [float(row[target]) for row in usable]
            per_partition.append(
                {
                    "split_id": split_id,
                    "signal": signal,
                    "target": target,
                    "n_rows": len(usable),
                    "n_claims": len({row["claim_id"] for row in usable}),
                    "spearman": spearman(risks, targets),
                    "aurc": aurc(risks, targets),
                }
            )
    aggregate = []
    by_signal = defaultdict(list)
    for row in per_partition:
        by_signal[row["signal"]].append(row)
    for signal, values in sorted(by_signal.items()):
        correlations = [row["spearman"] for row in values if row["spearman"] is not None]
        aggregate.append(
            {
                "signal": signal,
                "n_partitions": len(values),
                "n_claims": values[0]["n_claims"],
                "target": target,
                "mean_spearman": statistics.fmean(correlations) if correlations else None,
                "mean_aurc": statistics.fmean(row["aurc"] for row in values),
            }
        )
    return per_partition, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/uq_apo.yaml")
    parser.add_argument(
        "--stage", choices=("validation", "internal_test"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    seeds = config["stages"][args.stage]["seeds"]
    uq_config = config["uq"]
    partitions = seed_partitions(
        seeds,
        estimate_size=int(uq_config["estimate_runs"]),
        mode=uq_config["seed_partition_mode"],
        random_seed=int(config["split_seed"]),
    )
    rows = [row for path in args.raw for row in read_jsonl(path)]
    pair_rows = build_pair_uq_rows(rows, partitions)
    argument_rows = build_argument_uq_rows(rows, partitions)
    if not pair_rows or not argument_rows:
        raise ValueError(
            "No complete UQ cells: each prompt/claim/argument needs all configured seeds"
        )
    # Prompts are separate estimators. Pooling their rows here would make a
    # prompt appear calibrated partly because another prompt predicts well.
    pair_partition, pair_aggregate = [], []
    argument_partition, argument_aggregate = [], []
    prompt_ids = sorted({row["prompt_id"] for row in pair_rows})
    for prompt_id in prompt_ids:
        prompt_pairs = [row for row in pair_rows if row["prompt_id"] == prompt_id]
        per_split, aggregate = evaluate_uq_by_partition(
            prompt_pairs,
            signals=uq_config["pair_signals"],
            binary_target=uq_config["binary_target"],
            continuous_target=uq_config["continuous_target"],
            crossfit_folds=int(uq_config["crossfit_folds"]),
        )
        pair_partition.extend({"prompt_id": prompt_id, **row} for row in per_split)
        pair_aggregate.extend({"prompt_id": prompt_id, **row} for row in aggregate)

        prompt_arguments = [
            row for row in argument_rows if row["prompt_id"] == prompt_id
        ]
        per_split, aggregate = _argument_results(
            prompt_arguments,
            uq_config["argument_signals"],
            "future_absolute_deviation",
        )
        argument_partition.extend(
            {"prompt_id": prompt_id, **row} for row in per_split
        )
        argument_aggregate.extend(
            {"prompt_id": prompt_id, **row} for row in aggregate
        )

    write_csv(args.output_dir / "pair_uq_rows.csv", pair_rows)
    write_csv(args.output_dir / "argument_uq_rows.csv", argument_rows)
    write_csv(args.output_dir / "pair_uq_per_partition.csv", pair_partition)
    write_csv(args.output_dir / "pair_uq_summary.csv", pair_aggregate)
    write_csv(
        args.output_dir / "argument_uq_per_partition.csv", argument_partition
    )
    write_csv(args.output_dir / "argument_uq_summary.csv", argument_aggregate)
    write_json(
        args.output_dir / "uq_manifest.json",
        {
            "stage": args.stage,
            "seeds": seeds,
            "estimate_runs": uq_config["estimate_runs"],
            "heldout_runs": uq_config["heldout_runs"],
            "partition_mode": uq_config["seed_partition_mode"],
            "n_partitions": len(partitions),
            "n_pair_rows": len(pair_rows),
            "n_argument_rows": len(argument_rows),
            "warning": (
                "The 252 seed partitions are repeated views of the same claims, "
                "not 252 independent datasets."
            ),
        },
    )
    print(f"Pair UQ summary: {args.output_dir / 'pair_uq_summary.csv'}")
    print(f"Argument UQ summary: {args.output_dir / 'argument_uq_summary.csv'}")


if __name__ == "__main__":
    main()
