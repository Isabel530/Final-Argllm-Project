#!/usr/bin/env python3
"""Produce frozen-prompt claim-level summaries, paired CIs, tests, and plots."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import read_jsonl  # noqa: E402
from argstrength.paired_controls import (  # noqa: E402
    build_claim_metrics,
    paired_prompt_contrasts,
    summarise_prompts,
)
from argstrength.tabular import write_csv, write_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot(rows, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    primary = [row for row in rows if row["scope"] == "all" and row["metric"] in {"pairwise_accuracy", "seed_flip_rate"}]
    if not primary:
        return
    metrics = ("pairwise_accuracy", "seed_flip_rate")
    titles = ("Pairwise accuracy", "Seed flip rate")
    figure, axes = plt.subplots(1, 2, figsize=(10, max(3.2, 0.55 * len(primary))))
    for axis, metric, title in zip(axes, metrics, titles):
        values = [row for row in primary if row["metric"] == metric]
        labels = [row["baseline_label"] for row in values]
        means = [row["difference_winner_minus_baseline"] for row in values]
        lows = [row["difference_ci_low"] for row in values]
        highs = [row["difference_ci_high"] for row in values]
        positions = list(range(len(values)))
        axis.errorbar(
            means,
            positions,
            xerr=[[mean - low for mean, low in zip(means, lows)], [high - mean for mean, high in zip(means, highs)]],
            fmt="o",
            capsize=3,
        )
        axis.axvline(0, color="black", linewidth=1, linestyle="--")
        axis.set_yticks(positions, labels)
        axis.set_title(title)
        axis.set_xlabel("Winner minus baseline (95% claim bootstrap CI)")
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "paired_contrasts.png", dpi=200)
    figure.savefig(output_dir / "paired_contrasts.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path, nargs="+")
    parser.add_argument("--winner-id", required=True)
    parser.add_argument("--baseline-id", action="append", dest="baseline_ids")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--permutation-resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [row for path in args.raw for row in read_jsonl(path)]
    claim_rows = build_claim_metrics(rows, winner_id=args.winner_id)
    summary = summarise_prompts(
        claim_rows,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    contrasts = paired_prompt_contrasts(
        claim_rows,
        winner_id=args.winner_id,
        baseline_ids=args.baseline_ids,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "claim_metrics.csv", claim_rows)
    write_csv(args.output_dir / "prompt_summary.csv", summary)
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "raw": [{"path": str(path), "sha256": _sha256(path)} for path in args.raw],
            "winner_id": args.winner_id,
            "baseline_ids": args.baseline_ids,
            "independent_unit": "claim",
            "bootstrap_resamples": args.bootstrap_resamples,
            "permutation_resamples": args.permutation_resamples,
            "seed": args.seed,
            "holm_family": "within scope and metric",
        },
    )
    _plot(contrasts, args.output_dir)
    print(f"Prompt summary: {args.output_dir / 'prompt_summary.csv'}")
    print(f"Paired contrasts: {args.output_dir / 'paired_contrasts.csv'}")


if __name__ == "__main__":
    main()
