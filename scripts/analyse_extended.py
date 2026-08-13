#!/usr/bin/env python3
"""Run the A--F extended reliability analyses on existing generations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.extended_analysis import (  # noqa: E402
    accuracy_stability_cost_table,
    claim_difficulty_analysis,
    complete_score_cells,
    equivalence_tests,
    pair_observations,
    qwen_tie_transition_analysis,
    rank_distance_summary,
    read_jsonl_files,
    score_scale_analysis,
    write_csv,
)


def _default_paths(pattern: str) -> list[Path]:
    return sorted(ROOT.glob(pattern))


def _existing(paths: list[Path], label: str) -> list[Path]:
    available = [path for path in paths if path.is_file()]
    if not available:
        raise FileNotFoundError(f"No {label} files found: {paths}")
    return available


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse rank distance, tie transitions, score calibration, claim "
            "difficulty, accuracy/stability/cost trade-offs, and equivalence."
        )
    )
    parser.add_argument(
        "--main",
        type=Path,
        nargs="+",
        default=_default_paths("results/raw/main_study_*.jsonl"),
    )
    parser.add_argument(
        "--reasoning",
        type=Path,
        nargs="+",
        default=_default_paths("results/raw/reasoning_study_*.jsonl"),
    )
    parser.add_argument(
        "--decoding",
        type=Path,
        nargs="+",
        default=_default_paths("results/raw/decoding_validation_*.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "processed" / "extended_analysis",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--pairwise-equivalence-bound", type=float, default=0.02)
    parser.add_argument("--spearman-equivalence-bound", type=float, default=0.03)
    args = parser.parse_args()

    main_paths = _existing(args.main, "Main Study")
    reasoning_paths = _existing(args.reasoning, "Reasoning Study")
    decoding_paths = _existing(args.decoding, "Decoding Validation")
    paths = main_paths + reasoning_paths + decoding_paths

    print(f"[read] {len(paths)} raw result files")
    rows = read_jsonl_files(paths)
    cells = complete_score_cells(rows)
    pairs = pair_observations(cells)
    print(
        f"[data] {len(rows)} generations, {len(cells)} complete claim cells, "
        f"{len(pairs)} derived pairs"
    )

    claim_distance, rank_summary, rank_effects = rank_distance_summary(
        pairs, bootstrap_resamples=args.bootstrap_resamples
    )
    tie_detail, tie_summary = qwen_tie_transition_analysis(
        pairs, bootstrap_resamples=args.bootstrap_resamples
    )
    scale_summary, histogram, margins = score_scale_analysis(rows, cells, pairs)
    difficulty, pair_disagreement, audit = claim_difficulty_analysis(cells, pairs)
    tradeoff = accuracy_stability_cost_table(rows, cells, pairs)
    equivalence = equivalence_tests(
        cells,
        pairs,
        pairwise_bound=args.pairwise_equivalence_bound,
        spearman_bound=args.spearman_equivalence_bound,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "rank_distance_claim_metrics.csv": claim_distance,
        "rank_distance_metrics.csv": rank_summary,
        "rank_distance_effects.csv": rank_effects,
        "qwen_tie_transitions.csv": tie_detail,
        "qwen_tie_transition_by_distance.csv": tie_summary,
        "score_scale_summary.csv": scale_summary,
        "score_histogram.csv": histogram,
        "score_heaping.csv": [
            {
                key: row[key]
                for key in (
                    "study",
                    "model",
                    "prompt",
                    "decoding",
                    "thinking",
                    "n_scores",
                    "n_unique_scores",
                    "multiple_of_10_rate",
                    "multiple_of_5_rate",
                    "ends_in_5_rate",
                    "score_use_entropy",
                    "score_use_entropy_normalized_101",
                )
            }
            for row in scale_summary
        ],
        "pair_margin_analysis.csv": margins,
        "claim_difficulty_ranking.csv": difficulty,
        "claim_pair_disagreements.csv": pair_disagreement,
        "claim_error_audit.csv": audit,
        "tradeoff_cells.csv": tradeoff,
        "equivalence_tests.csv": equivalence,
    }
    for filename, table in outputs.items():
        path = output_dir / filename
        write_csv(path, table)
        print(f"[write] {path.relative_to(ROOT)} ({len(table)} rows)")


if __name__ == "__main__":
    main()
