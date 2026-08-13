#!/usr/bin/env python3
"""Create dissertation-ready figures from the extended-analysis CSV files."""

from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "processed" / "extended_analysis"
OUTPUT = ROOT / "results" / "figures" / "extended_analysis"

MODEL_LABELS = {
    "gpt_oss_20b": "GPT-OSS-20B",
    "mistral_7b": "Mistral-7B",
    "qwen3_8b": "Qwen3-8B",
}
MODEL_COLOURS = {
    "gpt_oss_20b": "#4472C4",
    "mistral_7b": "#ED7D31",
    "qwen3_8b": "#70AD47",
}
PROMPT_COLOURS = {
    "direct": "#4C78A8",
    "role": "#F58518",
    "rubric": "#54A24B",
    "role_rubric": "#B279A2",
}


def _read(filename: str) -> list[dict[str, str]]:
    path = INPUT / filename
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _boolean(value: str) -> bool:
    return value.lower() == "true"


def _save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _rank_distance_figure(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    stem: str,
) -> None:
    panels = sorted({(row["study"], row["model"]) for row in rows})
    ncols = 3
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 4.1 * nrows),
        sharex=True,
        squeeze=False,
    )
    for axis, (study, model) in zip(axes.flat, panels):
        available = [
            row for row in rows if row["study"] == study and row["model"] == model
        ]
        line_groups: dict[tuple[str, bool], list[dict[str, str]]] = defaultdict(list)
        for row in available:
            line_groups[(row["prompt"], _boolean(row["thinking"]))].append(row)
        for (prompt, thinking), values in sorted(line_groups.items()):
            values.sort(key=lambda row: int(row["rank_distance"]))
            x = [int(row["rank_distance"]) for row in values]
            y = [float(row[f"mean_{metric}"]) for row in values]
            low = [float(row[f"{metric}_ci_low"]) for row in values]
            high = [float(row[f"{metric}_ci_high"]) for row in values]
            label = prompt.replace("_", "+")
            if study == "reasoning_study":
                label += " · Think" if thinking else " · No-Think"
            axis.errorbar(
                x,
                y,
                yerr=[
                    [value - lower for value, lower in zip(y, low)],
                    [upper - value for value, upper in zip(y, high)],
                ],
                marker="o",
                linewidth=1.6,
                capsize=2,
                color=PROMPT_COLOURS[prompt],
                linestyle="--" if thinking else "-",
                alpha=0.9,
                label=label,
            )
        axis.set_title(
            f"{MODEL_LABELS.get(model, model)} · "
            f"{'Main' if study == 'main_study' else 'Reasoning' if study == 'reasoning_study' else 'Decoding'}"
        )
        axis.set_xticks([1, 2, 3])
        axis.set_xlabel("Reference-rank distance")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        if metric in {"pairwise_accuracy", "tie_rate", "seed_flip_rate"}:
            axis.set_ylim(bottom=0)
        axis.legend(fontsize=7, frameon=False, ncol=2)
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    fig.suptitle(f"{ylabel} by reference-rank distance", fontsize=15, y=1.01)
    fig.tight_layout()
    _save(fig, stem)


def plot_rank_distance() -> None:
    rows = _read("rank_distance_metrics.csv")
    _rank_distance_figure(
        rows,
        metric="pairwise_accuracy",
        ylabel="Tie-aware pairwise accuracy",
        stem="rank_distance_accuracy",
    )
    _rank_distance_figure(
        rows,
        metric="tie_rate",
        ylabel="Tie rate",
        stem="rank_distance_tie_rate",
    )
    _rank_distance_figure(
        rows,
        metric="seed_flip_rate",
        ylabel="Seed flip rate",
        stem="rank_distance_flip_rate",
    )


def plot_tie_transitions() -> None:
    rows = _read("qwen_tie_transition_by_distance.csv")
    rows.sort(
        key=lambda row: 0 if row["rank_distance"] == "all" else int(row["rank_distance"])
    )
    labels = [
        "All" if row["rank_distance"] == "all" else f"Distance {row['rank_distance']}"
        for row in rows
    ]
    correct = [float(row["tie_to_correct_rate"]) for row in rows]
    wrong = [float(row["tie_to_wrong_rate"]) for row in rows]
    tied = [float(row["tie_to_tie_rate"]) for row in rows]
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(x, correct, label="Tie → Correct", color="#59A14F")
    axis.bar(x, wrong, bottom=correct, label="Tie → Wrong", color="#E15759")
    axis.bar(
        x,
        tied,
        bottom=np.asarray(correct) + np.asarray(wrong),
        label="Tie → Tie",
        color="#BAB0AC",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Proportion of No-Think tied pairs")
    axis.set_title("Qwen: what happens to No-Think ties under Thinking?")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, "qwen_tie_transitions")


def plot_score_histograms() -> None:
    rows = [
        row
        for row in _read("score_histogram.csv")
        if row["study"] == "main_study"
    ]
    pooled: dict[tuple[str, int], int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        model = row["model"]
        score = int(row["score"])
        count = int(row["count"])
        pooled[(model, score)] += count
        totals[model] += count
    models = sorted(totals)
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 7.8), sharex=True)
    if len(models) == 1:
        axes = [axes]
    for axis, model in zip(axes, models):
        x = np.arange(101)
        y = np.asarray([pooled[(model, score)] / totals[model] for score in x])
        axis.bar(x, y, width=1.0, color=MODEL_COLOURS[model])
        axis.set_title(MODEL_LABELS.get(model, model), loc="left")
        axis.set_ylabel("Rate")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Generated score")
    fig.suptitle("Main Study score-use distributions", fontsize=15)
    fig.tight_layout()
    _save(fig, "score_histogram_by_model")


def plot_score_ranges() -> None:
    rows = [
        row
        for row in _read("score_scale_summary.csv")
        if row["study"] in {"main_study", "reasoning_study"}
    ]
    grouped: dict[tuple[str, str, bool], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["study"], row["model"], _boolean(row["thinking"]))
        ].append(float(row["mean_within_claim_score_range"]))
    entries = []
    for key, values in sorted(grouped.items()):
        entries.append((key, sum(values) / len(values)))
    labels = [
        f"{MODEL_LABELS.get(model, model)}\n"
        f"{'Main' if study == 'main_study' else ('Think' if thinking else 'No-Think')}"
        for (study, model, thinking), _ in entries
    ]
    values = [value for _, value in entries]
    colours = [MODEL_COLOURS[key[1]] for key, _ in entries]
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.bar(np.arange(len(entries)), values, color=colours)
    axis.set_xticks(np.arange(len(entries)), labels, rotation=25, ha="right")
    axis.set_ylabel("Mean within-claim score range")
    axis.set_title("Models use the 0–100 scale differently")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, "score_range_by_model")


def plot_margins() -> None:
    rows = [
        row
        for row in _read("pair_margin_analysis.csv")
        if row["rank_distance"] == "all"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for study, axis in zip(("main_study", "reasoning_study"), axes):
        available = [row for row in rows if row["study"] == study]
        for row in available:
            model = row["model"]
            marker = "^" if _boolean(row["thinking"]) else "o"
            axis.scatter(
                float(row["mean_raw_margin"]),
                float(row["mean_normalized_margin"]),
                color=MODEL_COLOURS[model],
                marker=marker,
                s=55,
                alpha=0.75,
            )
        axis.set_title("Main Study" if study == "main_study" else "Reasoning Study")
        axis.set_xlabel("Mean raw margin")
        axis.set_ylabel("Mean normalized margin")
        axis.grid(alpha=0.2)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MODEL_COLOURS[model],
            label=MODEL_LABELS[model],
            markersize=8,
        )
        for model in MODEL_COLOURS
    ]
    handles.extend(
        [
            plt.Line2D(
                [0], [0], marker="o", color="black", linestyle="", label="No-Think"
            ),
            plt.Line2D(
                [0], [0], marker="^", color="black", linestyle="", label="Think"
            ),
        ]
    )
    fig.legend(handles=handles, frameon=False, ncol=5, loc="upper center")
    fig.suptitle("Raw and within-claim-normalized score margins", y=1.04)
    fig.tight_layout()
    _save(fig, "raw_vs_normalized_margin")


def plot_claim_difficulty_heatmap() -> None:
    ranking = _read("claim_difficulty_ranking.csv")
    top = sorted(ranking, key=lambda row: int(row["difficulty_rank"]))[:20]
    top_ids = {int(row["claim_id"]) for row in top}
    distance_rows = [
        row
        for row in _read("rank_distance_claim_metrics.csv")
        if row["study"] == "main_study" and int(row["claim_id"]) in top_ids
    ]
    weighted: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
    for row in distance_rows:
        weighted[(int(row["claim_id"]), row["model"])].append(
            (float(row["pairwise_accuracy"]), int(row["n_pairs"]))
        )
    models = ["gpt_oss_20b", "mistral_7b", "qwen3_8b"]
    matrix = []
    labels = []
    for row in top:
        claim_id = int(row["claim_id"])
        labels.append(f"{claim_id}: {row['claim'][:42]}")
        model_values = []
        for model in models:
            values = weighted[(claim_id, model)]
            numerator = sum(value * weight for value, weight in values)
            denominator = sum(weight for _, weight in values)
            model_values.append(numerator / denominator)
        matrix.append(model_values)
    fig, axis = plt.subplots(figsize=(8.3, 9.5))
    image = axis.imshow(np.asarray(matrix), vmin=0, vmax=1, cmap="RdYlGn")
    axis.set_xticks(
        np.arange(len(models)), [MODEL_LABELS[model] for model in models]
    )
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    axis.set_title("Twenty most difficult Main Study claims")
    colourbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
    colourbar.set_label("Mean pairwise accuracy")
    fig.tight_layout()
    _save(fig, "claim_difficulty_heatmap")


def plot_tradeoff() -> None:
    rows = _read("tradeoff_cells.csv")
    studies = sorted({row["study"] for row in rows})
    fig, axes = plt.subplots(1, len(studies), figsize=(6 * len(studies), 5.2))
    if len(studies) == 1:
        axes = [axes]
    for axis, study in zip(axes, studies):
        available = [row for row in rows if row["study"] == study]
        token_values = [float(row["mean_generated_tokens"]) for row in available]
        max_log = max(math.log1p(value) for value in token_values) or 1
        for row in available:
            model = row["model"]
            size = 35 + 230 * math.log1p(float(row["mean_generated_tokens"])) / max_log
            marker = "^" if _boolean(row["thinking"]) else "o"
            axis.scatter(
                float(row["mean_pairwise_accuracy"]),
                float(row["mean_seed_consistency"]),
                s=size,
                color=MODEL_COLOURS[model],
                marker=marker,
                alpha=0.72,
                edgecolor="black",
                linewidth=0.4,
            )
            if study != "main_study" or row["prompt"] in {"direct", "rubric"}:
                label = row["prompt"].replace("_", "+")
                if study == "reasoning_study":
                    label += " T" if _boolean(row["thinking"]) else " NT"
                axis.annotate(
                    label,
                    (
                        float(row["mean_pairwise_accuracy"]),
                        float(row["mean_seed_consistency"]),
                    ),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
        axis.set_xlabel("Pairwise accuracy")
        axis.set_ylabel("Seed consistency")
        axis.set_title(study.replace("_", " ").title())
        axis.grid(alpha=0.2)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MODEL_COLOURS[model],
            markeredgecolor="black",
            label=MODEL_LABELS[model],
            markersize=8,
        )
        for model in MODEL_COLOURS
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center")
    fig.suptitle(
        "Accuracy–stability–cost trade-off\nBubble area scales with mean tokens",
        y=1.04,
    )
    fig.tight_layout()
    _save(fig, "accuracy_stability_cost")


def plot_equivalence() -> None:
    rows = _read("equivalence_tests.csv")
    rows.sort(key=lambda row: (row["model"], row["metric"], row["contrast"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=False)
    for axis, metric in zip(axes, ("pairwise_accuracy", "spearman")):
        available = [row for row in rows if row["metric"] == metric]
        y = np.arange(len(available))
        means = np.asarray([float(row["mean_difference"]) for row in available])
        low = np.asarray([float(row["ci_low"]) for row in available])
        high = np.asarray([float(row["ci_high"]) for row in available])
        colours = [
            MODEL_COLOURS[row["model"]]
            if _boolean(row["equivalent_0_05_holm"])
            else "#7F7F7F"
            for row in available
        ]
        for index, row in enumerate(available):
            axis.errorbar(
                means[index],
                y[index],
                xerr=[
                    [means[index] - low[index]],
                    [high[index] - means[index]],
                ],
                marker="o",
                color=colours[index],
                capsize=3,
            )
        bound = abs(float(available[0]["upper_bound"]))
        axis.axvline(0, color="black", linewidth=1)
        axis.axvspan(-bound, bound, color="#D9EAD3", alpha=0.5)
        axis.axvline(-bound, color="#38761D", linestyle="--", linewidth=1)
        axis.axvline(bound, color="#38761D", linestyle="--", linewidth=1)
        labels = [
            f"{MODEL_LABELS[row['model']]}\n{row['contrast'].replace('_minus_', ' − ')}"
            for row in available
        ]
        axis.set_yticks(y, labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("Prompt − Direct")
        axis.set_title(
            "Pairwise accuracy (±0.02)"
            if metric == "pairwise_accuracy"
            else "Spearman (±0.03)"
        )
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Secondary prompt-equivalence analysis (90% CIs)", fontsize=15)
    fig.tight_layout()
    _save(fig, "prompt_equivalence_forest")


def main() -> None:
    plot_rank_distance()
    plot_tie_transitions()
    plot_score_histograms()
    plot_score_ranges()
    plot_margins()
    plot_claim_difficulty_heatmap()
    plot_tradeoff()
    plot_equivalence()
    for path in sorted(OUTPUT.glob("*")):
        print(f"[figure] {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
