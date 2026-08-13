"""Uncertainty quantification for repeated LLM argument-strength scores.

The estimators use only an ``estimate`` set of stochastic generations.  Their
targets are computed from disjoint ``held-out`` generations.  This separation
prevents an uncertainty signal from being credited merely for describing the
same samples from which it was calculated.

Two units are supported:

* argument rows quantify future numerical score instability;
* reference-oriented argument-pair rows quantify future ranking errors and
  preference instability.

Claims remain the grouping unit for cross-fitting and resampling because the
six pairs from one four-argument claim are not independent observations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math
import random
import statistics
from typing import Any, Iterable, Sequence


DEFAULT_UQ_SIGNALS = (
    "sampling_variance",
    "sampling_range",
    "sampling_mad",
    "score_option_entropy",
)

PAIR_UQ_SIGNALS = DEFAULT_UQ_SIGNALS + (
    "preference_entropy",
    "pair_disagreement_probability",
    "margin_uncertainty",
)


def _relation(value: float, tie_tolerance: float = 0.0) -> int:
    if value > tie_tolerance:
        return 1
    if value < -tie_tolerance:
        return -1
    return 0


def _entropy(values: Sequence[Any], *, maximum_categories: int | None = None) -> float:
    """Normalised empirical entropy in [0, 1]."""

    if not values:
        raise ValueError("values cannot be empty")
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    entropy = -sum(
        (count / len(values)) * math.log(count / len(values))
        for count in counts.values()
    )
    possible = min(len(values), maximum_categories or len(values))
    return entropy / math.log(possible) if possible > 1 else 0.0


def _median_absolute_deviation(values: Sequence[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def disagreement_probability(labels: Sequence[Any]) -> float:
    """Probability that two uniformly sampled runs have different labels."""

    if len(labels) < 2:
        return 0.0
    disagreements = sum(left != right for left, right in combinations(labels, 2))
    return disagreements / math.comb(len(labels), 2)


def sampling_signals(
    values: Sequence[float],
    *,
    value_span: float,
    option_labels: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Compute dispersion signals from repeated numerical samples.

    ``value_span`` is 100 for a 0--100 point score and 200 for a difference
    between two such scores.  Variance, range, and MAD are scale-normalised so
    every returned signal has a common higher-means-more-uncertain direction.
    """

    if len(values) < 2:
        raise ValueError("at least two samples are required")
    if value_span <= 0:
        raise ValueError("value_span must be positive")
    numeric = [float(value) for value in values]
    labels = list(option_labels) if option_labels is not None else numeric
    if len(labels) != len(numeric):
        raise ValueError("option_labels and values must have equal length")
    return {
        "sampling_variance": min(statistics.variance(numeric) / (value_span**2), 1.0),
        "sampling_range": (max(numeric) - min(numeric)) / value_span,
        "sampling_mad": _median_absolute_deviation(numeric) / value_span,
        "score_option_entropy": _entropy(labels),
    }


def pair_uncertainty_signals(
    margins: Sequence[float],
    *,
    tie_tolerance: float = 0.0,
) -> dict[str, float]:
    """Compute repeated-sampling uncertainty for a reference-oriented pair."""

    values = [float(value) for value in margins]
    relations = [_relation(value, tie_tolerance) for value in values]
    result = sampling_signals(values, value_span=200.0)
    result.update(
        {
            "preference_entropy": _entropy(relations, maximum_categories=3),
            "pair_disagreement_probability": disagreement_probability(relations),
            # Small absolute aggregate margins indicate uncertainty. The 25
            # point soft scale avoids an arbitrary hard threshold.
            "margin_uncertainty": 1.0 / (
                1.0 + abs(statistics.fmean(values)) / 25.0
            ),
        }
    )
    return result


def seed_partitions(
    seeds: Sequence[int],
    *,
    estimate_size: int,
    mode: str = "all",
    n_random: int = 50,
    random_seed: int = 2026,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Split run seeds into estimate and held-out sets.

    Ten runs with ``estimate_size=5`` and ``mode='all'`` produce all 252
    cross-fit partitions.  ``fixed`` returns one 5+5 partition; ``random``
    returns a reproducible subset of unique partitions.
    """

    ordered = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if len(ordered) != len(seeds):
        raise ValueError("seeds must be unique")
    if estimate_size < 2 or len(ordered) - estimate_size < 2:
        raise ValueError("estimate and held-out sets each require at least two seeds")
    all_estimates = list(combinations(ordered, estimate_size))
    if mode == "fixed":
        selected = all_estimates[:1]
    elif mode == "all":
        selected = all_estimates
    elif mode == "random":
        if n_random <= 0:
            raise ValueError("n_random must be positive")
        rng = random.Random(random_seed)
        selected = rng.sample(all_estimates, min(n_random, len(all_estimates)))
    else:
        raise ValueError("mode must be 'fixed', 'all', or 'random'")
    seed_set = set(ordered)
    return [
        (estimate, tuple(seed for seed in ordered if seed in seed_set - set(estimate)))
        for estimate in selected
    ]


def _prompt_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("prompt_id", row.get("prompt_condition", "unknown")))


def _thinking(row: dict[str, Any]) -> bool:
    return bool(row.get("thinking", (row.get("metadata") or {}).get("thinking", False)))


def _complete_score_cube(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Construct complete claim cells without inferring independence from runs."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("reference_rank") is None
            or not row.get("score_parse_ok")
            or row.get("score") is None
        ):
            continue
        grouped[
            (
                row.get("study", "uq_apo"),
                row["model_key"],
                _prompt_id(row),
                row.get("decoding", "hard"),
                _thinking(row),
                int(row["claim_id"]),
                row["stance"],
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        by_rank: dict[int, dict[int, float]] = defaultdict(dict)
        arguments: dict[int, str] = {}
        for row in values:
            rank = int(row["reference_rank"])
            seed = int(row["seed"])
            if seed in by_rank[rank]:
                raise ValueError(f"duplicate score for cell={key}, rank={rank}, seed={seed}")
            by_rank[rank][seed] = float(row["score"])
            arguments[rank] = str(row.get("argument", ""))
        seed_sets = {tuple(sorted(seed_scores)) for seed_scores in by_rank.values()}
        if len(by_rank) < 2 or len(seed_sets) != 1:
            continue
        seeds = next(iter(seed_sets))
        output.append(
            {
                "study": key[0],
                "model": key[1],
                "prompt_id": key[2],
                "decoding": key[3],
                "thinking": key[4],
                "claim_id": key[5],
                "stance": key[6],
                "claim": str(values[0].get("claim", "")),
                "dataset_item_metadata": dict(
                    (values[0].get("metadata") or {}).get("dataset_item_metadata") or {}
                ),
                "scores_by_rank": dict(by_rank),
                "arguments_by_rank": arguments,
                "seeds": seeds,
            }
        )
    return output


def _future_deviation(estimate_values: Sequence[float], heldout: Sequence[float], span: float) -> float:
    centre = statistics.median(estimate_values)
    return statistics.fmean(abs(value - centre) for value in heldout) / span


def build_argument_uq_rows(
    rows: Sequence[dict[str, Any]],
    partitions: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> list[dict[str, Any]]:
    """Build argument-level UQ estimators and genuinely held-out targets."""

    output: list[dict[str, Any]] = []
    for cell in _complete_score_cube(rows):
        available = set(cell["seeds"])
        for split_index, (estimate_seeds, heldout_seeds) in enumerate(partitions):
            requested = set(estimate_seeds) | set(heldout_seeds)
            if not requested <= available:
                continue
            for rank, seed_scores in sorted(cell["scores_by_rank"].items()):
                estimate = [seed_scores[int(seed)] for seed in estimate_seeds]
                heldout = [seed_scores[int(seed)] for seed in heldout_seeds]
                signals = sampling_signals(estimate, value_span=100.0)
                output.append(
                    {
                        "unit": "argument",
                        "split_id": split_index,
                        "estimate_seeds": "|".join(map(str, estimate_seeds)),
                        "heldout_seeds": "|".join(map(str, heldout_seeds)),
                        "study": cell["study"],
                        "model": cell["model"],
                        "prompt_id": cell["prompt_id"],
                        "decoding": cell["decoding"],
                        "thinking": cell["thinking"],
                        "claim_id": cell["claim_id"],
                        "stance": cell["stance"],
                        "claim": cell["claim"],
                        "source_topic": cell["dataset_item_metadata"].get("source_topic"),
                        "source_group_id": cell["dataset_item_metadata"].get("source_group_id"),
                        "panel": cell["dataset_item_metadata"].get("panel"),
                        "argument_rank": rank,
                        "argument": cell["arguments_by_rank"][rank],
                        **signals,
                        "estimate_mean": statistics.fmean(estimate),
                        "heldout_mean": statistics.fmean(heldout),
                        "future_absolute_deviation": _future_deviation(
                            estimate, heldout, 100.0
                        ),
                        "future_sampling_variance": min(
                            statistics.variance(heldout) / (100.0**2), 1.0
                        ),
                    }
                )
    return output


def build_pair_uq_rows(
    rows: Sequence[dict[str, Any]],
    partitions: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    tie_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Build pair-level UQ signals, future errors, and future instability."""

    output: list[dict[str, Any]] = []
    for cell in _complete_score_cube(rows):
        available = set(cell["seeds"])
        ranks = sorted(cell["scores_by_rank"])
        for split_index, (estimate_seeds, heldout_seeds) in enumerate(partitions):
            requested = set(estimate_seeds) | set(heldout_seeds)
            if not requested <= available:
                continue
            for better_rank, worse_rank in combinations(ranks, 2):
                estimate = [
                    cell["scores_by_rank"][better_rank][int(seed)]
                    - cell["scores_by_rank"][worse_rank][int(seed)]
                    for seed in estimate_seeds
                ]
                heldout = [
                    cell["scores_by_rank"][better_rank][int(seed)]
                    - cell["scores_by_rank"][worse_rank][int(seed)]
                    for seed in heldout_seeds
                ]
                estimate_relation = _relation(statistics.fmean(estimate), tie_tolerance)
                heldout_relation = _relation(statistics.fmean(heldout), tie_tolerance)
                heldout_run_relations = [
                    _relation(value, tie_tolerance) for value in heldout
                ]
                signals = pair_uncertainty_signals(
                    estimate, tie_tolerance=tie_tolerance
                )
                output.append(
                    {
                        "unit": "pair",
                        "split_id": split_index,
                        "estimate_seeds": "|".join(map(str, estimate_seeds)),
                        "heldout_seeds": "|".join(map(str, heldout_seeds)),
                        "study": cell["study"],
                        "model": cell["model"],
                        "prompt_id": cell["prompt_id"],
                        "decoding": cell["decoding"],
                        "thinking": cell["thinking"],
                        "claim_id": cell["claim_id"],
                        "stance": cell["stance"],
                        "claim": cell["claim"],
                        "source_topic": cell["dataset_item_metadata"].get("source_topic"),
                        "source_group_id": cell["dataset_item_metadata"].get("source_group_id"),
                        "panel": cell["dataset_item_metadata"].get("panel"),
                        "pair_id": f"rank_{better_rank}_vs_rank_{worse_rank}",
                        "better_rank": better_rank,
                        "worse_rank": worse_rank,
                        "rank_distance": worse_rank - better_rank,
                        "better_argument": cell["arguments_by_rank"][better_rank],
                        "worse_argument": cell["arguments_by_rank"][worse_rank],
                        **signals,
                        "estimate_mean_margin": statistics.fmean(estimate),
                        "heldout_mean_margin": statistics.fmean(heldout),
                        "estimate_reference_error": int(estimate_relation != 1),
                        "future_reference_error": int(heldout_relation != 1),
                        "future_pairwise_loss": (
                            0.0 if heldout_relation == 1 else 0.5 if heldout_relation == 0 else 1.0
                        ),
                        "future_absolute_deviation": _future_deviation(
                            estimate, heldout, 200.0
                        ),
                        "future_pair_disagreement": disagreement_probability(
                            heldout_run_relations
                        ),
                        "future_flip_from_estimate": statistics.fmean(
                            relation != estimate_relation
                            for relation in heldout_run_relations
                        ),
                    }
                )
    return output


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("vectors must have equal length of at least two")
    left_ranks = _average_ranks([float(value) for value in left])
    right_ranks = _average_ranks([float(value) for value in right])
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    )
    return numerator / denominator if denominator else None


def auroc(risks: Sequence[float], outcomes: Sequence[int]) -> float | None:
    """Tie-aware AUROC; higher risk should indicate outcome=1."""

    if len(risks) != len(outcomes) or not risks:
        raise ValueError("risks and outcomes must be non-empty and equal length")
    positive = [risk for risk, outcome in zip(risks, outcomes) if outcome == 1]
    negative = [risk for risk, outcome in zip(risks, outcomes) if outcome == 0]
    if not positive or not negative:
        return None
    credit = sum(
        1.0 if pos > neg else 0.5 if pos == neg else 0.0
        for pos in positive
        for neg in negative
    )
    return credit / (len(positive) * len(negative))


def average_precision(risks: Sequence[float], outcomes: Sequence[int]) -> float | None:
    if len(risks) != len(outcomes) or not risks:
        raise ValueError("risks and outcomes must be non-empty and equal length")
    positives = sum(outcomes)
    if positives == 0:
        return None
    ordered = sorted(
        enumerate(zip(risks, outcomes)), key=lambda pair: (-pair[1][0], pair[0])
    )
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, (_, outcome)) in enumerate(ordered, start=1):
        if outcome == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def aurc(risks: Sequence[float], losses: Sequence[float]) -> float:
    """Area under risk-coverage curve (lower is better)."""

    if len(risks) != len(losses) or not risks:
        raise ValueError("risks and losses must be non-empty and equal length")
    ordered_losses = [
        loss
        for _, loss in sorted(
            enumerate(losses), key=lambda pair: (risks[pair[0]], pair[0])
        )
    ]
    cumulative = 0.0
    curve = []
    for count, loss in enumerate(ordered_losses, start=1):
        cumulative += loss
        curve.append(cumulative / count)
    return statistics.fmean(curve)


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[int], *, bins: int = 10
) -> float:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("probabilities and outcomes must be non-empty and equal")
    if bins <= 0:
        raise ValueError("bins must be positive")
    grouped: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, outcome in zip(probabilities, outcomes):
        if not 0 <= probability <= 1:
            raise ValueError("probabilities must be in [0, 1]")
        index = min(int(probability * bins), bins - 1)
        grouped[index].append((probability, outcome))
    return sum(
        len(group) / len(probabilities)
        * abs(
            statistics.fmean(value for value, _ in group)
            - statistics.fmean(outcome for _, outcome in group)
        )
        for group in grouped
        if group
    )


def _fit_isotonic(xs: Sequence[float], ys: Sequence[int]):
    """Fit a non-decreasing PAVA calibrator and return its step function."""

    if len(xs) != len(ys) or not xs:
        raise ValueError("xs and ys must be non-empty and equal length")
    grouped: dict[float, list[int]] = defaultdict(list)
    for x, y in zip(xs, ys):
        grouped[float(x)].append(int(y))
    blocks = [
        {
            "low": x,
            "high": x,
            "weight": len(grouped[x]),
            "mean": statistics.fmean(grouped[x]),
        }
        for x in sorted(grouped)
    ]
    cursor = 0
    while cursor < len(blocks) - 1:
        if blocks[cursor]["mean"] <= blocks[cursor + 1]["mean"]:
            cursor += 1
            continue
        left = blocks[cursor]
        right = blocks[cursor + 1]
        weight = left["weight"] + right["weight"]
        merged = {
            "low": left["low"],
            "high": right["high"],
            "weight": weight,
            "mean": (
                left["mean"] * left["weight"]
                + right["mean"] * right["weight"]
            )
            / weight,
        }
        blocks[cursor : cursor + 2] = [merged]
        cursor = max(0, cursor - 1)

    def predict(value: float) -> float:
        for block in blocks:
            if value <= block["high"]:
                return float(block["mean"])
        return float(blocks[-1]["mean"])

    return predict


def cross_fitted_isotonic_risk(
    risks: Sequence[float],
    outcomes: Sequence[int],
    groups: Sequence[Any],
    *,
    folds: int = 5,
    seed: int = 2026,
) -> list[float]:
    """Calibrate a UQ signal without putting a claim in train and test."""

    if not (len(risks) == len(outcomes) == len(groups)) or not risks:
        raise ValueError("risks, outcomes, and groups must be non-empty and equal")
    unique_groups = list(dict.fromkeys(groups))
    if folds < 2 or folds > len(unique_groups):
        raise ValueError("folds must be between 2 and the number of groups")
    random.Random(seed).shuffle(unique_groups)
    fold_by_group = {group: index % folds for index, group in enumerate(unique_groups)}
    predictions = [0.0] * len(risks)
    for fold in range(folds):
        train = [index for index, group in enumerate(groups) if fold_by_group[group] != fold]
        test = [index for index, group in enumerate(groups) if fold_by_group[group] == fold]
        predictor = _fit_isotonic(
            [risks[index] for index in train],
            [outcomes[index] for index in train],
        )
        for index in test:
            predictions[index] = predictor(float(risks[index]))
    return predictions


def evaluate_uq_signal(
    rows: Sequence[dict[str, Any]],
    *,
    signal: str,
    binary_target: str,
    continuous_target: str,
    crossfit_folds: int = 5,
    seed: int = 2026,
) -> dict[str, Any]:
    """Evaluate ranking, selective prediction, and calibrated risk quality."""

    filtered = [
        row
        for row in rows
        if row.get(signal) is not None
        and row.get(binary_target) is not None
        and row.get(continuous_target) is not None
    ]
    if len(filtered) < 2:
        raise ValueError("not enough complete UQ rows")
    risks = [float(row[signal]) for row in filtered]
    binary = [int(row[binary_target]) for row in filtered]
    continuous = [float(row[continuous_target]) for row in filtered]
    groups = [row["claim_id"] for row in filtered]
    probabilities = cross_fitted_isotonic_risk(
        risks, binary, groups, folds=crossfit_folds, seed=seed
    )
    return {
        "signal": signal,
        "binary_target": binary_target,
        "continuous_target": continuous_target,
        "n_rows": len(filtered),
        "n_claims": len(set(groups)),
        "spearman": spearman(risks, continuous),
        "auroc": auroc(risks, binary),
        "auprc": average_precision(risks, binary),
        "aurc": aurc(risks, continuous),
        "brier": statistics.fmean(
            (probability - outcome) ** 2
            for probability, outcome in zip(probabilities, binary)
        ),
        "ece": expected_calibration_error(probabilities, binary),
        "positive_rate": statistics.fmean(binary),
    }


def evaluate_uq_by_partition(
    rows: Sequence[dict[str, Any]],
    *,
    signals: Iterable[str],
    binary_target: str,
    continuous_target: str,
    crossfit_folds: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate every 5+5 partition, then average without inflating n."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["split_id"])].append(row)
    per_partition: list[dict[str, Any]] = []
    for split_id, split_rows in sorted(grouped.items()):
        for signal in signals:
            result = evaluate_uq_signal(
                split_rows,
                signal=signal,
                binary_target=binary_target,
                continuous_target=continuous_target,
                crossfit_folds=crossfit_folds,
                seed=2026 + split_id,
            )
            per_partition.append({"split_id": split_id, **result})

    aggregate: list[dict[str, Any]] = []
    by_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_partition:
        by_signal[row["signal"]].append(row)
    for signal, values in sorted(by_signal.items()):
        result: dict[str, Any] = {
            "signal": signal,
            "n_partitions": len(values),
            "n_claims": values[0]["n_claims"],
            "binary_target": binary_target,
            "continuous_target": continuous_target,
        }
        for metric in ("spearman", "auroc", "auprc", "aurc", "brier", "ece"):
            available = [float(row[metric]) for row in values if row[metric] is not None]
            result[f"mean_{metric}"] = statistics.fmean(available) if available else None
        aggregate.append(result)
    return per_partition, aggregate
