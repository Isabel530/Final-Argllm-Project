"""Generation and analysis adapters for UQ-guided prompt search."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence
from uuid import uuid4

from .apo import (
    CandidateEvaluation,
    FailureCase,
    PromptCandidate,
    accuracy_standard_error,
    build_candidate_prompt,
)
from .data import load_main_dataset
from .generation import GenerationSettings, HuggingFaceScoreGenerator
from .parsing import parse_score
from .runner import _stable_generation_seed, load_yaml
from .schemas import ArgumentSet, GenerationRecord
from .uq import _complete_score_cube, pair_uncertainty_signals


def write_candidate_registry(
    path: str | Path,
    candidates: Iterable[PromptCandidate],
    *,
    append: bool = False,
) -> Path:
    """Persist full readable prompt candidates separately from score rows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_candidate_registry(path: str | Path) -> list[PromptCandidate]:
    output: dict[str, PromptCandidate] = {}
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidate = PromptCandidate(
                    instruction=row["instruction"],
                    iteration=int(row["iteration"]),
                    source=row["source"],
                    parent_ids=tuple(row.get("parent_ids", ())),
                    label=row.get("label"),
                    metadata=dict(row.get("metadata", {})),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid candidate") from exc
            if row.get("prompt_id") not in {None, candidate.prompt_id}:
                raise ValueError(f"{path}:{line_number}: prompt hash does not match text")
            output[candidate.prompt_id] = candidate
    if not output:
        raise ValueError(f"{path}: no candidates found")
    return list(output.values())


def _completed_candidate_keys(path: Path) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: corrupt JSONL") from exc
            keys.add(
                (
                    row["model_key"],
                    (row.get("metadata") or {}).get(
                        "prompt_id", row["prompt_condition"]
                    ),
                    row["decoding"],
                    bool(row.get("thinking", False)),
                    int(row["seed"]),
                    int(row["claim_id"]),
                    row["argument_id"],
                )
            )
    return keys


def run_candidate_evaluation(
    *,
    models_path: str | Path,
    dataset_path: str | Path,
    model_key: str,
    dataset_items: Sequence[ArgumentSet] | None = None,
    candidates: Sequence[PromptCandidate],
    claim_ids: Sequence[int],
    seeds: Sequence[int],
    output_path: str | Path,
    stage: str,
    decoding: str = "hard",
    thinking: bool = False,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 12,
) -> Path:
    """Evaluate readable prompt candidates with resumable raw checkpoints."""

    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("claim_ids must be unique")
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("at least two unique seeds are required")
    models = load_yaml(models_path).get("models", {})
    if model_key not in models:
        raise KeyError(f"unknown model key {model_key!r}")
    items = list(dataset_items) if dataset_items is not None else load_main_dataset(dataset_path)
    by_id = {item.claim_id: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("dataset claim IDs must be unique")
    unknown = set(claim_ids) - set(by_id)
    if unknown:
        raise ValueError(f"unknown claim IDs: {sorted(unknown)}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_candidate_keys(output)
    run_id = uuid4().hex
    settings = GenerationSettings(
        decoding=decoding,
        thinking=thinking,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
    )
    model_config = dict(models[model_key])
    model_name = model_config.pop("name")
    supported = model_config.pop("thinking_modes", [False])
    if thinking not in supported:
        raise ValueError(f"{model_key} does not support thinking={thinking}")

    generator = HuggingFaceScoreGenerator(model_name, **model_config)
    try:
        with output.open("a", encoding="utf-8") as handle:
            for candidate in candidates:
                for seed in seeds:
                    for claim_id in claim_ids:
                        item = by_id[claim_id]
                        for argument_index, (argument, reference_rank) in enumerate(
                            zip(item.arguments, item.reference_ranks), start=1
                        ):
                            argument_id = f"rank_{argument_index}"
                            key = (
                                model_key,
                                candidate.prompt_id,
                                decoding,
                                thinking,
                                int(seed),
                                claim_id,
                                argument_id,
                            )
                            if key in completed:
                                continue
                            spec = build_candidate_prompt(
                                candidate,
                                claim=item.claim,
                                argument=argument,
                                stance=item.stance,
                            )
                            effective_seed = _stable_generation_seed(
                                int(seed), claim_id, argument_id
                            )
                            generated = generator.generate(
                                spec, seed=effective_seed, settings=settings
                            )
                            exact = parse_score(generated.raw_output, strict=True)
                            parsed = parse_score(generated.raw_output, strict=False)
                            record = GenerationRecord(
                                run_id=run_id,
                                study="uq_apo",
                                model_key=model_key,
                                model_name=model_name,
                                prompt_condition=candidate.prompt_id,
                                decoding=decoding,
                                thinking=thinking,
                                seed=int(seed),
                                claim_id=claim_id,
                                claim=item.claim,
                                stance=item.stance.value,
                                argument_id=argument_id,
                                argument=argument,
                                reference_rank=reference_rank,
                                raw_output=generated.raw_output,
                                score=parsed.score,
                                score_parse_ok=parsed.ok,
                                score_parse_error=parsed.error,
                                format_ok=exact.ok,
                                format_error=exact.error,
                                generated_tokens=generated.generated_tokens,
                                hit_max_new_tokens=generated.hit_max_new_tokens,
                                generation_seconds=generated.elapsed_seconds,
                                metadata={
                                    "prompt_id": candidate.prompt_id,
                                    "prompt_label": candidate.label,
                                    "prompt_iteration": candidate.iteration,
                                    "prompt_source": candidate.source,
                                    "stage": stage,
                                    "dataset_item_metadata": item.metadata,
                                    "effective_generation_seed": effective_seed,
                                    "max_new_tokens_setting": max_new_tokens,
                                    "max_new_tokens_limit": generated.max_new_tokens_limit,
                                },
                            )
                            handle.write(
                                json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                            )
                            handle.flush()
                            completed.add(key)
    finally:
        generator.close()
    return output


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    output = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return output


def summarise_candidate_evaluations(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[CandidateEvaluation], dict[str, list[FailureCase]], list[dict[str, Any]]]:
    """Calculate screening accuracy, variance, flips, cost, and failures."""

    cubes = _complete_score_cube(rows)
    claim_accuracy: dict[str, list[float]] = defaultdict(list)
    score_variances: dict[str, list[float]] = defaultdict(list)
    pair_flips: dict[str, list[float]] = defaultdict(list)
    failures: dict[str, list[FailureCase]] = defaultdict(list)
    pair_details: list[dict[str, Any]] = []

    for cell in cubes:
        prompt_id = cell["prompt_id"]
        ranks = sorted(cell["scores_by_rank"])
        per_claim_pair_accuracy = []
        for seed_scores in cell["scores_by_rank"].values():
            score_variances[prompt_id].append(
                statistics.variance(seed_scores.values()) / (100.0**2)
            )
        for better_rank, worse_rank in combinations(ranks, 2):
            seed_margins = tuple(
                cell["scores_by_rank"][better_rank][seed]
                - cell["scores_by_rank"][worse_rank][seed]
                for seed in cell["seeds"]
            )
            mean_margin = statistics.fmean(seed_margins)
            pair_accuracy = 1.0 if mean_margin > 0 else 0.5 if mean_margin == 0 else 0.0
            relations = [1 if value > 0 else -1 if value < 0 else 0 for value in seed_margins]
            flip = float(len(set(relations)) > 1)
            signals = pair_uncertainty_signals(seed_margins)
            uncertainty = statistics.fmean(
                (
                    signals["preference_entropy"],
                    signals["pair_disagreement_probability"],
                    signals["margin_uncertainty"],
                )
            )
            per_claim_pair_accuracy.append(pair_accuracy)
            pair_flips[prompt_id].append(flip)
            pair_details.append(
                {
                    "prompt_id": prompt_id,
                    "claim_id": cell["claim_id"],
                    "stance": cell["stance"],
                    "pair_id": f"rank_{better_rank}_vs_rank_{worse_rank}",
                    "rank_distance": worse_rank - better_rank,
                    "pairwise_accuracy": pair_accuracy,
                    "seed_flip": flip,
                    "uncertainty": uncertainty,
                    "mean_margin": mean_margin,
                    "seed_margins": "|".join(str(value) for value in seed_margins),
                }
            )
            if pair_accuracy < 1.0 or flip or uncertainty >= 0.5:
                failures[prompt_id].append(
                    FailureCase(
                        prompt_id=prompt_id,
                        claim_id=cell["claim_id"],
                        claim=cell["claim"],
                        stance=cell["stance"],
                        better_argument=cell["arguments_by_rank"][better_rank],
                        worse_argument=cell["arguments_by_rank"][worse_rank],
                        rank_distance=worse_rank - better_rank,
                        pairwise_accuracy=pair_accuracy,
                        uncertainty=uncertainty,
                        seed_flip_rate=flip,
                        seed_margins=seed_margins,
                    )
                )
        claim_accuracy[prompt_id].append(statistics.fmean(per_claim_pair_accuracy))

    cost_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt_id = (row.get("metadata") or {}).get(
            "prompt_id", row["prompt_condition"]
        )
        cost_rows[prompt_id].append(row)

    evaluations = []
    for prompt_id, values in sorted(claim_accuracy.items()):
        raw = cost_rows[prompt_id]
        valid = [
            row
            for row in raw
            if row.get("score_parse_ok") and row.get("score") is not None
        ]
        seeds = {int(row["seed"]) for row in raw}
        evaluations.append(
            CandidateEvaluation(
                prompt_id=prompt_id,
                stage=str((raw[0].get("metadata") or {}).get("stage", "screening")),
                n_claims=len(values),
                n_seeds=len(seeds),
                pairwise_accuracy=statistics.fmean(values),
                accuracy_se=accuracy_standard_error(values),
                score_variance=statistics.fmean(score_variances[prompt_id]),
                seed_flip_rate=statistics.fmean(pair_flips[prompt_id]),
                brier=None,
                mean_tokens=statistics.fmean(
                    float(row.get("generated_tokens", 0)) for row in raw
                ),
                mean_seconds=statistics.fmean(
                    float(row.get("generation_seconds", 0)) for row in raw
                ),
                invalid_rate=1.0 - len(valid) / len(raw),
                per_claim_accuracy=tuple(values),
            )
        )
    return evaluations, dict(failures), pair_details
