"""Configuration-driven, resumable experiment runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable
from uuid import uuid4

from .data import load_main_dataset, load_representation_dataset
from .generation import GenerationSettings, HuggingFaceScoreGenerator
from .parsing import parse_score
from .prompts import build_prompt
from .schemas import GenerationRecord


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML from requirements.txt") from exc
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _stable_generation_seed(seed: int, claim_id: int, argument_id: str) -> int:
    payload = f"{seed}|{claim_id}|{argument_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _sample(
    items: list[Any],
    sample_size: int | None,
    sample_seed: int,
    sample_by_stance: dict[str, int] | None = None,
) -> list[Any]:
    """Draw a reproducible random sample, optionally balanced by stance."""

    if sample_by_stance is not None:
        if set(sample_by_stance) != {"support", "attack"}:
            raise ValueError("sample_by_stance must contain support and attack")
        if any(type(count) is not int or count <= 0 for count in sample_by_stance.values()):
            raise ValueError("sample_by_stance counts must be positive integers")
        requested_total = sum(sample_by_stance.values())
        if sample_size is not None and sample_size != requested_total:
            raise ValueError("sample_size must equal the sum of sample_by_stance")

        grouped: dict[str, list[Any]] = {"support": [], "attack": []}
        for item in items:
            stance = getattr(item.stance, "value", item.stance)
            if stance not in grouped:
                raise ValueError(f"Unsupported stance in sampling: {stance!r}")
            grouped[stance].append(item)

        rng = random.Random(sample_seed)
        selected: list[Any] = []
        for stance in ("support", "attack"):
            count = sample_by_stance[stance]
            if count > len(grouped[stance]):
                raise ValueError(
                    f"Requested {count} {stance} items, only {len(grouped[stance])} available"
                )
            selected.extend(rng.sample(grouped[stance], count))
        rng.shuffle(selected)
        return selected

    if sample_size is None:
        return items
    if sample_size <= 0 or sample_size > len(items):
        raise ValueError(f"sample_size must be between 1 and {len(items)}")
    return random.Random(sample_seed).sample(items, sample_size)


def _record_thinking(row: dict[str, Any]) -> bool:
    return bool(row.get("thinking", row.get("metadata", {}).get("thinking", False)))


def _completed_keys(path: Path) -> set[tuple[Any, ...]]:
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
                    row["study"], row["model_key"], row["prompt_condition"],
                    row["decoding"], _record_thinking(row), row["seed"],
                    row["claim_id"], row["argument_id"],
                )
            )
    return keys


def _thinking_modes(config: dict[str, Any]) -> list[bool]:
    value = config.get("generation", {}).get("thinking", False)
    modes = [value] if type(value) is bool else value
    if not isinstance(modes, list) or not modes or any(type(x) is not bool for x in modes):
        raise ValueError("generation.thinking must be a boolean or non-empty list of booleans")
    return list(dict.fromkeys(modes))


def _setting(config: dict[str, Any], decoding: str, thinking: bool) -> GenerationSettings:
    generation = dict(config.get("generation", {}))
    generation.pop("thinking", None)
    by_mode = generation.pop("max_new_tokens_by_mode", {})
    if by_mode:
        if not isinstance(by_mode, dict):
            raise ValueError("max_new_tokens_by_mode must be a mapping")
        key = "think" if thinking else "no_think"
        if key in by_mode:
            generation["max_new_tokens"] = by_mode[key]
    generation["decoding"] = decoding
    generation["thinking"] = thinking
    return GenerationSettings(**generation)


def _iter_main_units(items: Iterable[Any]):
    for item in items:
        for index, (argument, rank) in enumerate(
            zip(item.arguments, item.reference_ranks), start=1
        ):
            yield item.claim_id, item.claim, item.stance, f"rank_{index}", argument, rank, {}


def _iter_representation_units(items: Iterable[Any]):
    for group in items:
        for variant_name, argument in group.variants:
            yield (
                group.claim_id, group.claim, group.stance,
                f"{group.group_id}:{variant_name}", argument, None,
                {"group_id": group.group_id, "representation": variant_name},
            )


def run_experiment(
    *,
    models_path: str | Path,
    experiments_path: str | Path,
    study_name: str,
    output_path: str | Path | None = None,
    model_keys: list[str] | None = None,
) -> Path:
    """Run and checkpoint one configured study."""

    model_document = load_yaml(models_path)
    experiment_document = load_yaml(experiments_path)
    models = model_document.get("models", {})
    studies = experiment_document.get("studies", {})
    if study_name not in studies:
        raise KeyError(f"Unknown study {study_name!r}")
    config = studies[study_name]

    selected_models = model_keys or list(config["models"])
    unknown = set(selected_models) - set(models)
    if unknown:
        raise KeyError(f"Unknown model keys: {sorted(unknown)}")

    project_root = Path(experiments_path).resolve().parent.parent
    dataset_path = project_root / config["dataset"]
    dataset_kind = config.get("dataset_kind", "main")
    if dataset_kind == "main":
        items = load_main_dataset(dataset_path)
        units_factory = _iter_main_units
    elif dataset_kind == "representation":
        items = load_representation_dataset(dataset_path)
        units_factory = _iter_representation_units
    else:
        raise ValueError("dataset_kind must be 'main' or 'representation'")
    items = _sample(
        items,
        config.get("sample_size"),
        config.get("sample_seed", 42),
        config.get("sample_by_stance"),
    )

    output = Path(output_path) if output_path else project_root / "results" / "raw" / f"{study_name}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(output)
    run_id = uuid4().hex
    requested_thinking = _thinking_modes(config)

    with output.open("a", encoding="utf-8") as handle:
        for model_key in selected_models:
            model_config = dict(models[model_key])
            model_name = model_config.pop("name")
            supported_thinking = model_config.pop("thinking_modes", [False])
            active_thinking = [mode for mode in requested_thinking if mode in supported_thinking]
            skipped = [mode for mode in requested_thinking if mode not in supported_thinking]
            if skipped:
                print(f"[mode] {model_key} skips unsupported thinking modes: {skipped}", flush=True)
            if not active_thinking:
                raise ValueError(f"{model_key} supports none of requested modes {requested_thinking}")

            print(f"[model] loading {model_key}: {model_name}", flush=True)
            generator = HuggingFaceScoreGenerator(model_name, **model_config)
            try:
                for decoding in config["decoding"]:
                    for thinking in active_thinking:
                        settings = _setting(config, decoding, thinking)
                        for condition in config["prompts"]:
                            for seed in config["seeds"]:
                                for (
                                    claim_id, claim, stance, argument_id, argument,
                                    reference_rank, metadata,
                                ) in units_factory(items):
                                    spec = build_prompt(
                                        condition, claim=claim, argument=argument, stance=stance
                                    )
                                    key = (
                                        study_name, model_key, spec.condition.value, decoding,
                                        settings.thinking, seed, claim_id, argument_id,
                                    )
                                    if key in completed:
                                        continue
                                    effective_seed = _stable_generation_seed(seed, claim_id, argument_id)
                                    generated = generator.generate(
                                        spec, seed=effective_seed, settings=settings
                                    )
                                    format_result = parse_score(generated.raw_output, strict=True)
                                    score_result = parse_score(generated.raw_output, strict=False)
                                    record = GenerationRecord(
                                        run_id=run_id,
                                        study=study_name,
                                        model_key=model_key,
                                        model_name=model_name,
                                        prompt_condition=spec.condition.value,
                                        decoding=decoding,
                                        thinking=settings.thinking,
                                        seed=seed,
                                        claim_id=claim_id,
                                        claim=claim,
                                        stance=stance.value,
                                        argument_id=argument_id,
                                        argument=argument,
                                        reference_rank=reference_rank,
                                        raw_output=generated.raw_output,
                                        score=score_result.score,
                                        score_parse_ok=score_result.ok,
                                        score_parse_error=score_result.error,
                                        format_ok=format_result.ok,
                                        format_error=format_result.error,
                                        generated_tokens=generated.generated_tokens,
                                        hit_max_new_tokens=generated.hit_max_new_tokens,
                                        generation_seconds=generated.elapsed_seconds,
                                        metadata={
                                            **metadata,
                                            "effective_generation_seed": effective_seed,
                                            "max_new_tokens_setting": settings.max_new_tokens,
                                            "max_new_tokens_limit": generated.max_new_tokens_limit,
                                        },
                                    )
                                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                                    handle.flush()
                                    completed.add(key)
            finally:
                generator.close()
    return output
