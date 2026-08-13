#!/usr/bin/env python3
"""Summarise a natural-stop output-length pilot from its raw JSONL."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


def _nearest_rank(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def summarise(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("No records found")

    token_counts = [int(row["generated_tokens"]) for row in rows]
    truncations = sum(bool(row["hit_max_new_tokens"]) for row in rows)
    claim_stances: dict[Any, str] = {}
    for row in rows:
        claim_id = row["claim_id"]
        stance = row["stance"]
        previous = claim_stances.setdefault(claim_id, stance)
        if previous != stance:
            raise ValueError(f"Claim {claim_id!r} has inconsistent stances")

    maximum = max(token_counts)
    score_parse_count = sum(bool(row["score_parse_ok"]) for row in rows)
    format_count = sum(bool(row["format_ok"]) for row in rows)
    return {
        "n_generations": len(rows),
        "n_claims": len(claim_stances),
        "claims_by_stance": dict(sorted(Counter(claim_stances.values()).items())),
        "generations_by_prompt": dict(
            sorted(Counter(row["prompt_condition"] for row in rows).items())
        ),
        "total_generated_tokens": sum(token_counts),
        "total_raw_output_characters": sum(len(row["raw_output"]) for row in rows),
        "min_generated_tokens": min(token_counts),
        "median_generated_tokens": statistics.median(token_counts),
        "p90_generated_tokens": _nearest_rank(token_counts, 0.90),
        "p95_generated_tokens": _nearest_rank(token_counts, 0.95),
        "p99_generated_tokens": _nearest_rank(token_counts, 0.99),
        "max_generated_tokens": maximum,
        "n_hit_generation_limit": truncations,
        "truncation_rate": truncations / len(rows),
        "n_score_parse_ok": score_parse_count,
        "score_parse_rate": score_parse_count / len(rows),
        "n_format_ok": format_count,
        "format_rate": format_count / len(rows),
        "rounded_max_power_of_two": (
            _next_power_of_two(maximum) if truncations == 0 else None
        ),
    }


def grouped_summaries(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: summarise(values) for name, values in sorted(groups.items())}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def validate_natural_pilot(rows: list[dict[str, Any]]) -> None:
    expected_prompts = {"direct", "role", "rubric", "role_rubric"}
    if len(rows) != 160:
        raise ValueError(f"Expected 160 completed generations, found {len(rows)}")
    if {row["study"] for row in rows} != {"output_length_natural_pilot"}:
        raise ValueError("Input is not the natural-stop pilot")
    models = {row["model_key"] for row in rows}
    allowed_models = {"qwen3_8b", "gpt_oss_20b"}
    if len(models) != 1 or not models <= allowed_models:
        raise ValueError(f"Expected one supported model per file, found {models}")
    if {row["decoding"] for row in rows} != {"soft"}:
        raise ValueError("Expected soft decoding only")
    if {bool(row["thinking"]) for row in rows} != {True}:
        raise ValueError("Expected Think mode only")
    if {row["seed"] for row in rows} != {42}:
        raise ValueError("Expected seed 42 only")
    prompt_counts = Counter(row["prompt_condition"] for row in rows)
    if set(prompt_counts) != expected_prompts or set(prompt_counts.values()) != {40}:
        raise ValueError(f"Expected 40 generations per prompt, found {prompt_counts}")
    claim_stances = {row["claim_id"]: row["stance"] for row in rows}
    stance_counts = Counter(claim_stances.values())
    if stance_counts != {"support": 5, "attack": 5}:
        raise ValueError(f"Expected 5 support and 5 attack claims, found {stance_counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    validate_natural_pilot(rows)
    report = {
        "overall": summarise(rows),
        "by_stance": grouped_summaries(rows, "stance"),
        "by_prompt": grouped_summaries(rows, "prompt_condition"),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
