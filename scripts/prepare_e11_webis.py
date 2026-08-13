#!/usr/bin/env python3
"""Prepare and fingerprint the claim-disjoint Webis E11 pilot/formal panels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import (  # noqa: E402
    read_candidate_registry,
    write_candidate_registry,
)
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.webis_external import (  # noqa: E402
    load_webis_generation_dataset,
    load_webis_pair_labels,
)


DIMENSIONS = ("logical", "rhetorical", "dialectical")
PAIR_KEY = (
    "Discussion ID A",
    "Argument ID A",
    "Discussion ID B",
    "Argument ID B",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid Webis boolean {value!r}")


def _argument_key(row: dict[str, Any]) -> str:
    return (
        f"t{int(row['topic_id'])}:d{int(row['discussion_id'])}:"
        f"a{int(row['argument_id'])}"
    )


def _read_arguments(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[int, int], list[dict[str, Any]]]]:
    required = {
        "Topic ID",
        "Argument ID",
        "Discussion ID",
        "Premise",
        "Is Argument?",
        "Stance",
        "Combined Quality",
    }
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing Webis full columns: {sorted(missing)}")
        for source_row, raw in enumerate(reader, start=2):
            row = {
                "source_row": source_row,
                "topic_id": int(raw["Topic ID"]),
                "argument_id": int(raw["Argument ID"]),
                "discussion_id": int(raw["Discussion ID"]),
                "premise": raw["Premise"].strip(),
                "is_argument": _parse_bool(raw["Is Argument?"]),
                "stance": _parse_bool(raw["Stance"]),
                "combined_quality": float(raw["Combined Quality"]),
            }
            if not row["premise"]:
                continue
            rows.append(row)
            lookup[(row["discussion_id"], row["argument_id"])].append(row)
    return rows, lookup


def _read_topics(path: Path) -> dict[int, dict[str, str]]:
    output = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            topic_id = int(raw["Topic ID"])
            output[topic_id] = {
                "category": raw["Category"].strip(),
                "long_query": raw["Long Query"].strip(),
                "short_query": raw["Short Query"].strip(),
            }
    if len(output) != 20:
        raise ValueError(f"expected 20 Webis topics, found {len(output)}")
    return output


def _read_dimension(path: Path, dimension: str) -> dict[tuple[int, int, int, int], str]:
    output = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            key = tuple(int(raw[column]) for column in PAIR_KEY)
            label = raw["Comparison"]
            if label not in {"A", "B", "Tie"}:
                raise ValueError(f"invalid {dimension} comparison {label!r}")
            if key in output:
                raise ValueError(f"duplicate {dimension} pair {key}")
            output[key] = label
    return output


def _resolve_pair(
    key: tuple[int, int, int, int],
    lookup: dict[tuple[int, int], list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    discussion_a, argument_a, discussion_b, argument_b = key
    candidates_a = lookup.get((discussion_a, argument_a), [])
    candidates_b = lookup.get((discussion_b, argument_b), [])
    matches = [
        (left, right)
        for left in candidates_a
        for right in candidates_b
        if left["topic_id"] == right["topic_id"]
    ]
    if len(matches) != 1:
        raise ValueError(f"pair {key} has {len(matches)} same-topic resolutions")
    return matches[0]


def _majority(labels: list[str]) -> tuple[str | None, int]:
    label, count = Counter(labels).most_common(1)[0]
    return (label if count >= 2 else None), count


def _candidate_registries(config: dict[str, Any]) -> tuple[Path, Path]:
    initial = read_candidate_registry(ROOT / "results/uq_apo/candidates/iteration_0.jsonl")
    finalists = read_candidate_registry(
        ROOT / "results/uq_apo/candidates/promotion_survivors.jsonl"
    )
    by_id = {candidate.prompt_id: candidate for candidate in initial + finalists}
    direct = config["prompts"]["manual"]["direct"]
    rubric = config["prompts"]["manual"]["rubric"]
    winner = config["prompts"]["winner"]
    missing = {direct, rubric, winner} - set(by_id)
    if missing:
        raise ValueError(f"E11 prompt IDs missing from registries: {sorted(missing)}")
    pilot_path = ROOT / "results/uq_apo/candidates/e11_pilot_candidates.jsonl"
    formal_path = ROOT / "results/uq_apo/candidates/e11_three_prompts.jsonl"
    write_candidate_registry(pilot_path, [by_id[direct], by_id[winner]])
    write_candidate_registry(formal_path, [by_id[direct], by_id[rubric], by_id[winner]])
    return pilot_path, formal_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/e11_webis_external.yaml"
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    source_config = config["source"]
    source_paths = {
        name: ROOT / value["path"] for name, value in source_config["files"].items()
    }
    for name, path in source_paths.items():
        observed = _sha256(path)
        expected = source_config["files"][name]["sha256"]
        if observed != expected:
            raise ValueError(f"{name} SHA-256 mismatch: {observed} != {expected}")

    _, lookup = _read_arguments(source_paths["full"])
    topics = _read_topics(source_paths["topics"])
    dimensions = {
        name: _read_dimension(source_paths[name], name) for name in DIMENSIONS
    }
    key_sets = [set(values) for values in dimensions.values()]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Webis quality dimensions do not cover identical oriented pairs")

    preparation = config["preparation"]
    topic_order = sorted(
        topics,
        key=lambda topic_id: hashlib.sha256(
            f"{preparation['pilot_hash_namespace']}\0{topic_id}".encode()
        ).hexdigest(),
    )
    pilot_topics = set(topic_order[: int(preparation["pilot_topic_count"])])

    eligible: list[dict[str, Any]] = []
    exclusions = Counter()
    for key in sorted(key_sets[0]):
        left, right = _resolve_pair(key, lookup)
        if not (left["is_argument"] and right["is_argument"]):
            exclusions["not_both_arguments"] += 1
            continue
        if left["stance"] != right["stance"]:
            exclusions["cross_stance"] += 1
            continue
        labels = [dimensions[name][key] for name in DIMENSIONS]
        majority, agreement_count = _majority(labels)
        combined_label = (
            "A"
            if left["combined_quality"] > right["combined_quality"]
            else "B"
            if left["combined_quality"] < right["combined_quality"]
            else "Tie"
        )
        eligible.append(
            {
                "topic_id": left["topic_id"],
                "stance": left["stance"],
                "left": left,
                "right": right,
                "labels": dict(zip(DIMENSIONS, labels)),
                "majority_label": majority,
                "agreement_count": agreement_count,
                "combined_label": combined_label,
            }
        )

    grouped_pairs: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped_pairs[(row["topic_id"], row["stance"])].append(row)

    generation_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for claim_id, (topic_id, stance) in enumerate(sorted(grouped_pairs)):
        raw_pairs = grouped_pairs[(topic_id, stance)]
        arguments_by_key: dict[str, dict[str, Any]] = {}
        for pair in raw_pairs:
            for endpoint in (pair["left"], pair["right"]):
                arguments_by_key[_argument_key(endpoint)] = endpoint
        ordered_keys = sorted(arguments_by_key)
        index_by_key = {key: index for index, key in enumerate(ordered_keys, start=1)}
        group_id = f"webis-t{topic_id}-{'pro' if stance else 'con'}"
        topic = topics[topic_id]
        pilot = topic_id in pilot_topics
        generation_rows.append(
            {
                "external_claim_id": claim_id,
                "group_id": group_id,
                "topic_id": topic_id,
                "source_topic": topic["long_query"],
                "statement": topic["long_query"],
                "short_query": topic["short_query"],
                "category": topic["category"],
                "is_effective_support": stance,
                "pilot": pilot,
                "argument_keys": ordered_keys,
                "arguments": [arguments_by_key[key]["premise"] for key in ordered_keys],
                "combined_quality": [
                    arguments_by_key[key]["combined_quality"] for key in ordered_keys
                ],
            }
        )
        for pair in raw_pairs:
            left_key = _argument_key(pair["left"])
            right_key = _argument_key(pair["right"])
            payload = f"{group_id}\0{left_key}\0{right_key}"
            pair_rows.append(
                {
                    "pair_id": f"webis-{hashlib.sha256(payload.encode()).hexdigest()[:16]}",
                    "external_claim_id": claim_id,
                    "group_id": group_id,
                    "topic_id": topic_id,
                    "source_topic": topic["long_query"],
                    "stance": "support" if stance else "attack",
                    "pilot": pilot,
                    "argument_a_key": left_key,
                    "argument_b_key": right_key,
                    "argument_a_id": f"rank_{index_by_key[left_key]}",
                    "argument_b_id": f"rank_{index_by_key[right_key]}",
                    "argument_a": pair["left"]["premise"],
                    "argument_b": pair["right"]["premise"],
                    **pair["labels"],
                    "majority_label": pair["majority_label"],
                    "agreement_count": pair["agreement_count"],
                    "unanimous": pair["agreement_count"] == 3,
                    "combined_label": pair["combined_label"],
                    "combined_quality_a": pair["left"]["combined_quality"],
                    "combined_quality_b": pair["right"]["combined_quality"],
                    "combined_quality_gap_a_minus_b": (
                        pair["left"]["combined_quality"]
                        - pair["right"]["combined_quality"]
                    ),
                }
            )

    pilot_generation = [row for row in generation_rows if row["pilot"]]
    formal_generation = [row for row in generation_rows if not row["pilot"]]
    pilot_pairs = [row for row in pair_rows if row["pilot"]]
    formal_pairs = [row for row in pair_rows if not row["pilot"]]
    paths = {
        "generation_pilot": ROOT / preparation["generation_pilot"],
        "generation_formal": ROOT / preparation["generation_formal"],
        "pairs_pilot": ROOT / preparation["pairs_pilot"],
        "pairs_formal": ROOT / preparation["pairs_formal"],
    }
    _write_jsonl(paths["generation_pilot"], pilot_generation)
    _write_jsonl(paths["generation_formal"], formal_generation)
    _write_jsonl(paths["pairs_pilot"], pilot_pairs)
    _write_jsonl(paths["pairs_formal"], formal_pairs)
    load_webis_generation_dataset(paths["generation_pilot"])
    load_webis_generation_dataset(paths["generation_formal"])
    load_webis_pair_labels(paths["pairs_pilot"])
    load_webis_pair_labels(paths["pairs_formal"])
    pilot_registry, formal_registry = _candidate_registries(config)

    def panel_summary(groups: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "topics": len({row["topic_id"] for row in groups}),
            "topic_stance_groups": len(groups),
            "unique_argument_generations_per_prompt_seed": sum(
                len(row["arguments"]) for row in groups
            ),
            "eligible_dimension_pairs": len(pairs),
            "majority_pairs": sum(row["majority_label"] is not None for row in pairs),
            "majority_decisive_pairs": sum(
                row["majority_label"] in {"A", "B"} for row in pairs
            ),
            "majority_ties": sum(row["majority_label"] == "Tie" for row in pairs),
            "unanimous_pairs": sum(row["unanimous"] for row in pairs),
        }

    manifest_path = ROOT / preparation["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "locked_before_generation",
        "source": {
            "dataset": source_config["dataset"],
            "version": source_config["version"],
            "doi": source_config["doi"],
            "license": source_config["license"],
            "files": {
                name: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                }
                for name, path in source_paths.items()
            },
        },
        "selection": {
            "require_is_argument": True,
            "require_same_topic": True,
            "require_same_stance": True,
            "primary_gold": preparation["primary_gold"],
            "primary_scope": preparation["primary_scope"],
            "pilot_hash_namespace": preparation["pilot_hash_namespace"],
            "pilot_topics": sorted(pilot_topics),
            "manual_selection": False,
            "excluded_source_pairs": dict(exclusions),
        },
        "pilot": {
            **panel_summary(pilot_generation, pilot_pairs),
            "generation_path": str(paths["generation_pilot"].relative_to(ROOT)),
            "generation_sha256": _sha256(paths["generation_pilot"]),
            "pairs_path": str(paths["pairs_pilot"].relative_to(ROOT)),
            "pairs_sha256": _sha256(paths["pairs_pilot"]),
            "candidate_registry": str(pilot_registry.relative_to(ROOT)),
            "candidate_registry_sha256": _sha256(pilot_registry),
            "excluded_from_formal": True,
        },
        "formal": {
            **panel_summary(formal_generation, formal_pairs),
            "generation_path": str(paths["generation_formal"].relative_to(ROOT)),
            "generation_sha256": _sha256(paths["generation_formal"]),
            "pairs_path": str(paths["pairs_formal"].relative_to(ROOT)),
            "pairs_sha256": _sha256(paths["pairs_formal"]),
            "candidate_registry": str(formal_registry.relative_to(ROOT)),
            "candidate_registry_sha256": _sha256(formal_registry),
        },
        "analysis_lock": config["analysis"],
        "interpretation_lock": config["interpretation_lock"],
        "warning": (
            "E11 uses direct human labels only for same-topic, same-stance, valid-argument "
            "pairs. Pilot topics are excluded from formal analysis. E11 cannot trigger "
            "prompt rewriting or replacement of the frozen winner."
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
