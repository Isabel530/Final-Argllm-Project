#!/usr/bin/env python3
"""Build locked broad and near-tie E9 panels from IBM-Rank-30k."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import read_candidate_registry, write_candidate_registry  # noqa: E402
from argstrength.external_data import load_external_rank_dataset  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402


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


def _read_source(path: Path, minimum_confidence: float) -> list[dict[str, Any]]:
    required = {"argument", "topic", "set", "WA", "MACE-P", "stance_WA", "stance_WA_conf"}
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing source columns: {sorted(missing)}")
        for source_row, raw in enumerate(reader, start=2):
            stance = int(raw["stance_WA"])
            confidence = float(raw["stance_WA_conf"])
            argument = raw["argument"].strip()
            topic = raw["topic"].strip()
            if raw["set"] != "test" or stance not in {-1, 1}:
                continue
            if confidence < minimum_confidence or not argument or not topic:
                continue
            rows.append(
                {
                    "source_row": source_row,
                    "argument": argument,
                    "topic": topic,
                    "wa": float(raw["WA"]),
                    "mace_p": float(raw["MACE-P"]),
                    "stance": stance,
                    "stance_confidence": confidence,
                }
            )
    return rows


def _quality_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one deterministic argument per distinct WA value."""

    by_quality: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_quality[row["wa"]].append(row)
    selected = []
    for quality, values in by_quality.items():
        selected.append(
            sorted(
                values,
                key=lambda row: (
                    -row["mace_p"],
                    -row["stance_confidence"],
                    row["argument"],
                    row["source_row"],
                ),
            )[0]
        )
    return sorted(selected, key=lambda row: (-quality_key(row), row["argument"]))


def quality_key(row: dict[str, Any]) -> float:
    return float(row["wa"])


def _broad_panel(
    unique_rows: list[dict[str, Any]], quantiles: list[float]
) -> list[dict[str, Any]]:
    indices = [int(math.floor(value * (len(unique_rows) - 1) + 0.5)) for value in quantiles]
    if len(set(indices)) != 4:
        raise ValueError("broad quantiles do not select four unique quality values")
    return [unique_rows[index] for index in indices]


def _near_tie_panel(
    unique_rows: list[dict[str, Any]],
    excluded_source_rows: set[int],
    central_fraction: tuple[float, float],
) -> list[dict[str, Any]]:
    remaining = [row for row in unique_rows if row["source_row"] not in excluded_source_rows]
    if len(remaining) < 4:
        raise ValueError("too few distinct quality values for near-tie panel")
    low, high = central_fraction
    windows = []
    denominator = max(1, len(remaining) - 1)
    for start in range(len(remaining) - 3):
        window = remaining[start : start + 4]
        midpoint_fraction = (start + 1.5) / denominator
        if low <= midpoint_fraction <= high:
            span = window[0]["wa"] - window[-1]["wa"]
            windows.append(
                (
                    span,
                    abs(midpoint_fraction - 0.5),
                    tuple(row["source_row"] for row in window),
                    window,
                )
            )
    if not windows:
        raise ValueError("no central four-value window available")
    return min(windows, key=lambda value: value[:3])[3]


def _record(
    *,
    claim_id: int,
    topic: str,
    stance: int,
    panel: str,
    rows: list[dict[str, Any]],
    pilot: bool,
) -> dict[str, Any]:
    namespace = f"{topic}\0{stance}"
    source_group_id = hashlib.sha256(namespace.encode()).hexdigest()[:16]
    group_id = hashlib.sha256(f"{namespace}\0{panel}".encode()).hexdigest()[:16]
    quality = [float(row["wa"]) for row in rows]
    if any(quality[index] <= quality[index + 1] for index in range(3)):
        raise ValueError("selected WA labels are not strictly descending")
    return {
        "external_claim_id": claim_id,
        "group_id": f"ibm-{group_id}",
        "source_group_id": f"ibm-{source_group_id}",
        "source_topic": topic,
        "statement": topic,
        "is_effective_support": stance == 1,
        "source_stance_wa": stance,
        "panel": panel,
        "pilot": pilot,
        "ground_truth_ranking": [row["argument"] for row in rows],
        "quality_wa": quality,
        "quality_mace_p": [float(row["mace_p"]) for row in rows],
        "stance_confidence": [float(row["stance_confidence"]) for row in rows],
        "source_rows": [int(row["source_row"]) for row in rows],
        "quality_gaps": [quality[index] - quality[index + 1] for index in range(3)],
    }


def _candidate_registries(config: dict[str, Any]) -> tuple[Path, Path]:
    manual = read_candidate_registry(ROOT / "results/uq_apo/candidates/iteration_0.jsonl")
    winner = read_candidate_registry(ROOT / "results/uq_apo/candidates/promotion_survivors.jsonl")
    expected = config["prompts"]
    by_id = {candidate.prompt_id: candidate for candidate in manual + winner}
    manual_ids = list(expected["manual"].values())
    all_ids = manual_ids + [expected["winner"]]
    missing = set(all_ids) - set(by_id)
    if missing:
        raise ValueError(f"prompt IDs missing from registries: {sorted(missing)}")
    formal_path = ROOT / "results/uq_apo/candidates/e9_five_prompts.jsonl"
    pilot_path = ROOT / "results/uq_apo/candidates/e9_pilot_candidates.jsonl"
    write_candidate_registry(formal_path, [by_id[prompt_id] for prompt_id in all_ids])
    write_candidate_registry(
        pilot_path,
        [by_id[expected["manual"]["direct"]], by_id[expected["winner"]]],
    )
    return pilot_path, formal_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e9_external.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    source_config = config["source"]
    preparation = config["preparation"]
    source = ROOT / source_config["csv"]
    observed_sha = _sha256(source)
    if observed_sha != source_config["sha256"]:
        raise ValueError(
            f"source SHA-256 mismatch: expected {source_config['sha256']}, got {observed_sha}"
        )
    source_rows = _read_source(
        source, float(preparation["minimum_stance_confidence"])
    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[(row["topic"], row["stance"])].append(row)
    topics = sorted({topic for topic, _ in grouped})
    namespace = preparation["pilot_hash_namespace"]
    topic_hashes = sorted(
        topics,
        key=lambda topic: hashlib.sha256(f"{namespace}\0{topic}".encode()).hexdigest(),
    )
    pilot_topics = set(topic_hashes[: int(preparation["pilot_topic_count"])])

    pilot_records, formal_records = [], []
    claim_id = 0
    for topic, stance in sorted(grouped):
        unique_rows = _quality_representatives(grouped[(topic, stance)])
        if len(unique_rows) < 8:
            raise ValueError(f"{topic!r}/{stance}: too few distinct WA labels")
        broad = _broad_panel(unique_rows, list(preparation["broad_quantiles"]))
        near_tie = _near_tie_panel(
            unique_rows,
            {row["source_row"] for row in broad},
            tuple(float(value) for value in preparation["near_tie_central_fraction"]),
        )
        pilot = topic in pilot_topics
        target = pilot_records if pilot else formal_records
        for panel, selected in (("broad", broad), ("near_tie", near_tie)):
            target.append(
                _record(
                    claim_id=claim_id,
                    topic=topic,
                    stance=stance,
                    panel=panel,
                    rows=selected,
                    pilot=pilot,
                )
            )
            claim_id += 1

    pilot_path = ROOT / preparation["output_pilot"]
    formal_path = ROOT / preparation["output_formal"]
    _write_jsonl(pilot_path, pilot_records)
    _write_jsonl(formal_path, formal_records)
    load_external_rank_dataset(pilot_path)
    load_external_rank_dataset(formal_path)
    pilot_registry, formal_registry = _candidate_registries(config)

    manifest_path = ROOT / preparation["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source": {
            "dataset": source_config["dataset"],
            "revision": source_config["revision"],
            "split": source_config["split"],
            "path": str(source.relative_to(ROOT)),
            "sha256": observed_sha,
            "rows": len(source_rows),
            "topics": len(topics),
        },
        "selection": {
            "ranking_label": preparation["ranking_label"],
            "minimum_stance_confidence": preparation["minimum_stance_confidence"],
            "pilot_hash_namespace": namespace,
            "pilot_topics": sorted(pilot_topics),
            "broad_quantiles": preparation["broad_quantiles"],
            "near_tie_central_fraction": preparation["near_tie_central_fraction"],
            "manual_selection": False,
        },
        "pilot": {
            "path": str(pilot_path.relative_to(ROOT)),
            "sha256": _sha256(pilot_path),
            "records": len(pilot_records),
            "topics": len({row["source_topic"] for row in pilot_records}),
            "excluded_from_formal": True,
            "candidate_registry": str(pilot_registry.relative_to(ROOT)),
            "candidate_registry_sha256": _sha256(pilot_registry),
        },
        "formal": {
            "path": str(formal_path.relative_to(ROOT)),
            "sha256": _sha256(formal_path),
            "records": len(formal_records),
            "topics": len({row["source_topic"] for row in formal_records}),
            "source_groups": len({row["source_group_id"] for row in formal_records}),
            "panels": {
                panel: sum(row["panel"] == panel for row in formal_records)
                for panel in ("broad", "near_tie")
            },
            "candidate_registry": str(formal_registry.relative_to(ROOT)),
            "candidate_registry_sha256": _sha256(formal_registry),
        },
        "quality_gap_summary": {
            panel: {
                "mean_adjacent_gap": statistics.fmean(
                    gap
                    for row in formal_records
                    if row["panel"] == panel
                    for gap in row["quality_gaps"]
                ),
                "mean_total_span": statistics.fmean(
                    row["quality_wa"][0] - row["quality_wa"][-1]
                    for row in formal_records
                    if row["panel"] == panel
                ),
            }
            for panel in ("broad", "near_tie")
        },
        "warning": (
            "Pilot topics are fully excluded from formal E9. Formal prompt selection is frozen; "
            "external results cannot trigger prompt rewriting or winner replacement."
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
