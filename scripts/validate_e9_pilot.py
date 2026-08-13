#!/usr/bin/env python3
"""Gate the formal E9 run on format/completeness checks, never on accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import read_jsonl  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.tabular import write_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e9_external.yaml")
    parser.add_argument("--raw", type=Path, default=ROOT / "results/uq_apo/raw/e9_pilot.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/uq_apo/processed/e9_pilot/pilot_validation.json",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    with (ROOT / config["preparation"]["manifest"]).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = read_jsonl(args.raw)
    pilot_stage = config["stages"]["pilot"]
    prompt_ids = {
        config["prompts"]["manual"]["direct"],
        config["prompts"]["winner"],
    }
    pilot_topics = set(manifest["selection"]["pilot_topics"])
    expected_rows = manifest["pilot"]["records"] * 4 * len(prompt_ids) * len(
        pilot_stage["seeds"]
    )
    failures = []
    if len(rows) != expected_rows:
        failures.append(f"row_count={len(rows)} expected={expected_rows}")
    observed_prompts = {
        (row.get("metadata") or {}).get("prompt_id", row.get("prompt_condition"))
        for row in rows
    }
    if observed_prompts != prompt_ids:
        failures.append(f"prompt_ids={sorted(observed_prompts)}")
    observed_seeds = {int(row["seed"]) for row in rows}
    if observed_seeds != set(pilot_stage["seeds"]):
        failures.append(f"seeds={sorted(observed_seeds)}")
    if any(not row.get("score_parse_ok") for row in rows):
        failures.append("one or more scores failed parsing")
    if any(not row.get("format_ok") for row in rows):
        failures.append("one or more outputs failed exact hard format")
    if any(row.get("score") is None or not 0 <= int(row["score"]) <= 100 for row in rows):
        failures.append("one or more scores are outside 0--100")
    if any((row.get("metadata") or {}).get("stage") != "external_pilot" for row in rows):
        failures.append("stage metadata mismatch")
    item_metadata = [
        (row.get("metadata") or {}).get("dataset_item_metadata") or {} for row in rows
    ]
    if any(not metadata.get("pilot") for metadata in item_metadata):
        failures.append("non-pilot item found")
    observed_topics = {metadata.get("source_topic") for metadata in item_metadata}
    if observed_topics != pilot_topics:
        failures.append(f"pilot_topics={sorted(observed_topics)}")

    report = {
        "status": "pass" if not failures else "fail",
        "gate_uses_accuracy": False,
        "raw": str(args.raw),
        "raw_sha256": _sha256(args.raw),
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "prompt_ids": sorted(observed_prompts),
        "seeds": sorted(observed_seeds),
        "pilot_topics": sorted(observed_topics),
        "parse_rate": statistics.fmean(bool(row.get("score_parse_ok")) for row in rows)
        if rows
        else 0.0,
        "exact_format_rate": statistics.fmean(bool(row.get("format_ok")) for row in rows)
        if rows
        else 0.0,
        "failures": failures,
        "decision_rule": (
            "Formal E9 may start only if completeness, identity, parse, and exact-format "
            "checks pass. Pilot accuracy is deliberately not calculated."
        ),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
