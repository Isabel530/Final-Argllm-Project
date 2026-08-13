#!/usr/bin/env python3
"""Validate E11 plumbing on excluded topics without inspecting formal topics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo_experiment import read_jsonl  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.tabular import write_json  # noqa: E402
from argstrength.webis_external import load_webis_generation_dataset  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = load_yaml(ROOT / "configs/e11_webis_external.yaml")
    stage = config["stages"]["pilot"]
    raw_path = ROOT / stage["output"]
    rows = read_jsonl(raw_path)
    items = load_webis_generation_dataset(ROOT / stage["dataset"])
    expected = sum(len(item.arguments) for item in items) * 2 * len(stage["seeds"])
    valid = sum(row.get("score_parse_ok") and row.get("score") is not None for row in rows)
    exact = sum(bool(row.get("format_ok")) for row in rows)
    observed_topics = sorted(
        {
            int(((row.get("metadata") or {}).get("dataset_item_metadata") or {})["topic_id"])
            for row in rows
        }
    )
    with (ROOT / config["preparation"]["manifest"]).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_topics = manifest["selection"]["pilot_topics"]
    result = {
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": _sha256(raw_path),
        "expected_rows": expected,
        "observed_rows": len(rows),
        "valid_score_rate": valid / len(rows) if rows else 0.0,
        "exact_format_rate": exact / len(rows) if rows else 0.0,
        "observed_topics": observed_topics,
        "expected_topics": expected_topics,
        "pass": (
            len(rows) == expected
            and valid == len(rows)
            and exact == len(rows)
            and observed_topics == expected_topics
        ),
        "decision_rule": (
            "Formal E11 may start only if all expected rows are present, every hard-decoded "
            "score is valid/exact, and only the pre-excluded pilot topics were used."
        ),
    }
    output = ROOT / "results/uq_apo/processed/e11_webis_pilot/pilot_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("E11 pilot validation failed; do not submit formal E11")


if __name__ == "__main__":
    main()
