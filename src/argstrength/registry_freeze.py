"""Freeze a validation winner and verify post-selection candidate registries."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .apo import PromptCandidate
from .apo_experiment import read_candidate_registry, write_candidate_registry
from .tabular import write_json


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry_record(path: Path, candidates: Sequence[PromptCandidate]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "prompt_ids": [candidate.prompt_id for candidate in candidates],
    }


def freeze_validation_winner(
    *,
    candidate_registry: str | Path,
    selection_path: str | Path,
    baseline_registry: str | Path,
    winner_output: str | Path,
    transfer_output: str | Path,
    manifest_output: str | Path,
) -> dict[str, Any]:
    """Write one winner plus Direct/Rubric/winner transfer registries."""

    candidate_registry = Path(candidate_registry)
    selection_path = Path(selection_path)
    baseline_registry = Path(baseline_registry)
    winner_output = Path(winner_output)
    transfer_output = Path(transfer_output)
    manifest_output = Path(manifest_output)
    with selection_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    selected_ids = selection.get("prompt_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) != 1:
        raise ValueError("validation selection must contain exactly one prompt_id")
    winner_id = selected_ids[0]
    candidates = read_candidate_registry(candidate_registry)
    winners = [candidate for candidate in candidates if candidate.prompt_id == winner_id]
    if len(winners) != 1:
        raise ValueError(f"winner {winner_id!r} is absent from candidate registry")
    winner = winners[0]
    baselines = read_candidate_registry(baseline_registry)
    baseline_by_label = {candidate.label: candidate for candidate in baselines}
    missing = {"direct", "rubric"} - set(baseline_by_label)
    if missing:
        raise ValueError(f"baseline registry lacks labels: {sorted(missing)}")
    transfer_candidates = [baseline_by_label["direct"], baseline_by_label["rubric"]]
    if winner.prompt_id not in {candidate.prompt_id for candidate in transfer_candidates}:
        transfer_candidates.append(winner)
    write_candidate_registry(winner_output, [winner])
    write_candidate_registry(transfer_output, transfer_candidates)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_after_validation": True,
        "winner_prompt_id": winner.prompt_id,
        "winner_instruction": winner.instruction,
        "selection_rule": selection.get("rule"),
        "uq_signal": selection.get("uq_signal"),
        "inputs": {
            "candidate_registry": {"path": str(candidate_registry.resolve()), "sha256": file_sha256(candidate_registry)},
            "selection": {"path": str(selection_path.resolve()), "sha256": file_sha256(selection_path)},
            "baseline_registry": {"path": str(baseline_registry.resolve()), "sha256": file_sha256(baseline_registry)},
        },
        "registries": {
            "qwen_internal_test": _registry_record(winner_output, [winner]),
            "cross_model_transfer": _registry_record(transfer_output, transfer_candidates),
        },
        "note": (
            "No prompt rewriting or reselection is permitted after this manifest. "
            "If the winner is Direct or Rubric, the transfer registry contains two "
            "unique prompts rather than duplicating the winner."
        ),
    }
    write_json(manifest_output, manifest)
    return manifest


def verify_frozen_registry(
    registry_path: str | Path,
    candidates: Sequence[PromptCandidate],
    manifest_path: str | Path,
) -> str:
    """Return the registered purpose or reject an altered/unfrozen registry."""

    registry_path = Path(registry_path)
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("locked_after_validation") is not True:
        raise ValueError("freeze manifest is not locked_after_validation")
    actual_hash = file_sha256(registry_path)
    actual_ids = [candidate.prompt_id for candidate in candidates]
    for purpose, record in manifest.get("registries", {}).items():
        if record.get("sha256") == actual_hash:
            if record.get("prompt_ids") != actual_ids:
                raise ValueError("frozen registry prompt IDs do not match manifest")
            return purpose
    raise ValueError("candidate registry hash is not authorised by freeze manifest")
