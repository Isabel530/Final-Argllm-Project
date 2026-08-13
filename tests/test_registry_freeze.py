from __future__ import annotations

import json

import pytest

from argstrength.apo import PromptCandidate
from argstrength.apo_experiment import write_candidate_registry
from argstrength.registry_freeze import freeze_validation_winner, verify_frozen_registry


def test_freeze_winner_and_transfer_registry(tmp_path):
    direct = PromptCandidate("", 0, "seed", label="direct")
    rubric = PromptCandidate("Use a clear rubric.", 0, "seed", label="rubric")
    winner = PromptCandidate("Check evidence carefully.", 1, "rewrite")
    baselines = tmp_path / "baselines.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    selection = tmp_path / "selection.json"
    write_candidate_registry(baselines, [direct, rubric])
    write_candidate_registry(candidates, [rubric, winner])
    selection.write_text(json.dumps({"prompt_ids": [winner.prompt_id], "uq_signal": "sampling_variance", "rule": "validation only"}), encoding="utf-8")
    winner_path = tmp_path / "winner.jsonl"
    transfer_path = tmp_path / "transfer.jsonl"
    manifest_path = tmp_path / "manifest.json"
    freeze_validation_winner(candidate_registry=candidates, selection_path=selection, baseline_registry=baselines, winner_output=winner_path, transfer_output=transfer_path, manifest_output=manifest_path)
    assert verify_frozen_registry(winner_path, [winner], manifest_path) == "qwen_internal_test"
    assert verify_frozen_registry(transfer_path, [direct, rubric, winner], manifest_path) == "cross_model_transfer"


def test_freeze_manifest_rejects_modified_registry(tmp_path):
    direct = PromptCandidate("", 0, "seed", label="direct")
    rubric = PromptCandidate("Use a clear rubric.", 0, "seed", label="rubric")
    baselines = tmp_path / "baselines.jsonl"
    selection = tmp_path / "selection.json"
    write_candidate_registry(baselines, [direct, rubric])
    selection.write_text(json.dumps({"prompt_ids": [rubric.prompt_id]}), encoding="utf-8")
    winner_path = tmp_path / "winner.jsonl"
    transfer_path = tmp_path / "transfer.jsonl"
    manifest_path = tmp_path / "manifest.json"
    freeze_validation_winner(candidate_registry=baselines, selection_path=selection, baseline_registry=baselines, winner_output=winner_path, transfer_output=transfer_path, manifest_output=manifest_path)
    winner_path.write_text(winner_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not authorised"):
        verify_frozen_registry(winner_path, [rubric], manifest_path)
