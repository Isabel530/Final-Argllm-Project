import json

import pytest

from argstrength.data import DataValidationError
from argstrength.external_data import load_external_rank_dataset


def _row():
    return {
        "external_claim_id": 0,
        "group_id": "ibm-example",
        "source_group_id": "ibm-source",
        "source_topic": "Example topic",
        "statement": "Example topic",
        "is_effective_support": True,
        "panel": "broad",
        "ground_truth_ranking": ["a", "b", "c", "d"],
        "quality_wa": [0.9, 0.7, 0.5, 0.3],
    }


def test_external_loader_preserves_cluster_metadata(tmp_path):
    path = tmp_path / "external.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    item = load_external_rank_dataset(path)[0]
    assert item.claim_id == 0
    assert item.metadata["source_topic"] == "Example topic"
    assert item.metadata["panel"] == "broad"
    assert item.reference_ranks == (1, 2, 3, 4)


def test_external_loader_rejects_non_strict_quality(tmp_path):
    row = _row()
    row["quality_wa"] = [0.9, 0.7, 0.7, 0.3]
    path = tmp_path / "external.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="strictly descending"):
        load_external_rank_dataset(path)
