import json

import pytest

from argstrength.webis_external import (
    load_webis_generation_dataset,
    load_webis_pair_labels,
)


def test_load_variable_sized_generation_group(tmp_path):
    path = tmp_path / "groups.jsonl"
    row = {
        "external_claim_id": 7,
        "group_id": "webis-t1-pro",
        "statement": "Should X be allowed?",
        "is_effective_support": True,
        "arguments": ["A", "B", "C"],
        "argument_keys": ["a", "b", "c"],
        "combined_quality": [0.2, 0.8, -0.1],
        "topic_id": 1,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    items = load_webis_generation_dataset(path)
    assert len(items) == 1
    assert items[0].arguments == ("A", "B", "C")
    assert items[0].reference_ranks == (2, 1, 3)
    assert items[0].metadata["argument_keys"] == ["a", "b", "c"]


def test_pair_loader_rejects_bad_human_label(tmp_path):
    path = tmp_path / "pairs.jsonl"
    row = {
        "pair_id": "p1",
        "external_claim_id": 1,
        "argument_a_id": "rank_1",
        "argument_b_id": "rank_2",
        "logical": "A",
        "rhetorical": "B",
        "dialectical": "unknown",
        "majority_label": None,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="invalid dialectical"):
        load_webis_pair_labels(path)
