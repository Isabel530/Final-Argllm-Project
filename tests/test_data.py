import json

import pytest

from argstrength.data import DataValidationError, load_main_dataset
from argstrength.schemas import Stance


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_effective_support_is_the_operative_stance(tmp_path):
    path = tmp_path / "data.jsonl"
    _write(
        path,
        [
            {
                "statement": "Claim",
                "is_support": None,
                "is_effective_support": True,
                "num_arguments": 2,
                "ground_truth_ranking": ["Strong", "Weak"],
            }
        ],
    )
    item = load_main_dataset(path, expected_arguments=2)[0]
    assert item.stance is Stance.SUPPORT
    assert item.source_is_support is None


def test_missing_effective_support_fails(tmp_path):
    path = tmp_path / "data.jsonl"
    _write(
        path,
        [{"statement": "Claim", "ground_truth_ranking": ["A", "B"]}],
    )
    with pytest.raises(DataValidationError, match="is_effective_support"):
        load_main_dataset(path, expected_arguments=2)
