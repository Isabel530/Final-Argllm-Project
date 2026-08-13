from __future__ import annotations

from argstrength.optimizer_model import OptimizerOutputError, _extract_json_object


def test_extract_json_prefers_later_proposals_document():
    text = (
        'Reasoning example: {"diagnosis": "ambiguous"}\n'
        '```json\n'
        '{"proposals": [{"critique": "c", "instruction": "i"}]}\n'
        '```'
    )
    document = _extract_json_object(text)
    assert document["proposals"][0]["instruction"] == "i"


def test_extract_json_accepts_top_level_proposal_list():
    document = _extract_json_object(
        '[{"critique": "c", "instruction": "readable instruction"}]'
    )
    assert len(document["proposals"]) == 1


def test_optimizer_output_error_retains_all_raw_attempts():
    error = OptimizerOutputError("invalid", ["first", "second", "third"])
    assert str(error) == "invalid"
    assert error.raw_outputs == ("first", "second", "third")
