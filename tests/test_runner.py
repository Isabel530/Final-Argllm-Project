import json
from types import SimpleNamespace

import pytest

from argstrength.generation import GenerationSettings
from argstrength.runner import _completed_keys, _sample, _setting, _thinking_modes
from argstrength.schemas import Stance


def _row(thinking):
    return {
        "study": "s",
        "model_key": "m",
        "prompt_condition": "direct",
        "decoding": "soft",
        "thinking": thinking,
        "seed": 42,
        "claim_id": 1,
        "argument_id": "rank_1",
    }


def test_checkpoint_key_distinguishes_thinking_modes(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        "\n".join(json.dumps(_row(mode)) for mode in (False, True)) + "\n",
        encoding="utf-8",
    )
    keys = _completed_keys(path)
    assert len(keys) == 2
    assert {key[4] for key in keys} == {False, True}


def test_checkpoint_reads_legacy_metadata_thinking(tmp_path):
    row = _row(False)
    del row["thinking"]
    row["metadata"] = {"thinking": True}
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert next(iter(_completed_keys(path)))[4] is True


def test_multiple_thinking_modes_and_mode_specific_lengths():
    config = {
        "generation": {
            "thinking": [False, True],
            "max_new_tokens": 64,
            "max_new_tokens_by_mode": {"no_think": 64, "think": 512},
        }
    }
    assert _thinking_modes(config) == [False, True]
    assert _setting(config, "soft", False).max_new_tokens == 64
    assert _setting(config, "soft", True).max_new_tokens == 512


def test_context_window_is_a_valid_natural_stop_setting():
    settings = GenerationSettings(
        decoding="soft",
        max_new_tokens="context_window",
        thinking=True,
    )
    assert settings.max_new_tokens == "context_window"


@pytest.mark.parametrize("invalid", [[], [True, "false"], "both", None])
def test_invalid_thinking_modes_fail(invalid):
    with pytest.raises(ValueError):
        _thinking_modes({"generation": {"thinking": invalid}})


def test_stance_balanced_sampling_is_exact_and_reproducible():
    items = [
        SimpleNamespace(claim_id=index, stance=Stance.SUPPORT)
        for index in range(15)
    ] + [
        SimpleNamespace(claim_id=100 + index, stance=Stance.ATTACK)
        for index in range(15)
    ]
    first = _sample(
        items,
        sample_size=20,
        sample_seed=2026,
        sample_by_stance={"support": 10, "attack": 10},
    )
    second = _sample(
        items,
        sample_size=20,
        sample_seed=2026,
        sample_by_stance={"support": 10, "attack": 10},
    )
    assert [item.claim_id for item in first] == [item.claim_id for item in second]
    assert sum(item.stance is Stance.SUPPORT for item in first) == 10
    assert sum(item.stance is Stance.ATTACK for item in first) == 10


def test_stance_sample_total_must_match_sample_size():
    with pytest.raises(ValueError, match="sample_size"):
        _sample(
            [],
            sample_size=20,
            sample_seed=2026,
            sample_by_stance={"support": 9, "attack": 10},
        )
