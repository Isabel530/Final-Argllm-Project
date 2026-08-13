import pytest

from argstrength.prompts import build_prompt


@pytest.mark.parametrize("condition", ["direct", "role", "rubric", "role_rubric"])
@pytest.mark.parametrize("stance", ["support", "attack"])
def test_all_conditions_receive_same_required_information(condition, stance):
    spec = build_prompt(
        condition,
        claim="The claim text",
        argument="The argument text",
        stance=stance,
    )
    assert "Claim: The claim text" in spec.text
    assert "Argument: The argument text" in spec.text
    assert f"intended to {stance}" in spec.text
    assert "0 to 100" in spec.text
    assert spec.allowed_outputs[0] == "Score: 0"
    assert spec.allowed_outputs[-1] == "Score: 100"


def test_conditions_have_interpretable_differences():
    kwargs = dict(claim="C", argument="A", stance="attack")
    direct = build_prompt("direct", **kwargs).text
    role = build_prompt("role", **kwargs).text
    rubric = build_prompt("rubric", **kwargs).text
    combined = build_prompt("role_rubric", **kwargs).text
    assert "impartial expert analyst" not in direct
    assert "impartial expert analyst" in role
    assert "Factuality:" not in role
    assert "Factuality:" not in direct
    assert "Factuality:" in rubric
    assert "impartial expert analyst" not in rubric
    assert "claim is false, doubtful, or less credible" in rubric
    assert "impartial expert analyst" in combined
    assert "Factuality:" in combined


def test_legacy_names_are_aliases():
    kwargs = dict(claim="C", argument="A", stance="support")
    assert build_prompt("chatgpt", **kwargs).condition.value == "direct"
    assert build_prompt("analyst", **kwargs).condition.value == "role"
    assert build_prompt("opro", **kwargs).condition.value == "rubric"


def test_stance_is_required_and_validated():
    with pytest.raises(ValueError, match="stance"):
        build_prompt("direct", claim="C", argument="A", stance="unknown")
