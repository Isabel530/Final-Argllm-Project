"""Controlled prompt conditions for pointwise argument-strength scoring.

All conditions receive identical task information and output scales.  The
2x2 design independently varies persona conditioning and rubric guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import PromptCondition, Stance


SCORE_PREFIX = "Score:"
ALLOWED_SCORE_OUTPUTS = tuple(f"{SCORE_PREFIX} {score}" for score in range(101))

_TASK = """Evaluate the strength of the argument in performing its stated role with respect to the claim.

Claim: {claim}
Role: The argument is intended to {relation} the claim.
Argument: {argument}"""

_SCALE = """Assign an integer score from 0 to 100, where:
0 means the argument provides no valid or effective reason for its stated role;
100 means the argument provides an exceptionally strong and effective reason for its stated role.

Respond with exactly:
Score: <integer>"""

_ROLE = (
    "You are an impartial expert analyst in argumentation. Assess arguments "
    "consistently and do not favour supporting or attacking arguments."
)

_RUBRIC = """Assess the argument using the following criteria:
1. Factuality: Are its factual statements accurate and credible?
2. Relevance: Does it directly address the claim?
3. Logical quality: Does its reasoning justify its intended conclusion?
4. Effectiveness: How strongly does it perform its stated supporting or attacking role?

{relation_rule}

Give one holistic score. Do not output separate criterion scores."""


@dataclass(frozen=True)
class PromptSpec:
    condition: PromptCondition
    text: str
    output_prefix: str = SCORE_PREFIX
    allowed_outputs: tuple[str, ...] = ALLOWED_SCORE_OUTPUTS


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _rubric_for(stance: Stance) -> str:
    relation_rule = (
        "For a supporting argument, a strong argument should provide valid "
        "reasons that the claim is true or credible."
        if stance is Stance.SUPPORT
        else
        "For an attacking argument, a strong argument should provide valid "
        "reasons that the claim is false, doubtful, or less credible."
    )
    return _RUBRIC.format(relation_rule=relation_rule)


def build_prompt(
    condition: str | PromptCondition,
    *,
    claim: str,
    argument: str,
    stance: str | Stance,
) -> PromptSpec:
    """Build one of four controlled prompt conditions."""

    condition = PromptCondition.normalise(condition)
    claim = _require_text("claim", claim)
    argument = _require_text("argument", argument)
    try:
        stance = stance if isinstance(stance, Stance) else Stance(stance)
    except ValueError as exc:
        raise ValueError("stance must be 'support' or 'attack'") from exc

    relation = "support" if stance is Stance.SUPPORT else "attack"
    task = _TASK.format(claim=claim, relation=relation, argument=argument)

    if condition is PromptCondition.DIRECT:
        blocks = (task, _SCALE)
    elif condition is PromptCondition.ROLE:
        blocks = (_ROLE, task, _SCALE)
    elif condition is PromptCondition.RUBRIC:
        blocks = (task, _rubric_for(stance), _SCALE)
    else:
        blocks = (_ROLE, task, _rubric_for(stance), _SCALE)

    return PromptSpec(condition=condition, text="\n\n".join(blocks))
