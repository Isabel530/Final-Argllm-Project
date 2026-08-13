"""Compatibility interface for the controlled argument-strength prompts.

New experiments should import ``argstrength.prompts.build_prompt`` directly.
This module keeps the former ``UncertaintyEvaluatorPrompts`` calling style so
small legacy utilities can be migrated gradually.

Canonical names:
    chatgpt -> direct
    analyst -> role
    opro    -> rubric (OPRO-derived, not a new OPRO optimisation run)
    role_rubric -> role plus rubric

All conditions now receive the same claim, argument, stance and 0--100 scale.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from argstrength.parsing import parse_score  # noqa: E402
from argstrength.prompts import SCORE_PREFIX, build_prompt  # noqa: E402
from argstrength.schemas import Stance  # noqa: E402


def _formatter(output: str) -> int:
    """Return the common 0--100 integer, including legacy manager outputs."""

    stripped = output.strip()
    try:
        numeric = float(stripped)
    except ValueError:
        numeric = None
    if numeric is not None:
        # The old manager rescales its historical ChatGPT branch to 0--1.
        if 0 <= numeric <= 1 and "." in stripped:
            numeric *= 100
        if numeric.is_integer() and 0 <= numeric <= 100:
            stripped = f"{SCORE_PREFIX} {int(numeric)}"

    parsed = parse_score(stripped, strict=True)
    if not parsed.ok:
        raise ValueError(f"Invalid score output: {parsed.error}: {output!r}")
    assert parsed.score is not None
    return parsed.score


def _legacy_prompt(
    condition: str,
    statement: str,
    *,
    claim: str | None,
    support: bool | None,
    stance: str | Stance | None,
    verbal: bool,
    topic: bool,
):
    if verbal:
        raise ValueError("The formal study uses only the common numeric 0--100 scale")
    if topic:
        raise ValueError("Topic-only scoring is outside the formal argument-strength study")
    if claim is None:
        raise ValueError("claim is required for every formal prompt condition")

    if stance is None:
        if type(support) is not bool:
            raise ValueError(
                "Pass stance='support'/'attack' or an explicit boolean support value"
            )
        stance = Stance.SUPPORT if support else Stance.ATTACK

    spec = build_prompt(
        condition,
        claim=claim,
        argument=statement,
        stance=stance,
    )
    constraints = {
        "constraint_prefix": SCORE_PREFIX,
        "constraint_options": [f" {score}" for score in range(101)],
        "constraint_end_after_options": True,
    }
    return spec.text, constraints, _formatter


class UncertaintyEvaluatorPrompts:
    """Legacy-shaped access to the controlled prompt conditions."""

    @staticmethod
    def direct(
        statement,
        claim=None,
        support=None,
        stance=None,
        verbal=False,
        topic=False,
        **_,
    ):
        return _legacy_prompt(
            "direct",
            statement,
            claim=claim,
            support=support,
            stance=stance,
            verbal=verbal,
            topic=topic,
        )

    @staticmethod
    def role(
        statement,
        claim=None,
        support=None,
        stance=None,
        verbal=False,
        topic=False,
        **_,
    ):
        return _legacy_prompt(
            "role",
            statement,
            claim=claim,
            support=support,
            stance=stance,
            verbal=verbal,
            topic=topic,
        )

    @staticmethod
    def rubric(
        statement,
        claim=None,
        support=None,
        stance=None,
        verbal=False,
        topic=False,
        **_,
    ):
        return _legacy_prompt(
            "rubric",
            statement,
            claim=claim,
            support=support,
            stance=stance,
            verbal=verbal,
            topic=topic,
        )

    @staticmethod
    def role_rubric(
        statement,
        claim=None,
        support=None,
        stance=None,
        verbal=False,
        topic=False,
        **_,
    ):
        return _legacy_prompt(
            "role_rubric",
            statement,
            claim=claim,
            support=support,
            stance=stance,
            verbal=verbal,
            topic=topic,
        )

    # Historical aliases. Result files use canonical names.
    chatgpt = direct
    analyst = role
    opro = rubric
    analyst_rubric = role_rubric
