"""Typed records shared by data loading, generation, and analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Stance(str, Enum):
    SUPPORT = "support"
    ATTACK = "attack"

    @classmethod
    def from_effective_support(cls, value: bool) -> "Stance":
        if type(value) is not bool:
            raise TypeError("is_effective_support must be a JSON boolean")
        return cls.SUPPORT if value else cls.ATTACK


class PromptCondition(str, Enum):
    DIRECT = "direct"
    ROLE = "role"
    RUBRIC = "rubric"
    ROLE_RUBRIC = "role_rubric"

    @classmethod
    def normalise(cls, value: str | "PromptCondition") -> "PromptCondition":
        if isinstance(value, cls):
            return value
        aliases = {
            "chatgpt": cls.DIRECT,
            "analyst": cls.ROLE,
            "opro": cls.RUBRIC,
            "opro-derived": cls.RUBRIC,
            "analyst_rubric": cls.ROLE_RUBRIC,
            "role+rubric": cls.ROLE_RUBRIC,
        }
        key = value.strip().lower()
        if key in aliases:
            return aliases[key]
        try:
            return cls(key)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown prompt condition {value!r}; expected {allowed}") from exc


@dataclass(frozen=True)
class ArgumentSet:
    claim_id: int
    claim: str
    stance: Stance
    arguments: tuple[str, ...]
    reference_ranks: tuple[int, ...]
    source_is_support: bool | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepresentationGroup:
    group_id: str
    claim_id: int
    claim: str
    stance: Stance
    summary: str
    expanded_arguments: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def variants(self) -> tuple[tuple[str, str], ...]:
        expanded = tuple(
            (f"expanded_{index}", text)
            for index, text in enumerate(self.expanded_arguments, start=1)
        )
        return (("summary", self.summary),) + expanded


@dataclass(frozen=True)
class GenerationRecord:
    run_id: str
    study: str
    model_key: str
    model_name: str
    prompt_condition: str
    decoding: str
    thinking: bool
    seed: int
    claim_id: int
    claim: str
    stance: str
    argument_id: str
    argument: str
    reference_rank: int | None
    raw_output: str
    score: int | None
    score_parse_ok: bool
    score_parse_error: str | None
    format_ok: bool
    format_error: str | None
    generated_tokens: int
    hit_max_new_tokens: bool
    generation_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
