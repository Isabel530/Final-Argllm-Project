"""Dataset loading and validation.

The main study uses ``is_effective_support`` as the operative stance.  The
original ``is_support`` field is retained only as provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .schemas import ArgumentSet, RepresentationGroup, Stance


class DataValidationError(ValueError):
    pass


def _jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise DataValidationError(
                    f"{path}:{line_number}: each line must contain a JSON object"
                )
            yield line_number, value


def _nonempty_string(record: dict[str, Any], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{where}: {key} must be a non-empty string")
    return value.strip()


def load_main_dataset(
    path: str | Path,
    *,
    expected_arguments: int | None = 4,
) -> list[ArgumentSet]:
    items: list[ArgumentSet] = []
    seen_claims: set[str] = set()
    for claim_id, (line_number, record) in enumerate(_jsonl(path)):
        where = f"{path}:{line_number}"
        claim = _nonempty_string(record, "statement", where)
        if claim in seen_claims:
            raise DataValidationError(f"{where}: duplicate claim")
        seen_claims.add(claim)

        effective = record.get("is_effective_support")
        if type(effective) is not bool:
            raise DataValidationError(
                f"{where}: is_effective_support must be true or false"
            )
        stance = Stance.from_effective_support(effective)

        arguments = record.get("ground_truth_ranking")
        if not isinstance(arguments, list) or not arguments:
            raise DataValidationError(
                f"{where}: ground_truth_ranking must be a non-empty list"
            )
        if expected_arguments is not None and len(arguments) != expected_arguments:
            raise DataValidationError(
                f"{where}: expected {expected_arguments} arguments, found {len(arguments)}"
            )
        if any(not isinstance(arg, str) or not arg.strip() for arg in arguments):
            raise DataValidationError(f"{where}: all arguments must be non-empty strings")
        arguments_tuple = tuple(arg.strip() for arg in arguments)
        if len(set(arguments_tuple)) != len(arguments_tuple):
            raise DataValidationError(f"{where}: duplicate argument within claim")

        declared_count = record.get("num_arguments")
        if declared_count is not None and declared_count != len(arguments_tuple):
            raise DataValidationError(
                f"{where}: num_arguments={declared_count} but found {len(arguments_tuple)}"
            )

        source_support = record.get("is_support")
        if source_support is not None and type(source_support) is not bool:
            raise DataValidationError(f"{where}: is_support must be boolean or null")

        items.append(
            ArgumentSet(
                claim_id=claim_id,
                claim=claim,
                stance=stance,
                arguments=arguments_tuple,
                reference_ranks=tuple(range(1, len(arguments_tuple) + 1)),
                source_is_support=source_support,
                metadata={"source_line": line_number},
            )
        )
    if not items:
        raise DataValidationError(f"{path}: dataset is empty")
    return items


def load_representation_dataset(path: str | Path) -> list[RepresentationGroup]:
    groups: list[RepresentationGroup] = []
    seen_keys: set[str] = set()
    for line_number, record in _jsonl(path):
        where = f"{path}:{line_number}"
        key = _nonempty_string(record, "key", where)
        if key in seen_keys:
            raise DataValidationError(f"{where}: duplicate key {key!r}")
        seen_keys.add(key)

        claim = _nonempty_string(record, "claim", where)
        summary = _nonempty_string(record, "summary", where)
        section = record.get("section")
        if section not in {"support", "attack"}:
            raise DataValidationError(
                f"{where}: section must be 'support' or 'attack'"
            )
        expanded = record.get("expanded_arguments")
        if not isinstance(expanded, list) or not expanded:
            raise DataValidationError(
                f"{where}: expanded_arguments must be a non-empty list"
            )
        if any(not isinstance(arg, str) or not arg.strip() for arg in expanded):
            raise DataValidationError(
                f"{where}: all expanded arguments must be non-empty strings"
            )
        claim_index = record.get("claim_index")
        if type(claim_index) is not int or claim_index < 0:
            raise DataValidationError(f"{where}: claim_index must be non-negative int")

        groups.append(
            RepresentationGroup(
                group_id=key,
                claim_id=claim_index,
                claim=claim,
                stance=Stance(section),
                summary=summary,
                expanded_arguments=tuple(arg.strip() for arg in expanded),
                metadata={
                    "source_line": line_number,
                    "summary_index": record.get("summary_index"),
                },
            )
        )
    if not groups:
        raise DataValidationError(f"{path}: dataset is empty")
    return groups


def summarise_main_dataset(items: list[ArgumentSet]) -> dict[str, int]:
    return {
        "claims": len(items),
        "arguments": sum(len(item.arguments) for item in items),
        "support_claims": sum(item.stance is Stance.SUPPORT for item in items),
        "attack_claims": sum(item.stance is Stance.ATTACK for item in items),
        "source_stance_null": sum(item.source_is_support is None for item in items),
        "source_effective_mismatches": sum(
            item.source_is_support is not None
            and item.source_is_support != (item.stance is Stance.SUPPORT)
            for item in items
        ),
    }
