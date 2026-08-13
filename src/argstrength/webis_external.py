"""Locked Webis-ArgQuality-20 data adapters for the E11 external study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import DataValidationError
from .schemas import ArgumentSet, Stance


HUMAN_LABELS = {"A", "B", "Tie"}


def load_webis_generation_dataset(path: str | Path) -> list[ArgumentSet]:
    """Load variable-sized topic/stance groups prepared for E11 generation."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    output: list[ArgumentSet] = []
    seen_claim_ids: set[int] = set()
    seen_group_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{source}:{line_number}"
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"{where}: invalid JSON") from exc

            claim_id = row.get("external_claim_id")
            if type(claim_id) is not int or claim_id < 0:
                raise DataValidationError(
                    f"{where}: external_claim_id must be a non-negative integer"
                )
            if claim_id in seen_claim_ids:
                raise DataValidationError(f"{where}: duplicate external_claim_id")
            seen_claim_ids.add(claim_id)

            group_id = row.get("group_id")
            if not isinstance(group_id, str) or not group_id.strip():
                raise DataValidationError(f"{where}: group_id must be non-empty")
            if group_id in seen_group_ids:
                raise DataValidationError(f"{where}: duplicate group_id")
            seen_group_ids.add(group_id)

            claim = row.get("statement")
            if not isinstance(claim, str) or not claim.strip():
                raise DataValidationError(f"{where}: statement must be non-empty")
            effective = row.get("is_effective_support")
            if type(effective) is not bool:
                raise DataValidationError(
                    f"{where}: is_effective_support must be boolean"
                )
            arguments = row.get("arguments")
            argument_keys = row.get("argument_keys")
            combined_quality = row.get("combined_quality")
            if (
                not isinstance(arguments, list)
                or len(arguments) < 2
                or any(not isinstance(value, str) or not value.strip() for value in arguments)
            ):
                raise DataValidationError(
                    f"{where}: arguments must contain at least two non-empty strings"
                )
            if (
                not isinstance(argument_keys, list)
                or len(argument_keys) != len(arguments)
                or len(set(argument_keys)) != len(argument_keys)
            ):
                raise DataValidationError(
                    f"{where}: argument_keys must uniquely match arguments"
                )
            if not isinstance(combined_quality, list) or len(combined_quality) != len(arguments):
                raise DataValidationError(
                    f"{where}: combined_quality must match arguments"
                )

            quality = [float(value) for value in combined_quality]
            ordered_indices = sorted(
                range(len(quality)), key=lambda index: (-quality[index], argument_keys[index])
            )
            ranks = [0] * len(quality)
            for rank, index in enumerate(ordered_indices, start=1):
                ranks[index] = rank
            metadata = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "external_claim_id",
                    "statement",
                    "is_effective_support",
                    "arguments",
                }
            }
            output.append(
                ArgumentSet(
                    claim_id=claim_id,
                    claim=claim.strip(),
                    stance=Stance.from_effective_support(effective),
                    arguments=tuple(value.strip() for value in arguments),
                    reference_ranks=tuple(ranks),
                    source_is_support=effective,
                    metadata=metadata,
                )
            )
    if not output:
        raise DataValidationError(f"{source}: dataset is empty")
    return output


def load_webis_pair_labels(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the direct-human E11 comparison table."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    output: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{source}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"{where}: invalid JSON") from exc
            pair_id = row.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id:
                raise DataValidationError(f"{where}: pair_id must be non-empty")
            if pair_id in seen_pair_ids:
                raise DataValidationError(f"{where}: duplicate pair_id")
            seen_pair_ids.add(pair_id)
            if type(row.get("external_claim_id")) is not int:
                raise DataValidationError(f"{where}: external_claim_id must be integer")
            if row.get("argument_a_id") == row.get("argument_b_id"):
                raise DataValidationError(f"{where}: pair endpoints must differ")
            for dimension in ("logical", "rhetorical", "dialectical"):
                if row.get(dimension) not in HUMAN_LABELS:
                    raise DataValidationError(f"{where}: invalid {dimension} label")
            majority = row.get("majority_label")
            if majority is not None and majority not in HUMAN_LABELS:
                raise DataValidationError(f"{where}: invalid majority_label")
            output.append(row)
    if not output:
        raise DataValidationError(f"{source}: pair table is empty")
    return output
