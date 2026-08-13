"""Loading and validation for locked external argument-ranking panels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import DataValidationError
from .schemas import ArgumentSet, Stance


def load_external_rank_dataset(path: str | Path) -> list[ArgumentSet]:
    """Load a prepared E9 JSONL while allowing repeated topics across panels.

    Unlike the main dataset, the same topic legitimately appears under both
    stances and in both the broad and near-tie panels.  ``external_claim_id``
    is therefore the unique generation key; ``source_topic`` is retained as
    the independent statistical cluster.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    output: list[ArgumentSet] = []
    seen_ids: set[int] = set()
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
            if claim_id in seen_ids:
                raise DataValidationError(f"{where}: duplicate external_claim_id")
            seen_ids.add(claim_id)

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
            arguments = row.get("ground_truth_ranking")
            if (
                not isinstance(arguments, list)
                or len(arguments) != 4
                or any(not isinstance(value, str) or not value.strip() for value in arguments)
            ):
                raise DataValidationError(
                    f"{where}: ground_truth_ranking must contain four strings"
                )
            cleaned = tuple(value.strip() for value in arguments)
            if len(set(cleaned)) != 4:
                raise DataValidationError(f"{where}: duplicate selected argument")

            quality = row.get("quality_wa")
            if not isinstance(quality, list) or len(quality) != 4:
                raise DataValidationError(f"{where}: quality_wa must have length four")
            quality_values = tuple(float(value) for value in quality)
            if any(
                quality_values[index] <= quality_values[index + 1]
                for index in range(3)
            ):
                raise DataValidationError(
                    f"{where}: quality_wa must be strictly descending"
                )

            panel = row.get("panel")
            if panel not in {"broad", "near_tie"}:
                raise DataValidationError(f"{where}: invalid panel")
            source_topic = row.get("source_topic")
            if not isinstance(source_topic, str) or not source_topic.strip():
                raise DataValidationError(f"{where}: source_topic must be non-empty")

            metadata = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "external_claim_id",
                    "statement",
                    "is_effective_support",
                    "ground_truth_ranking",
                }
            }
            output.append(
                ArgumentSet(
                    claim_id=claim_id,
                    claim=claim.strip(),
                    stance=Stance.from_effective_support(effective),
                    arguments=cleaned,
                    reference_ranks=(1, 2, 3, 4),
                    source_is_support=effective,
                    metadata=metadata,
                )
            )
    if not output:
        raise DataValidationError(f"{source}: dataset is empty")
    return output
