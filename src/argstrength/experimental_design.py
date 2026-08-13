"""Locked claim splits and rotating screening batches for UQ/APO studies.

The original three studies remain untouched.  This module creates a new,
claim-disjoint development protocol for uncertainty estimation and automatic
prompt optimisation.  Claims, rather than arguments or runs, are split so no
claim can leak across optimisation, validation, and internal test partitions.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence

from .schemas import ArgumentSet


SPLIT_NAMES = ("apo_train", "apo_validation", "internal_test")


def dataset_fingerprint(items: Sequence[ArgumentSet]) -> str:
    """Return a stable fingerprint of claim IDs, text, stance, and arguments."""

    digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: value.claim_id):
        payload = {
            "claim_id": item.claim_id,
            "claim": item.claim,
            "stance": item.stance.value,
            "arguments": list(item.arguments),
            "reference_ranks": list(item.reference_ranks),
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _allocate_counts(
    total_by_stance: dict[str, int], split_sizes: dict[str, int]
) -> dict[str, dict[str, int]]:
    """Allocate split sizes proportionally while preserving exact totals."""

    total = sum(total_by_stance.values())
    if sum(split_sizes.values()) != total:
        raise ValueError("split sizes must sum to the number of claims")
    if set(total_by_stance) != {"support", "attack"}:
        raise ValueError("expected support and attack stance counts")

    # Allocate support with largest-remainder apportionment. Attack counts are
    # then determined by each split's requested total.
    exact = {
        name: split_sizes[name] * total_by_stance["support"] / total
        for name in SPLIT_NAMES
    }
    support = {name: int(exact[name]) for name in SPLIT_NAMES}
    remaining = total_by_stance["support"] - sum(support.values())
    for name in sorted(
        SPLIT_NAMES,
        key=lambda value: (exact[value] - support[value], split_sizes[value]),
        reverse=True,
    )[:remaining]:
        support[name] += 1

    allocation = {
        name: {
            "support": support[name],
            "attack": split_sizes[name] - support[name],
        }
        for name in SPLIT_NAMES
    }
    if sum(row["attack"] for row in allocation.values()) != total_by_stance["attack"]:
        raise AssertionError("stance allocation failed to preserve totals")
    return allocation


def make_claim_splits(
    items: Sequence[ArgumentSet],
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, Any]:
    """Create reproducible, stance-stratified, claim-disjoint partitions."""

    if not items:
        raise ValueError("items cannot be empty")
    sizes = {
        "apo_train": train_size,
        "apo_validation": validation_size,
        "internal_test": test_size,
    }
    if any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise ValueError("all split sizes must be positive integers")
    if sum(sizes.values()) != len(items):
        raise ValueError(
            f"split sizes sum to {sum(sizes.values())}, expected {len(items)}"
        )

    grouped: dict[str, list[int]] = defaultdict(list)
    for item in items:
        grouped[item.stance.value].append(item.claim_id)
    allocation = _allocate_counts(
        {stance: len(ids) for stance, ids in grouped.items()}, sizes
    )

    rng = random.Random(seed)
    for claim_ids in grouped.values():
        rng.shuffle(claim_ids)

    partitions: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for stance in ("support", "attack"):
        cursor = 0
        for name in SPLIT_NAMES:
            count = allocation[name][stance]
            partitions[name].extend(grouped[stance][cursor : cursor + count])
            cursor += count
    for name in SPLIT_NAMES:
        rng.shuffle(partitions[name])

    document = {
        "schema_version": 1,
        "seed": seed,
        "dataset_sha256": dataset_fingerprint(items),
        "n_claims": len(items),
        "splits": partitions,
        "stance_counts": allocation,
        "warning": (
            "Internal APO holdout only: all source claims were examined in earlier "
            "dissertation analyses, so this is not a pristine external test set."
        ),
    }
    validate_claim_splits(document, items)
    return document


def validate_claim_splits(
    document: dict[str, Any], items: Sequence[ArgumentSet]
) -> None:
    """Fail on overlap, missing claims, fingerprint drift, or stance drift."""

    splits = document.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"splits must contain exactly {SPLIT_NAMES}")
    lists = [splits[name] for name in SPLIT_NAMES]
    if any(not isinstance(values, list) for values in lists):
        raise ValueError("each split must be a list of claim IDs")
    flattened = [claim_id for values in lists for claim_id in values]
    if any(type(claim_id) is not int for claim_id in flattened):
        raise ValueError("claim IDs must be integers")
    if len(flattened) != len(set(flattened)):
        raise ValueError("claim splits overlap or contain duplicate IDs")
    expected = {item.claim_id for item in items}
    if set(flattened) != expected:
        missing = sorted(expected - set(flattened))
        extra = sorted(set(flattened) - expected)
        raise ValueError(f"claim split mismatch; missing={missing}, extra={extra}")
    if document.get("dataset_sha256") != dataset_fingerprint(items):
        raise ValueError("dataset fingerprint does not match the locked split")

    by_id = {item.claim_id: item for item in items}
    declared = document.get("stance_counts", {})
    for name in SPLIT_NAMES:
        actual = {
            stance: sum(by_id[claim_id].stance.value == stance for claim_id in splits[name])
            for stance in ("support", "attack")
        }
        if declared.get(name) != actual:
            raise ValueError(
                f"stance counts for {name} are {actual}, not {declared.get(name)}"
            )


def load_claim_splits(
    path: str | Path, items: Sequence[ArgumentSet]
) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    validate_claim_splits(document, items)
    return document


def screening_batch(
    items: Sequence[ArgumentSet],
    train_claim_ids: Sequence[int],
    *,
    iteration: int,
    batch_size: int = 10,
    seed: int = 2026,
) -> list[int]:
    """Select a balanced, rotating APO screening batch from training only."""

    if type(iteration) is not int or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if batch_size % 2:
        raise ValueError("balanced screening requires an even batch_size")

    by_id = {item.claim_id: item for item in items}
    unknown = set(train_claim_ids) - set(by_id)
    if unknown:
        raise ValueError(f"unknown training claim IDs: {sorted(unknown)}")
    per_stance = batch_size // 2
    selected: list[int] = []
    for offset, stance in enumerate(("support", "attack")):
        pool = [
            claim_id
            for claim_id in train_claim_ids
            if by_id[claim_id].stance.value == stance
        ]
        if len(pool) < per_stance:
            raise ValueError(f"not enough {stance} claims for a balanced batch")
        random.Random(seed + 10_007 * iteration + offset).shuffle(pool)
        selected.extend(pool[:per_stance])
    random.Random(seed + 65_537 * iteration).shuffle(selected)
    return selected
