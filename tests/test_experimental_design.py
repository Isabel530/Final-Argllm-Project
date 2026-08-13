from __future__ import annotations

from argstrength.experimental_design import make_claim_splits, screening_batch
from argstrength.schemas import ArgumentSet, Stance


def _items():
    output = []
    for index in range(20):
        stance = Stance.SUPPORT if index < 10 else Stance.ATTACK
        output.append(
            ArgumentSet(
                claim_id=index,
                claim=f"claim {index}",
                stance=stance,
                arguments=("a", "b", "c", "d"),
                reference_ranks=(1, 2, 3, 4),
                source_is_support=stance is Stance.SUPPORT,
            )
        )
    return output


def test_locked_splits_are_disjoint_complete_and_reproducible():
    items = _items()
    first = make_claim_splits(
        items, train_size=10, validation_size=4, test_size=6, seed=7
    )
    second = make_claim_splits(
        items, train_size=10, validation_size=4, test_size=6, seed=7
    )
    assert first == second
    ids = [claim for values in first["splits"].values() for claim in values]
    assert len(ids) == len(set(ids)) == 20


def test_screening_batch_has_three_run_design_and_balanced_claims():
    items = _items()
    selected = screening_batch(items, range(20), iteration=2, batch_size=10)
    by_id = {item.claim_id: item for item in items}
    assert len(selected) == 10
    assert sum(by_id[claim_id].stance is Stance.SUPPORT for claim_id in selected) == 5
