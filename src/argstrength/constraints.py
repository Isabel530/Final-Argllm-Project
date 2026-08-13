"""Tokenizer-aware hard constraints for complete ``Score: N`` outputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    terminal: bool = False


class TokenTrie:
    def __init__(self, sequences: Iterable[Sequence[int]]) -> None:
        self.root = _TrieNode()
        count = 0
        for sequence in sequences:
            if not sequence:
                raise ValueError("Allowed token sequence cannot be empty")
            node = self.root
            for token_id in sequence:
                node = node.children.setdefault(int(token_id), _TrieNode())
            node.terminal = True
            count += 1
        if count == 0:
            raise ValueError("At least one allowed token sequence is required")

    def allowed_next(self, prefix: Sequence[int], eos_token_id: int) -> list[int]:
        node = self.root
        for token_id in prefix:
            child = node.children.get(int(token_id))
            if child is None:
                return [eos_token_id]
            node = child
        allowed = list(node.children)
        if node.terminal:
            allowed.append(eos_token_id)
        return sorted(set(allowed))


def encode_allowed_outputs(tokenizer: Any, outputs: Iterable[str]) -> list[list[int]]:
    sequences = []
    for output in outputs:
        token_ids = tokenizer.encode(output, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Tokenizer produced no tokens for {output!r}")
        sequences.append(token_ids)
    return sequences


def make_prefix_allowed_tokens_fn(
    *,
    tokenizer: Any,
    prompt_length: int,
    allowed_outputs: Iterable[str],
):
    """Return a Transformers ``prefix_allowed_tokens_fn``.

    ``prompt_length`` must be taken from the actual tokenised model input.
    This avoids the legacy bug caused by re-tokenising a prompt inside every
    callback and assuming that every tokenizer inserts exactly one BOS token.
    """

    if prompt_length < 0:
        raise ValueError("prompt_length must be non-negative")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    trie = TokenTrie(encode_allowed_outputs(tokenizer, allowed_outputs))

    def allowed_tokens(batch_id: int, input_ids: Any) -> list[int]:
        del batch_id
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        generated = ids[prompt_length:]
        return trie.allowed_next(generated, eos_token_id)

    return allowed_tokens
