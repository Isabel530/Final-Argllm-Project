"""Embedding utilities for interpretable, discrete prompt optimisation.

Embeddings organise failure descriptions, critiques, and candidate prompts;
they never replace the readable prompt text.  The transformer backend is the
research backend.  The hashing backend is deterministic and useful for unit
tests or dry runs, but should be labelled lexical rather than semantic.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence


class TextEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...

    def close(self) -> None: ...


class HashingTextEmbedder:
    """Dependency-free lexical hashing vectors for tests and dry runs."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        output = []
        for text in texts:
            vector = [0.0] * self.dimensions
            tokens = re.findall(r"[\w']+", text.lower())
            features = tokens + [
                f"{left}_{right}" for left, right in zip(tokens, tokens[1:])
            ]
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            output.append(_normalise(vector))
        return output

    def close(self) -> None:
        return None


class TransformerTextEmbedder:
    """Mean-pooled Hugging Face encoder with L2-normalised output vectors."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer embeddings require torch and transformers"
            ) from exc
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        self._torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            output.extend(pooled.detach().cpu().float().tolist())
        return output

    def close(self) -> None:
        import gc

        del self.model
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [float(value / norm) for value in vector] if norm else list(vector)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("vectors must have equal non-zero dimensions")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def cluster_embeddings(
    vectors: Sequence[Sequence[float]], *, similarity_threshold: float = 0.72
) -> list[list[int]]:
    """Connected-component clustering under a cosine-similarity threshold."""

    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in [-1, 1]")
    if not vectors:
        return []
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or next(iter(dimensions)) == 0:
        raise ValueError("vectors must share a non-zero dimension")
    parents = list(range(len(vectors)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left, right in combinations_indices(len(vectors)):
        if cosine_similarity(vectors[left], vectors[right]) >= similarity_threshold:
            union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(vectors)):
        grouped.setdefault(find(index), []).append(index)
    return sorted(grouped.values(), key=lambda group: group[0])


def combinations_indices(size: int):
    for left in range(size):
        for right in range(left + 1, size):
            yield left, right


def cluster_representatives(
    vectors: Sequence[Sequence[float]], clusters: Sequence[Sequence[int]]
) -> list[int]:
    """Return each cluster's cosine medoid, with deterministic tie breaking."""

    representatives = []
    for cluster in clusters:
        if not cluster:
            raise ValueError("clusters cannot contain empty groups")
        representatives.append(
            max(
                cluster,
                key=lambda candidate: (
                    sum(
                        cosine_similarity(vectors[candidate], vectors[other])
                        for other in cluster
                    )
                    / len(cluster),
                    -candidate,
                ),
            )
        )
    return representatives


def deduplicate_texts(
    texts: Sequence[str],
    embedder: TextEmbedder,
    *,
    similarity_threshold: float = 0.92,
) -> list[int]:
    """Keep the earliest readable text from each semantic-near-duplicate set."""

    if not texts:
        return []
    vectors = embedder.encode(texts)
    kept: list[int] = []
    for index, vector in enumerate(vectors):
        if any(texts[index].strip() == texts[previous].strip() for previous in kept):
            continue
        if all(
            cosine_similarity(vector, vectors[previous]) < similarity_threshold
            for previous in kept
        ):
            kept.append(index)
    return kept
