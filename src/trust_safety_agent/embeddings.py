"""Deterministic offline embeddings used by the Day 2 baseline."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[:'-][a-z0-9]+)*")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "when",
    "with",
}


class HashingEmbedder:
    """Dependency-free lexical baseline with stable, normalized vectors."""

    def __init__(self, dimension: int = 2048) -> None:
        if dimension < 64:
            raise ValueError("dimension must be at least 64")
        self.dimension = dimension
        self.name = f"hashing-v1-{dimension}"

    def _features(self, text: str) -> Iterable[Tuple[str, float]]:
        tokens = [
            token
            for token in TOKEN_PATTERN.findall(text.lower())
            if token not in STOP_WORDS
        ]
        for token in tokens:
            yield f"u:{token}", 1.0
            if len(token) >= 5:
                for offset in range(len(token) - 2):
                    yield f"c:{token[offset:offset + 3]}", 0.2
        for left, right in zip(tokens, tokens[1:]):
            yield f"b:{left}_{right}", 1.4

    def embed_query(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
