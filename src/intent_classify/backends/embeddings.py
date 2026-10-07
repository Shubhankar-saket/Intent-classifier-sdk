"""Embedding adapters."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from .base import l2_normalize


class HashingEmbedder:
    """Dependency-free character n-gram embedder (hashing trick).

    Purely lexical: good for tests, offline baselines and near-duplicate detection, not for
    paraphrase understanding. Language-agnostic, so it degrades gracefully on non-English.
    """

    name = "hashing"
    default_accept_threshold = 0.75

    def __init__(self, dim: int = 4096, ngram_range: tuple[int, int] = (2, 4)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _features(self, text: str) -> list[str]:
        text = f" {re.sub(r'[^\w\s]', ' ', text.lower())} "
        text = re.sub(r"\s+", " ", text)
        feats = [f"w:{w}" for w in text.split()]
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            feats.extend(text[i : i + n] for i in range(len(text) - n + 1))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return l2_normalize(out)


class OpenAIEmbedder:
    name = "openai"
    default_accept_threshold = 0.80

    def __init__(self, model: str = "text-embedding-3-small", client: Any = None, **client_kwargs: Any):
        import openai

        self.model = model
        self.client = client or openai.OpenAI(**client_kwargs)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 512):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + 512])
            vectors.extend(d.embedding for d in resp.data)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


class SentenceTransformerEmbedder:
    """Local model. The default is multilingual so non-English projects work out of the box."""

    name = "sentence-transformers"
    default_accept_threshold = 0.82

    def __init__(self, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32))
