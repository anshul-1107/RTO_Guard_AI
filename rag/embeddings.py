"""
Embedding providers.

- GeminiEmbedder: gemini-embedding-001 with task types (document vs query),
  reduced to EMBED_DIM via Matryoshka truncation and re-normalised.
- FakeEmbedder: deterministic hashed bag-of-words. No API key, used in tests
  and offline dev. Retrieval quality is keyword-level only.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Protocol

import numpy as np

from rto_guard.config import settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


class GeminiEmbedder:
    BATCH = 50

    def __init__(self, model: str | None = None, dim: int | None = None):
        from google import genai

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (or use EMBEDDINGS_PROVIDER=fake)")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = model or settings.gemini_embed_model
        self.dim = dim or settings.embed_dim

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            for attempt in range(4):
                try:
                    res = self.client.models.embed_content(
                        model=self.model, contents=batch, config=cfg
                    )
                    break
                except Exception:  # rate limits / transient errors
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            vecs = np.array([e.values for e in res.embeddings], dtype=np.float32)
            if len(vecs) != len(batch):
                raise RuntimeError(f"expected {len(batch)} embeddings, got {len(vecs)}")
            out.extend(_normalize(vecs).tolist())  # truncated MRL vectors need re-normalising
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class FakeEmbedder:
    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.embed_dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9₹%]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        return _normalize(v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def get_embedder() -> Embedder:
    if settings.embeddings_provider == "fake" or not (settings.gemini_api_key and settings.gemini_api_key.strip()):
        return FakeEmbedder()
    return GeminiEmbedder()

