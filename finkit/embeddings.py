"""Pluggable embedders.

Preference order: the local sentence-transformers bi-encoder the project
already used, then a hosted API if a key is present, then a pure-numpy TF-IDF
fallback so tests and CI always run with no download and no network.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Protocol

import numpy as np


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Embedder(Protocol):
    name: str
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 — 384 dimensions, ~80MB, CPU-friendly.

    Kept from the original project. The honest trade-off: a larger or
    finance-tuned encoder would retrieve better, and the right move is to
    measure that uplift before paying for it in latency and memory. It also
    runs in-process, which matters when the alternative is shipping filing
    text to a third party.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name.split('/')[-1]}"
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vecs = self.model.encode(missing, convert_to_numpy=True, show_progress_bar=False,
                                     normalize_embeddings=True, batch_size=64)
            for t, v in zip(missing, vecs):
                self._cache[t] = v.astype(np.float32)
        return np.vstack([self._cache[t] for t in texts])


class OpenAIEmbedder:  # pragma: no cover - needs a key
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 128):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + 128])
            out.extend(d.embedding for d in resp.data)
        arr = np.array(out, dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


class TfidfEmbedder:
    """Deterministic numpy fallback. Not competitive; keeps CI green offline."""
    name = "tfidf-hash"

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        n = len(texts)
        df: Counter = Counter()
        for t in texts:
            df.update(set(_tokens(t)))
        self.idf = {tok: math.log((n + 1) / (c + 1)) + 1.0 for tok, c in df.items()}
        return self

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok, tf in Counter(_tokens(t)).items():
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self.dim
                out[i, h] += (1 + math.log(tf)) * self.idf.get(tok, 1.0)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def get_embedder(prefer: str = "auto", corpus: list[str] | None = None) -> Embedder:
    if prefer in ("auto", "local"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:
            if prefer == "local":
                raise
            print(f"  [embeddings] sentence-transformers unavailable ({type(exc).__name__}); falling back")
    if prefer in ("auto", "openai") and os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except Exception:
            pass
    emb = TfidfEmbedder()
    if corpus:
        emb.fit(corpus)
    return emb
