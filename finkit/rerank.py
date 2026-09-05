"""Cross-encoder reranking.

Bi-encoder: query and document encoded separately, so document vectors
precompute and search is a matmul — but the query never attends to the
document. Cross-encoder: the pair goes through together with full
cross-attention. Far more accurate, one forward pass per candidate, nothing
precomputes. Hence retrieve ~40 cheaply, rerank, keep ~5.
"""
from __future__ import annotations

from .retrieval import tokenize


class CrossEncoderReranker:
    name = "cross-encoder:ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class LexicalOverlapReranker:
    """Fallback with no model. Weak, and worth naming as weak: an abstention
    threshold is only as trustworthy as the scorer behind it."""
    name = "lexical-overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        out = []
        for t in texts:
            d = tokenize(t)
            if not q or not d:
                out.append(0.0)
                continue
            dset = set(d)
            coverage = len(q & dset) / len(q)
            pos = [i for i, tok in enumerate(d) if tok in q]
            prox = 1.0 / (1.0 + (max(pos) - min(pos)) / len(d)) if len(pos) > 1 else 0.5
            out.append(0.75 * coverage + 0.25 * prox)
        return out


def get_reranker(prefer: str = "auto"):
    if prefer in ("auto", "cross-encoder"):
        try:
            return CrossEncoderReranker()
        except Exception as exc:
            if prefer == "cross-encoder":
                raise
            print(f"  [rerank] cross-encoder unavailable ({type(exc).__name__}); using lexical overlap")
    return LexicalOverlapReranker()
