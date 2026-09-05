"""BM25, dense retrieval, rank fusion and MMR.

BM25 earns its place in a filings corpus specifically: tickers, product names
("H20", "CUDA", "Azure"), segment labels and regulatory terms ("State Aid
Decision") are exact tokens. A bi-encoder maps them to a fuzzy neighbourhood;
BM25 matches them outright.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOPWORDS = {
    "the","a","an","and","or","of","to","in","for","on","is","are","be","by","with","as","at",
    "that","this","from","not","it","its","our","we","their","which","what","how","much","many",
    "do","does","did","was","were","has","have","had","will","would","can","could","i","you",
}


def tokenize(text: str) -> list[str]:
    """Keep alphanumeric runs and hyphenated/dotted codes intact: H20, 10-K,
    non-GAAP, R&D all have to survive as searchable tokens."""
    raw = re.findall(r"[A-Za-z0-9]+(?:[-&][A-Za-z0-9]+)*", text.lower())
    out: list[str] = []
    for tok in raw:
        if tok in STOPWORDS:
            continue
        out.append(tok)
        if "-" in tok or "&" in tok:
            out.extend(p for p in re.split(r"[-&]", tok) if p and p not in STOPWORDS)
    return out


class BM25:
    """score = Σ_t IDF(t)·f(t,d)(k1+1) / (f(t,d) + k1(1 − b + b·|d|/avgdl))

    k1 saturates term frequency (the tenth "revenue" adds almost nothing);
    b normalises for length, which matters here because Risk Factors sections
    run 5-20x longer than Properties.
    """
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        self.tf = [Counter(d) for d in corpus_tokens]
        df: Counter = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        # inverted index: scoring only touches documents containing a query term
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(corpus_tokens):
            for t in set(d):
                self.postings.setdefault(t, []).append(i)

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for t in tokenize(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


class DenseIndex:
    """Exact inner product over L2-normalised vectors == cosine.

    Identical to FAISS IndexFlatIP, which is what the original code used. Exact
    search has no training step and no recall loss; below roughly a million
    vectors it is the right default. HNSW is where you go when latency forces
    it, and then you measure the recall you traded rather than assuming none.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self.matrix: np.ndarray | None = None

    def build(self, texts: list[str]) -> "DenseIndex":
        self.matrix = self.embedder.encode(texts)
        return self

    def score(self, query: str) -> np.ndarray:
        assert self.matrix is not None, "call build() first"
        return self.matrix @ self.embedder.encode([query], is_query=True)[0]


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(len(scores))
    return ranks


def reciprocal_rank_fusion(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """RRF(d) = Σ_lists 1/(k + rank_list(d))

    Rank-based, so no score normalisation is needed. That is the argument over
    weighted score addition: BM25 scores are unbounded and shift with the
    corpus, cosine sits in [-1, 1], and any weighting tuned on today's filings
    is wrong after the next 10-K lands.
    """
    fused = np.zeros(len(score_lists[0]), dtype=np.float32)
    for s in score_lists:
        fused += 1.0 / (k + ranks_from_scores(s) + 1)
    return fused


def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, candidates: list[int],
        k: int = 5, lam: float = 0.7) -> list[int]:
    """Relevance against novelty. Risk Factors sections repeat themselves
    heavily, so without MMR the top-5 is often five paraphrases of one risk."""
    selected: list[int] = []
    remaining = list(candidates)
    rel = doc_vecs @ query_vec
    while remaining and len(selected) < k:
        best, best_score = remaining[0], -np.inf
        for i in remaining:
            novelty = 0.0 if not selected else max(float(doc_vecs[i] @ doc_vecs[j]) for j in selected)
            s = lam * float(rel[i]) - (1 - lam) * novelty
            if s > best_score:
                best, best_score = i, s
        selected.append(best)
        remaining.remove(best)
    return selected
