"""Pipeline assembly. One config object, so a variant is a one-line change and
the ablation table in the README is reproducible.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunking import chunk_sections
from .embeddings import Embedder, get_embedder
from .filings import Chunk, FilingSection, sections_to_documents
from .rerank import get_reranker
from .retrieval import BM25, DenseIndex, mmr, reciprocal_rank_fusion, tokenize

FIXTURES = Path(__file__).resolve().parent.parent / "eval" / "fixtures"


def load_fixture_sections(path: str | Path = FIXTURES) -> list[FilingSection]:
    """Load the cached filings. Offline and deterministic on purpose."""
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no cached filings in {path}; run `python fetch_fixtures.py`")
    out: list[FilingSection] = []
    for f in files:
        d = json.loads(f.read_text())
        out.extend(sections_to_documents(
            d["ticker"], d["filing_type"], d["filing_date"], d["url"], d["sections"]))
    return out


@dataclass(frozen=True)
class RagConfig:
    name: str
    chunker: str = "section_aware"        # word_window | section_aware
    chunk_kwargs: dict = field(default_factory=dict)
    retrieval: str = "hybrid"             # dense | bm25 | hybrid
    candidates: int = 40
    rerank: bool = True
    diversify: bool = False
    top_k: int = 5
    ticker_filter: bool = False           # scope to the ticker named in the query
    abstain_below: float | None = None


@dataclass
class Hit:
    chunk: Chunk
    score: float
    stage: str
    rank: int


class FilingRAG:
    def __init__(self, sections: list[FilingSection], config: RagConfig,
                 embedder: Embedder | None = None, reranker=None):
        self.sections = sections
        self.config = config
        self.chunks: list[Chunk] = chunk_sections(sections, config.chunker, **config.chunk_kwargs)
        texts = [c.text for c in self.chunks]

        self.embedder = embedder or get_embedder(corpus=texts)
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(texts)
        self.dense = DenseIndex(self.embedder).build(texts)
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.reranker = reranker if reranker is not None else (get_reranker() if config.rerank else None)

        self.tickers = sorted({c.ticker for c in self.chunks})
        self._ticker_of = np.array([c.ticker for c in self.chunks])

    # ---------------------------------------------------------------- query
    def resolve_ticker(self, query: str) -> str | None:
        """Scope the search to the company the question is about.

        In a multi-company corpus this is the same control as per-tenant
        filtering: without it, a question about NVDA's export-licence charge
        happily retrieves Microsoft's cloud commentary, which reads plausible
        and is wrong. Applied BEFORE ranking, never after -- post-filtering
        silently shrinks top-k.
        """
        upper = query.upper()
        for t in self.tickers:
            if re.search(rf"\b{re.escape(t)}\b", upper):
                return t
        aliases = {"APPLE": "AAPL", "MICROSOFT": "MSFT", "NVIDIA": "NVDA",
                   "JPMORGAN": "JPM", "JP MORGAN": "JPM", "PFIZER": "PFE"}
        for name, tick in aliases.items():
            if name in upper and tick in self.tickers:
                return tick
        return None

    def retrieve(self, query: str, ticker: str | None = None) -> list[Hit]:
        cfg = self.config
        bm = self.bm25.score(query)
        dn = self.dense.score(query)

        if cfg.retrieval == "bm25":
            base = bm
        elif cfg.retrieval == "dense":
            base = dn
        else:
            base = reciprocal_rank_fusion([bm, dn])

        scope = ticker or (self.resolve_ticker(query) if cfg.ticker_filter else None)
        if scope:
            base = np.where(self._ticker_of == scope, base, -np.inf)

        n_cand = max(cfg.candidates, cfg.top_k)
        cand = [int(i) for i in np.argsort(-base)[:n_cand] if np.isfinite(base[i])]
        if not cand:
            return []

        if cfg.diversify and len(cand) > cfg.top_k:
            qv = self.embedder.encode([query], is_query=True)[0]
            cand = mmr(qv, self.dense.matrix, cand, k=max(cfg.top_k, 10), lam=0.7)

        if self.reranker is not None:
            scored = sorted(
                zip(cand, self.reranker.score(query, [self.chunks[i].text for i in cand])),
                key=lambda t: -t[1],
            )[: cfg.top_k]
            return [Hit(self.chunks[i], float(s), "rerank", r) for r, (i, s) in enumerate(scored, 1)]

        return [Hit(self.chunks[i], float(base[i]), cfg.retrieval, r)
                for r, i in enumerate(cand[: cfg.top_k], 1)]

    def build_context(self, hits: list[Hit], token_budget: int = 2000) -> tuple[str, list[dict]]:
        """Numbered sources, best first.

        Ordering is deliberate: attention over a long context is strongest at
        the beginning and end, so the top-ranked chunk leads rather than
        landing wherever the retriever emitted it.
        """
        parts, sources, used = [], [], 0
        for i, h in enumerate(hits, 1):
            cost = max(1, len(h.chunk.text) // 4)
            if used + cost > token_budget:
                break
            parts.append(f"[{i}] ({h.chunk.citation})\n{h.chunk.text}")
            sources.append({
                "n": i, "doc_id": h.chunk.doc_id, "ticker": h.chunk.ticker,
                "item": h.chunk.item, "title": h.chunk.title,
                "filing_date": h.chunk.filing_date, "chunk_id": h.chunk.chunk_id,
                "url": h.chunk.url, "score": round(h.score, 4),
            })
            used += cost
        return "\n\n".join(parts), sources

    def query(self, question: str, ticker: str | None = None) -> dict:
        t0 = time.perf_counter()
        hits = self.retrieve(question, ticker=ticker)
        context, sources = self.build_context(hits)
        abstain = (not hits) or (
            self.config.abstain_below is not None and hits[0].score < self.config.abstain_below)
        return {
            "question": question,
            "hits": hits,
            "context": context,
            "sources": sources,
            "retrieved_doc_ids": [h.chunk.doc_id for h in hits],
            "retrieved_tickers": [h.chunk.ticker for h in hits],
            "abstain": abstain,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


# --- the ablation ladder ---------------------------------------------------
# V0 reproduces what rag_embeddings.py actually did: 500-WORD windows over
# whatever text made it past the 50KB truncation, dense-only retrieval, top-3.
BASELINE = RagConfig(
    name="V0 baseline (original pipeline)",
    chunker="word_window", chunk_kwargs={"size": 500, "overlap": 50},
    retrieval="dense", candidates=3, rerank=False, top_k=3,
)

LADDER: list[RagConfig] = [
    BASELINE,
    RagConfig(name="V1 + section-aware chunking",
              chunker="section_aware", retrieval="dense", candidates=5, rerank=False, top_k=5),
    RagConfig(name="V2 + hybrid BM25/dense (RRF)",
              chunker="section_aware", retrieval="hybrid", candidates=5, rerank=False, top_k=5),
    RagConfig(name="V3 + cross-encoder rerank",
              chunker="section_aware", retrieval="hybrid", candidates=40, rerank=True, top_k=5),
    RagConfig(name="V4 + ticker scoping + MMR",
              chunker="section_aware", retrieval="hybrid", candidates=40, rerank=True,
              diversify=True, top_k=5, ticker_filter=True,
              # calibrated by `python cli.py calibrate`; see README for why a
              # single reranker score is a weak abstention signal.
              abstain_below=0.30),
]

DIAGNOSTICS: list[RagConfig] = [
    RagConfig(name="dense only (bi-encoder)", chunker="section_aware", retrieval="dense",
              candidates=5, rerank=False, top_k=5),
    RagConfig(name="BM25 only (lexical)", chunker="section_aware", retrieval="bm25",
              candidates=5, rerank=False, top_k=5),
    RagConfig(name="hybrid RRF", chunker="section_aware", retrieval="hybrid",
              candidates=5, rerank=False, top_k=5),
]
