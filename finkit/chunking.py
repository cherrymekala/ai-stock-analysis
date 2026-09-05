"""Chunking. Two strategies so the ablation can price the naive one.

`word_window` reproduces what `rag_embeddings.chunk_text` did: 500-word windows
with 50-word overlap, no awareness of section boundaries. Note that 500 *words*
is roughly 3,000 characters -- far larger than the 500 the parameter name
suggests, and large enough that a single chunk routinely spans three unrelated
risk factors. The units bug is preserved here on purpose so the ablation
measures the real baseline rather than a flattering reconstruction.

`section_aware` splits inside an already-extracted 10-K item, breaks on
paragraph boundaries, and prefixes every chunk with ticker, fiscal date and
item title. That prefix is the cheapest quality win available: an isolated
paragraph about "our data center segment" is otherwise indistinguishable
between NVDA and MSFT once it has been embedded.
"""
from __future__ import annotations

import re

from .filings import Chunk, FilingSection


def chunk_word_window(sec: FilingSection, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """The baseline. Word-count window, no structure, no prefix."""
    words = sec.text.split()
    step = max(1, size - overlap)
    out: list[Chunk] = []
    for i, start in enumerate(range(0, max(1, len(words)), step)):
        piece = " ".join(words[start : start + size]).strip()
        if not piece:
            continue
        out.append(Chunk(
            chunk_id=f"{sec.doc_id}#w{i}", doc_id=sec.doc_id, ticker=sec.ticker,
            item=sec.item, title=sec.title, filing_date=sec.filing_date,
            text=piece, url=sec.url,
        ))
        if start + size >= len(words):
            break
    return out


def _paragraphs(text: str) -> list[str]:
    """EDGAR text arrives as one long line, so paragraph structure has to be
    inferred. Sentence boundaries followed by a capitalised word are the most
    reliable signal available after tag stripping."""
    parts = re.split(r"(?<=[.:;])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_section_aware(sec: FilingSection, max_chars: int = 1200,
                        overlap_sentences: int = 1) -> list[Chunk]:
    prefix = f"{sec.ticker} {sec.filing_type} {sec.filing_date} — Item {sec.item} {sec.title}. "
    out: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    idx = 0

    def flush() -> None:
        nonlocal buf, buf_len, idx
        if not buf:
            return
        out.append(Chunk(
            chunk_id=f"{sec.doc_id}#s{idx}", doc_id=sec.doc_id, ticker=sec.ticker,
            item=sec.item, title=sec.title, filing_date=sec.filing_date,
            text=prefix + " ".join(buf), url=sec.url,
        ))
        idx += 1
        buf = buf[-overlap_sentences:] if overlap_sentences else []
        buf_len = sum(len(b) for b in buf)

    for para in _paragraphs(sec.text):
        # a single paragraph can exceed the budget on its own (EDGAR tables and
        # long enumerated risk factors do this constantly); hard-split those
        # rather than emitting a chunk many times the target size.
        for piece in _hard_split(para, max_chars):
            if buf and buf_len + len(piece) > max_chars:
                flush()
            buf.append(piece)
            buf_len += len(piece)
    flush()
    return out


def _hard_split(para: str, max_chars: int) -> list[str]:
    if len(para) <= max_chars:
        return [para]
    words, out, cur, cur_len = para.split(), [], [], 0
    for w in words:
        if cur and cur_len + len(w) + 1 > max_chars:
            out.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_sections(sections: list[FilingSection], strategy: str = "section_aware", **kw) -> list[Chunk]:
    fn = {"word_window": chunk_word_window, "section_aware": chunk_section_aware}[strategy]
    out: list[Chunk] = []
    for s in sections:
        out.extend(fn(s, **kw))
    return out
