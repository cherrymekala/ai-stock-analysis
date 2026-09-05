"""SEC filing parsing.

This module exists because the original `sec_filings.extract_key_sections` was
silently indexing the table of contents. Three defects compounded:

1. `fetch_filing_text` capped the document at 50,000 characters. A large 10-K
   runs 200-500KB, so the cap kept roughly the cover page, the TOC and the
   start of Item 1. Item 1A (Risk Factors) and Item 7 (MD&A) begin well past
   the cut and were never in the indexed text at all.

2. The section regexes used a lazy `(.*?)` between an item heading and the next
   one. Every item heading appears at least twice in a filing -- once in the
   table of contents and once at the real section -- and lazy matching always
   binds to the first, so the "MD&A section" was a 135-character TOC row:
       "and Analysis of Financial Condition and Results of Operations 21 ..."

3. Entity decoding replaced `&nbsp;` but not the numeric form `&#160;`, which
   is what EDGAR actually emits, so heading text was littered with entities and
   never matched a heading pattern cleanly.

The fix below decodes entities properly, keeps the whole document, finds every
item heading, and then discards the table of contents by density: a TOC packs
many distinct item headings into a very short span, whereas real sections are
spread across the document.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

# Item headings we care about, in filing order.
ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}

HEADING = re.compile(r"Item\s+(\d{1,2}[A-C]?)\s*\.", re.IGNORECASE)

# A table of contents packs this many distinct item headings into this many
# characters. Real section headings are hundreds of paragraphs apart.
TOC_WINDOW_CHARS = 4000
TOC_DENSITY = 6


def html_to_text(raw: str) -> str:
    """Strip tags and decode entities properly.

    `html.unescape` handles named AND numeric entities (&#160;, &#8217;), which
    the original string-replace chain did not. Block-level tags become spaces
    so words either side of a tag boundary do not fuse together.
    """
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace(" ", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split())


def find_item_headings(text: str) -> list[tuple[str, int]]:
    """All (item_key, position) pairs, document order, TOC entries removed."""
    hits = [(m.group(1).upper(), m.start()) for m in HEADING.finditer(text)]
    if not hits:
        return []

    keep: list[tuple[str, int]] = []
    for i, (key, pos) in enumerate(hits):
        # how many DISTINCT item headings sit within a short window around this one?
        near = {k for k, p in hits if abs(p - pos) <= TOC_WINDOW_CHARS // 2}
        if len(near) >= TOC_DENSITY:
            continue                     # dense cluster -> table of contents
        keep.append((key, pos))

    # a filing occasionally repeats a heading (e.g. a cross-reference); keep the
    # occurrence that opens the longest span, which is the real section body.
    best: dict[str, int] = {}
    for idx, (key, pos) in enumerate(keep):
        nxt = keep[idx + 1][1] if idx + 1 < len(keep) else len(text)
        if key not in best or (nxt - pos) > (_span_of(keep, best[key], len(text))):
            best[key] = pos
    return sorted(best.items(), key=lambda kv: kv[1])


def _span_of(keep: list[tuple[str, int]], pos: int, end: int) -> int:
    for i, (_, p) in enumerate(keep):
        if p == pos:
            return (keep[i + 1][1] if i + 1 < len(keep) else end) - p
    return 0


def extract_sections(text: str, min_chars: int = 400) -> dict[str, dict[str, Any]]:
    """Return {item_key: {title, text, start, end}} for the real section bodies."""
    headings = find_item_headings(text)
    out: dict[str, dict[str, Any]] = {}
    for i, (key, start) in enumerate(headings):
        end = headings[i + 1][1] if i + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        if len(body) < min_chars or key not in ITEM_TITLES:
            continue
        out[key] = {
            "item": key,
            "title": ITEM_TITLES[key],
            "text": body,
            "start": start,
            "end": end,
            "n_chars": len(body),
        }
    return out


# --------------------------------------------------------------------------
# documents and chunks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FilingSection:
    doc_id: str          # AAPL-10-K-2025-10-31-item7
    ticker: str
    filing_type: str
    filing_date: str
    item: str
    title: str
    text: str
    url: str

    @property
    def label(self) -> str:
        return f"{self.ticker} {self.filing_type} ({self.filing_date}) Item {self.item} — {self.title}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ticker: str
    item: str
    title: str
    filing_date: str
    text: str
    url: str = ""
    metadata: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def citation(self) -> str:
        return f"{self.ticker} {self.filing_date} Item {self.item}"


def sections_to_documents(ticker: str, filing_type: str, filing_date: str, url: str,
                          sections: dict[str, dict]) -> list[FilingSection]:
    return [
        FilingSection(
            doc_id=f"{ticker}-{filing_type}-{filing_date}-item{s['item'].lower()}",
            ticker=ticker, filing_type=filing_type, filing_date=filing_date,
            item=s["item"], title=s["title"], text=s["text"], url=url,
        )
        for s in sorted(sections.values(), key=lambda s: s["start"])
    ]
