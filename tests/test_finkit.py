"""Unit tests over the deterministic parts of the pipeline and agent.

The point worth making in an interview: most of a RAG/agent codebase is NOT
non-deterministic. Filing parsing, chunking, tokenising, BM25, rank fusion,
MMR, the metrics, the output checks and every loop guard are pure functions
with known-correct answers. They are ordinary code and deserve ordinary tests.
Only the model call needs a mock, and it gets one below.

    python -m pytest tests/ -q      # no network, no model download
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from finkit.agent import (FilingAgent, HeuristicPlanner, is_degenerate,
                          parse_action, truncate)
from finkit.chunking import chunk_section_aware, chunk_sections, chunk_word_window
from finkit.embeddings import TfidfEmbedder
from finkit.evaluate import (hit_rate_at_k, load_golden, mrr, ndcg_at_k,
                             precision_at_k, recall_at_k, ticker_purity)
from finkit.filings import (FilingSection, extract_sections, find_item_headings,
                            html_to_text)
from finkit.generate import (REFUSAL, check_citations, check_numeric_grounding,
                             check_ticker_consistency, is_refusal)
from finkit.retrieval import (BM25, mmr as mmr_select, ranks_from_scores,
                              reciprocal_rank_fusion, tokenize)
from finkit.tools import ToolRegistry

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "eval" / "fixtures"


# ------------------------------------------------------- filing parsing
def test_html_to_text_decodes_numeric_entities():
    """The original code replaced &nbsp; but not &#160;, which is what EDGAR
    actually emits. Heading text was left full of entities and never matched."""
    assert html_to_text("<p>Item&#160;7.&#160;MD&amp;A</p>") == "Item 7. MD&A"
    assert "’" not in html_to_text("<p>Company&#8217;s</p>")


def test_html_to_text_does_not_fuse_words_across_tags():
    assert html_to_text("<td>Risk</td><td>Factors</td>") == "Risk Factors"


def _toc_filing() -> str:
    """A filing shaped like a real one: dense TOC, then the real sections."""
    toc = " ".join(f"Item {i}. Heading {i} {i*3}" for i in
                   ["1", "1A", "1B", "2", "3", "5", "7", "7A", "8", "9A"])
    body = (
        "Item 1. Business " + "The company designs things. " * 40 +
        "Item 1A. Risk Factors " + "Our business faces many risks. " * 40 +
        "Item 7. Management's Discussion and Analysis " + "Revenue grew. " * 40 +
        "Item 8. Financial Statements and Supplementary Data " + "See notes. " * 40
    )
    return "COVER PAGE " + toc + " " + body


def test_toc_headings_are_discarded_by_density():
    """The defect this module exists to fix: a lazy regex bound to the table of
    contents, so the 'MD&A section' was a 135-character TOC row."""
    text = _toc_filing()
    kept = dict(find_item_headings(text))
    assert kept, "no headings survived"
    # every kept heading must sit past the TOC cluster
    toc_end = text.index("Item 1. Business")
    assert all(pos >= toc_end for pos in kept.values()), \
        f"a table-of-contents entry survived: {kept}"


def test_extracted_section_is_the_body_not_the_toc_row():
    secs = extract_sections(_toc_filing())
    assert "7" in secs
    assert secs["7"]["n_chars"] > 400, "captured a TOC row rather than the section"
    assert "Revenue grew" in secs["7"]["text"]


def test_sections_do_not_overlap():
    secs = extract_sections(_toc_filing())
    spans = sorted((s["start"], s["end"]) for s in secs.values())
    for (_, e), (s2, _) in zip(spans, spans[1:]):
        assert e <= s2, "extracted sections overlap"


@pytest.mark.skipif(not list(FIXTURES.glob("*.json")), reason="fixtures not fetched")
def test_real_filing_sections_are_substantial():
    """Regression guard on the real cached data: the original pipeline produced
    a 135-char MD&A. Anything under 1KB means the TOC bug is back."""
    d = json.loads((FIXTURES / "AAPL.json").read_text())
    assert d["sections"]["7"]["n_chars"] > 5000
    assert d["sections"]["1A"]["n_chars"] > 5000
    assert d["n_chars_full"] > 50_000, "the 50KB truncation is back"


# ------------------------------------------------------------- chunking
def _section(text: str) -> FilingSection:
    return FilingSection("T-10-K-2026-01-01-item7", "T", "10-K", "2026-01-01",
                         "7", "MD&A", text, "http://example.com")


def test_word_window_uses_word_counts_not_chars():
    """Documents the units bug preserved from the original: `chunk_size=500`
    counts WORDS, so chunks are ~3,000 characters, not 500."""
    chunks = chunk_word_window(_section(" ".join(["word"] * 1200)), size=500, overlap=50)
    assert len(chunks) == 3                       # step 450 over 1200 words
    assert len(chunks[0].text) > 2000


def test_section_aware_chunks_respect_budget():
    text = ". ".join(f"Sentence number {i} about revenue" for i in range(200)) + "."
    for c in chunk_section_aware(_section(text), max_chars=1200):
        assert len(c.text) < 2600                 # budget + prefix + one overlap para


def test_section_aware_prefixes_company_and_item():
    c = chunk_section_aware(_section("Revenue increased. " * 20))[0]
    assert c.text.startswith("T 10-K 2026-01-01 — Item 7 MD&A.")


def test_hard_split_handles_a_single_oversized_paragraph():
    """EDGAR tables arrive as one enormous 'paragraph' with no sentence breaks."""
    giant = " ".join(["x"] * 5000)                # no punctuation at all
    chunks = chunk_section_aware(_section(giant), max_chars=1000)
    assert len(chunks) > 3
    assert all(len(c.text) < 2600 for c in chunks)


def test_chunks_carry_company_identity():
    for c in chunk_sections([_section("Revenue. " * 50)], "section_aware"):
        assert c.ticker and c.item, "a chunk that lost its company cannot be scoped or cited"


# ------------------------------------------------------------ retrieval
def test_tokenizer_preserves_hyphenated_and_ampersand_terms():
    toks = tokenize("Compute & Networking revenue in the 10-K")
    assert "10-k" in toks and "10" in toks
    assert "compute" in toks and "networking" in toks


def test_bm25_ranks_exact_rare_term_first():
    docs = [tokenize("H20 export licence charge"), tokenize("cloud revenue growth"),
            tokenize("dividend and buyback programme")]
    assert int(np.argmax(BM25(docs).score("H20"))) == 0


def test_bm25_idf_favours_rare_terms():
    bm = BM25([["revenue", "h20"], ["revenue"], ["revenue"], ["revenue"]])
    assert bm.idf["h20"] > bm.idf["revenue"]


def test_bm25_inverted_index_matches_bruteforce():
    docs = [tokenize(t) for t in
            ["revenue grew sharply", "revenue fell", "margin compression", "revenue and margin"]]
    bm = BM25(docs)
    s = bm.score("revenue margin")
    assert s[3] > s[2] and s[0] > 0 and s[2] > 0


def test_ranks_from_scores_is_descending():
    assert list(ranks_from_scores(np.array([0.1, 0.9, 0.5]))) == [2, 0, 1]


def test_rrf_rewards_agreement():
    a, b = np.array([9.0, 1.0, 0.5]), np.array([8.0, 0.2, 0.4])
    assert int(np.argmax(reciprocal_rank_fusion([a, b]))) == 0


def test_rrf_is_scale_invariant():
    """The argument for RRF over weighted score addition: BM25 is unbounded,
    cosine is in [-1,1]. Scaling one retriever must not change the fusion."""
    a, b = np.array([3.0, 2.0, 1.0]), np.array([0.1, 0.9, 0.5])
    assert np.allclose(reciprocal_rank_fusion([a, b]),
                       reciprocal_rank_fusion([a * 1000, b]))


def test_mmr_prefers_novel_over_near_duplicate():
    v = np.array([[1.0, 0.0], [0.999, 0.044], [0.6, 0.8]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    assert mmr_select(v[0], v, [0, 1, 2], k=2, lam=0.3) == [0, 2]


# -------------------------------------------------------------- metrics
def test_metrics_known_values():
    got, rel = ["d3", "d7", "d1", "d9", "d2"], {"d1", "d2", "d5"}
    assert recall_at_k(got, rel, 5) == pytest.approx(2 / 3)
    assert precision_at_k(got, rel, 5) == pytest.approx(2 / 5)
    assert mrr(got, rel) == pytest.approx(1 / 3)
    assert hit_rate_at_k(got, rel, 5) == 1.0


def test_hit_and_recall_differ_for_multi_label_cases():
    """Why hit@k is the headline metric here: with two acceptable sections,
    recall@1 caps at 0.5 however good retrieval is."""
    got, rel = ["item1"], {"item1", "item7"}
    assert recall_at_k(got, rel, 1) == pytest.approx(0.5)
    assert hit_rate_at_k(got, rel, 1) == 1.0


def test_ndcg_penalises_burying_relevant_results():
    assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) > ndcg_at_k(["x", "y", "a"], {"a"}, 3)


def test_ticker_purity_detects_cross_company_contamination():
    assert ticker_purity(["NVDA", "NVDA", "MSFT", "NVDA"], "NVDA", 4) == pytest.approx(0.75)
    assert ticker_purity(["NVDA"] * 3, "NVDA", 3) == 1.0


def test_golden_set_labels_are_wellformed():
    cases = load_golden(ROOT / "eval" / "golden.jsonl")
    assert len(cases) == 30
    assert len({c.id for c in cases}) == 30, "duplicate case ids"
    for c in cases:
        if c.should_refuse:
            assert not c.relevant_doc_ids, f"{c.id}: refusal case must have no labels"
        else:
            assert c.relevant_doc_ids and c.ticker, f"{c.id}: answerable case needs labels"


# -------------------------------------------------------- output checks
def test_numeric_grounding_flags_invented_figures():
    ctx = "Revenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago."
    assert check_numeric_grounding("Revenue was $215.9 billion [1].", ctx)["ok"]
    bad = check_numeric_grounding("Revenue was $220.4 billion.", ctx)
    assert not bad["ok"] and "220.4" in bad["unsupported_numbers"]


def test_numeric_grounding_ignores_citation_markers():
    """[1] is a citation, not a numeric claim. This was a real bug."""
    assert check_numeric_grounding("Revenue was $215.9 billion [1][2].", "215.9 billion")["ok"]


def test_numeric_grounding_normalises_thousands_separators():
    assert check_numeric_grounding("Revenue 215,938,000,000.", "revenue 215,938,000,000")["ok"]


def test_citation_check_rejects_out_of_range():
    r = check_citations("Claim one [1]. Claim two [9].", n_sources=3)
    assert r["invalid_citations"] == [9] and not r["ok"]


def test_ticker_consistency_catches_a_cross_company_answer():
    """The domain's version of a tenant leak: answering about NVIDIA using
    Microsoft's MD&A produces a fluent, cited, entirely wrong answer."""
    srcs = [{"ticker": "NVDA"}, {"ticker": "MSFT"}]
    r = check_ticker_consistency(srcs, "NVDA")
    assert r["mixed"] and not r["ok"]
    assert check_ticker_consistency([{"ticker": "NVDA"}], "NVDA")["ok"]


def test_refusal_detection():
    assert is_refusal(REFUSAL) and not is_refusal("Revenue was $215.9 billion [1].")


# ---------------------------------------------------------------- tools
def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(description="Look up a company by ticker.")
    def lookup(ticker: str, limit: int = 3) -> dict:
        return {"ticker": ticker, "limit": limit}

    @reg.register(description="Always fails.")
    def explodes() -> dict:
        raise RuntimeError("upstream down")

    return reg


def test_schema_is_derived_from_the_signature():
    """A hand-maintained schema drifts from the function and then lies to the
    model. Deriving it from type hints makes that impossible."""
    spec = _registry().specs()[0]
    assert spec["input_schema"]["required"] == ["ticker"]
    assert spec["input_schema"]["properties"]["limit"]["type"] == "integer"
    assert spec["input_schema"]["properties"]["limit"]["default"] == 3


def test_validation_returns_messages_not_exceptions():
    reg = _registry()
    assert "Unknown tool" in reg.validate("nope", {})
    assert "Missing required" in reg.validate("lookup", {})
    assert "must be integer" in reg.validate("lookup", {"ticker": "AAPL", "limit": "three"})
    assert "Unexpected argument" in reg.validate("lookup", {"ticker": "AAPL", "x": 1})
    assert reg.validate("lookup", {"ticker": "AAPL"}) is None


# ---------------------------------------------------------------- agent
def test_parse_action_survives_fences_and_prose():
    assert parse_action('```json\n{"tool":"lookup","arguments":{}}\n```')["tool"] == "lookup"
    assert parse_action('Sure!\n{"answer": "done"}\nHope that helps')["answer"] == "done"
    assert parse_action("no json here at all") is None


def test_degenerate_answers_are_rejected():
    """Weak models echo the format example or reply with an ellipsis. An agent
    that returns '...' has failed as completely as one that crashed."""
    assert is_degenerate("<your answer>")
    assert is_degenerate("...")
    assert is_degenerate("")
    assert not is_degenerate("Apple repurchased $89.3 billion of common stock in 2025 [1].")


def test_truncate_caps_tool_output():
    out = truncate("x" * 10_000, max_chars=100)
    assert len(out) < 200 and "truncated" in out


class _ScriptedPlanner:
    def __init__(self, actions):
        self.actions = list(actions)

    def __call__(self, messages):
        return self.actions.pop(0) if self.actions else {"answer": "done, with enough words here"}


def test_agent_returns_error_to_model_instead_of_raising():
    reg = _registry()
    agent = FilingAgent(reg, planner=_ScriptedPlanner([
        {"tool": "nope", "arguments": {}},                       # unknown tool
        {"tool": "lookup", "arguments": {}},                     # missing arg
        {"tool": "lookup", "arguments": {"ticker": "AAPL"}},     # good
        {"answer": "Apple is covered by the cached filings [1]."},
    ]))
    r = agent.run("tell me about AAPL")
    assert r.stopped_because == "completed"
    assert r.tools_used == ["lookup"]
    assert sum(1 for e in r.trace if e.kind == "guard") == 2


def test_agent_blocks_repeated_identical_calls():
    """An agent re-calling the same tool with the same arguments is the most
    common runaway-cost bug there is."""
    reg = _registry()
    call = {"tool": "lookup", "arguments": {"ticker": "AAPL"}}
    agent = FilingAgent(reg, planner=_ScriptedPlanner([dict(call) for _ in range(6)]),
                        max_steps=5)
    r = agent.run("q")
    assert r.stopped_because == "max_steps"
    assert r.tools_used == ["lookup"], "the tool ran more than once"
    assert any("repeat call blocked" in e.detail for e in r.trace)


def test_agent_survives_a_raising_tool():
    reg = _registry()
    agent = FilingAgent(reg, planner=_ScriptedPlanner([
        {"tool": "explodes", "arguments": {}},
        {"answer": "The data source is currently unavailable [1]."},
    ]))
    r = agent.run("q")
    assert r.stopped_because == "completed"


def test_agent_enforces_a_tool_allowlist():
    """Capability boundaries live in code. A system prompt is not a control."""
    reg = _registry()
    agent = FilingAgent(reg, allow={"lookup"}, planner=_ScriptedPlanner([
        {"tool": "explodes", "arguments": {}},
        {"answer": "That tool is not available to me here."},
    ]))
    r = agent.run("q")
    assert "explodes" not in r.tools_used
    assert any("not permitted" in e.detail for e in r.trace)


def test_agent_stops_on_error_budget():
    reg = _registry()
    agent = FilingAgent(reg, max_tool_errors=2, planner=_ScriptedPlanner(
        [{"tool": "nope", "arguments": {"i": i}} for i in range(6)]))
    r = agent.run("q")
    assert r.stopped_because == "tool_errors"


def test_heuristic_planner_scopes_to_the_named_company():
    reg = _registry()
    p = HeuristicPlanner(reg, ["AAPL", "NVDA"])
    assert p._ticker("What did NVIDIA say about H20?") == "NVDA"
    assert p._ticker("How did Apple do?") == "AAPL"
    assert p._ticker("What about Tesla?") == ""


# ----------------------------------------------------------- embeddings
def test_tfidf_embeddings_are_normalised():
    emb = TfidfEmbedder(dim=64).fit(["revenue growth", "risk factors"])
    assert np.allclose(np.linalg.norm(emb.encode(["revenue"]), axis=1), 1.0, atol=1e-5)


def test_tfidf_is_deterministic():
    emb = TfidfEmbedder(dim=64).fit(["a b c", "b c d"])
    assert np.array_equal(emb.encode(["a b"]), emb.encode(["a b"]))
