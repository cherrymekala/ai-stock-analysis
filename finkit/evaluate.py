"""Evaluation.

Layers measured separately -- retrieval, then generation -- because a single
end-to-end score tells you something is broken but never which stage broke it.

Relevance is labelled at SECTION level (ticker + 10-K item), not chunk level.
Chunk ids change whenever the chunking strategy changes, so chunk labels are
not portable across the very ablation being run. "Did any chunk from the right
section reach the top-k" is stable, and it is the question that matters for
grounding.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- retrieval

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    window = retrieved[:k]
    return sum(1 for d in window if d in relevant) / len(window)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    seen: set[str] = set()
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], 1):
        if d in relevant and d not in seen:
            seen.add(d)
            dcg += 1.0 / math.log2(i + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0


def ticker_purity(retrieved_tickers: list[str], expected: str | None, k: int) -> float:
    """Fraction of the top-k drawn from the right company.

    The metric the original pipeline had no way to see. A question about NVIDIA
    answered from Microsoft's MD&A is a confident, well-cited, wrong answer.
    """
    if not expected or not retrieved_tickers:
        return float("nan")
    window = retrieved_tickers[:k]
    return sum(1 for t in window if t == expected) / len(window)


# ------------------------------------------------------------------ harness

@dataclass
class EvalCase:
    id: str
    question: str
    ticker: str | None = None
    relevant_doc_ids: set[str] = field(default_factory=set)
    must_include: list[str] = field(default_factory=list)
    should_refuse: bool = False
    tags: list[str] = field(default_factory=list)
    note: str = ""


def load_golden(path: str | Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cases.append(EvalCase(
            id=d["id"], question=d["question"], ticker=d.get("ticker"),
            relevant_doc_ids=set(d.get("relevant_doc_ids") or []),
            must_include=d.get("must_include") or [],
            should_refuse=bool(d.get("should_refuse")),
            tags=d.get("tags") or [], note=d.get("note", ""),
        ))
    return cases


def evaluate_retrieval(rag, cases: list[EvalCase], k: int = 5) -> dict:
    per_case, latencies = [], []
    for c in cases:
        r = rag.query(c.question)
        latencies.append(r["latency_ms"])
        got, got_t = r["retrieved_doc_ids"], r["retrieved_tickers"]
        if c.should_refuse:
            per_case.append({"id": c.id, "tags": c.tags, "out_of_scope": True,
                             "abstained": r["abstain"], "retrieved": got[:k]})
            continue
        per_case.append({
            "id": c.id, "tags": c.tags, "out_of_scope": False,
            "retrieved_full": got,
            f"recall@{k}": recall_at_k(got, c.relevant_doc_ids, k),
            f"precision@{k}": precision_at_k(got, c.relevant_doc_ids, k),
            f"hit@{k}": hit_rate_at_k(got, c.relevant_doc_ids, k),
            "mrr": mrr(got, c.relevant_doc_ids),
            f"ndcg@{k}": ndcg_at_k(got, c.relevant_doc_ids, k),
            f"ticker_purity@{k}": ticker_purity(got_t, c.ticker, k),
            "retrieved": got[:k],
            "expected": sorted(c.relevant_doc_ids),
        })

    scored = [p for p in per_case if not p["out_of_scope"]]
    oos = [p for p in per_case if p["out_of_scope"]]
    agg = {m: round(statistics.mean(p[m] for p in scored), 3)
           for m in (f"recall@{k}", f"precision@{k}", f"hit@{k}", "mrr", f"ndcg@{k}")}
    # The k-curve, because recall@5 saturates and stops discriminating.
    #
    # hit@k is the PRIMARY metric here, not recall@k, and the reason is a label
    # semantics point worth stating: several questions are answerable from more
    # than one section (NVIDIA names its two segments in both Item 1 and Item
    # 7). Either one grounds the answer, so the labels mean "any of these", not
    # "all of these". recall@1 mechanically caps at 0.5 for a two-label case no
    # matter how good retrieval is, which would understate every configuration
    # equally and make the tag breakdown unreadable. hit@k asks the question
    # that actually matters: did usable evidence reach the top-k?
    for kk in (1, 3):
        agg[f"recall@{kk}"] = round(
            statistics.mean(recall_at_k(p["retrieved_full"], set(p["expected"]), kk) for p in scored), 3)
        agg[f"hit@{kk}"] = round(
            statistics.mean(hit_rate_at_k(p["retrieved_full"], set(p["expected"]), kk) for p in scored), 3)

    purities = [p[f"ticker_purity@{k}"] for p in scored if not math.isnan(p[f"ticker_purity@{k}"])]
    agg[f"ticker_purity@{k}"] = round(statistics.mean(purities), 3) if purities else float("nan")

    by_tag: dict[str, list[float]] = {}
    for p in scored:
        for t in p["tags"]:
            by_tag.setdefault(t, []).append(p[f"hit@{k}"])

    return {
        "n_scored": len(scored),
        "n_out_of_scope": len(oos),
        **agg,
        "abstain_rate_out_of_scope": round(sum(p["abstained"] for p in oos) / len(oos), 3) if oos else float("nan"),
        "p50_latency_ms": round(statistics.median(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[min(int(0.95 * len(latencies)), len(latencies) - 1)], 1),
        "recall_by_tag": {t: round(statistics.mean(v), 3) for t, v in sorted(by_tag.items())},
        "failures": [p for p in scored if p[f"hit@{k}"] < 1.0],
        "per_case": per_case,
    }


def evaluate_generation(rag, cases: list[EvalCase], generator=None) -> dict:
    """Deterministic faithfulness proxy: no judge, no key, no drift.

    "Faithful" here means three checkable things at once: every figure in the
    answer appears in the retrieved context, every citation resolves to a
    supplied source, and every source came from the company asked about. Those
    catch the three ways a filings answer goes wrong in practice. What they do
    NOT catch is a claim that is unsupported but contains no numbers -- that is
    what an LLM judge is for, and it needs calibrating before it can be trusted.
    """
    from .generate import answer_question, is_refusal

    rows = []
    for c in cases:
        ctx = rag.query(c.question)["context"].lower()
        a = answer_question(rag, c.question, generator=generator)
        if c.should_refuse:
            rows.append({"id": c.id, "kind": "refusal", "passed": is_refusal(a.text),
                         "answer": a.text[:90]})
            continue
        text_l = a.text.lower()
        contains = all(m.lower() in text_l for m in c.must_include)
        # Two different questions, and conflating them flatters or maligns the
        # wrong stage. `context_contains` asks whether RETRIEVAL put the fact in
        # front of the model -- measurable with no LLM at all. `contains` asks
        # whether the ANSWER states it, which depends entirely on the generator;
        # with the extractive stub (top passages truncated to 300 chars) it is
        # an artifact of that stub, not a property of the pipeline. Run with
        # --llm for a number that means something.
        context_contains = all(m.lower() in ctx for m in c.must_include)
        numeric_ok = a.checks.get("numeric", {}).get("ok", True)
        cites_ok = a.checks.get("citations", {}).get("ok", True)
        ticker_ok = a.checks.get("ticker", {}).get("ok", True)
        rows.append({
            "id": c.id, "kind": "answer",
            "passed": context_contains and numeric_ok and cites_ok and ticker_ok,
            "contains_required": contains, "context_contains": context_contains,
            "numeric_grounded": numeric_ok,
            "citations_ok": cites_ok, "ticker_ok": ticker_ok,
            "unsupported_numbers": a.checks.get("numeric", {}).get("unsupported_numbers", []),
            "tickers_in_context": a.checks.get("ticker", {}).get("tickers_in_context", []),
        })

    answers = [r for r in rows if r["kind"] == "answer"]
    return {
        "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
        "contains_required_rate": round(statistics.mean([r["contains_required"] for r in answers]), 3),
        "context_contains_rate": round(statistics.mean([r["context_contains"] for r in answers]), 3),
        "numeric_grounded_rate": round(statistics.mean([r["numeric_grounded"] for r in answers]), 3),
        "ticker_correct_rate": round(statistics.mean([r["ticker_ok"] for r in answers]), 3),
        "rows": rows,
    }


# ---------------------------------------------------------------- abstention

def calibrate_abstention(rag, cases: list[EvalCase], cost_ratio: float = 5.0) -> dict:
    """Choose the abstention threshold from labelled data, not intuition.

    Two error rates trade against each other: a missed refusal (an unanswerable
    question answered anyway -- the dangerous one) and a false refusal (an
    answerable question refused -- the annoying one). `cost_ratio` is how many
    false refusals one missed refusal is worth, and it is the only genuinely
    subjective input. Everything else is arithmetic.
    """
    in_scope, out_scope = [], []
    for c in cases:
        r = rag.query(c.question)
        top = r["hits"][0].score if r["hits"] else float("-inf")
        (out_scope if c.should_refuse else in_scope).append((c.id, top))

    candidates = sorted({round(s, 2) for _, s in in_scope + out_scope if s != float("-inf")})
    if not candidates:
        return {"error": "no scores"}

    rows = []
    for t in candidates:
        missed = [cid for cid, s in out_scope if s >= t]
        false_ref = [cid for cid, s in in_scope if s < t]
        rows.append({"threshold": t, "missed_refusals": len(missed),
                     "false_refusals": len(false_ref),
                     "missed_ids": missed, "false_ids": false_ref,
                     "cost": len(missed) * cost_ratio + len(false_ref)})

    recommended = min(rows, key=lambda r: (r["cost"], r["false_refusals"], r["threshold"]))
    lo_in = min(s for _, s in in_scope)
    hi_out = max(s for _, s in out_scope) if out_scope else float("-inf")
    return {
        "cost_ratio": cost_ratio,
        "in_scope_score_range": [round(lo_in, 3), round(max(s for _, s in in_scope), 3)],
        "out_scope_score_range": [round(min(s for _, s in out_scope), 3), round(hi_out, 3)] if out_scope else None,
        "separable": lo_in > hi_out,
        "overlap": round(hi_out - lo_in, 3) if out_scope else None,
        "recommended": recommended,
        "curve": rows,
    }
