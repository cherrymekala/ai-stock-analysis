#!/usr/bin/env python3
"""finkit CLI — grounded SEC-filing QA, built to be measured.

    python cli.py ask "What charge did NVIDIA take on H20 inventory?"
    python cli.py eval                        # full ablation V0 -> V4
    python cli.py eval --config V4 --generation
    python cli.py compare                     # markdown tables for the README
    python cli.py calibrate                   # abstention threshold sweep
    python cli.py ingest                      # chunking statistics
    python cli.py agent "Is NVDA's margin story holding up?"

Everything runs against the filings cached under eval/fixtures by
`fetch_fixtures.py`, so results are offline and reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from finkit.chunking import chunk_sections
from finkit.embeddings import get_embedder
from finkit.evaluate import (calibrate_abstention, evaluate_generation,
                             evaluate_retrieval, hit_rate_at_k, load_golden)
from finkit.generate import answer_question, get_generator
from finkit.pipeline import (DIAGNOSTICS, LADDER, FilingRAG, RagConfig,
                             load_fixture_sections)
from finkit.rerank import get_reranker

load_dotenv()

ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDEN = ROOT / "eval" / "golden.jsonl"
DEFAULT_FIXTURES = ROOT / "eval" / "fixtures"
K = 5


def _build(config: RagConfig, fixtures: Path, shared: dict) -> FilingRAG:
    sections = shared.setdefault("sections", load_fixture_sections(fixtures))
    emb = shared.get("embedder") or shared.setdefault("embedder", get_embedder())
    rr = None
    if config.rerank:
        rr = shared.get("reranker") or shared.setdefault("reranker", get_reranker())
    return FilingRAG(sections, config, embedder=emb, reranker=rr)


# ----------------------------------------------------------------------- ask
def cmd_ask(args) -> int:
    shared: dict = {}
    cfg = next((c for c in LADDER if c.name.startswith(args.config)), LADDER[-1])
    rag = _build(cfg, Path(args.fixtures), shared)
    gen = get_generator("openrouter" if args.llm else "none")
    print(f"\nconfig: {cfg.name}   embedder: {rag.embedder.name}   "
          f"generator: {getattr(gen, 'model', 'extractive')}")
    scope = rag.resolve_ticker(args.question) if cfg.ticker_filter else None
    print(f"ticker scope: {scope or '(none — searching all filings)'}\n")

    a = answer_question(rag, args.question, generator=gen)
    print(a.text, "\n")
    if a.sources:
        print("SOURCES")
        for s in a.sources:
            print(f"  [{s['n']}] {s['ticker']} {s['filing_date']} Item {s['item']} "
                  f"— {s['title'][:44]}  (score {s['score']})")
    c = a.checks
    print("\nCHECKS")
    print(f"  numeric grounding : {'PASS' if c.get('numeric',{}).get('ok') else 'FAIL ' + str(c.get('numeric',{}).get('unsupported_numbers'))}")
    print(f"  citations         : {'PASS' if c.get('citations',{}).get('ok') else 'FAIL'}"
          f"   sentence rate {c.get('citations',{}).get('sentence_citation_rate')}")
    tk = c.get("ticker", {})
    print(f"  company purity    : {'PASS' if tk.get('ok') else 'FAIL'}   context tickers {tk.get('tickers_in_context')}")
    print(f"  latency           : {c.get('latency_ms')} ms")
    return 0


# ---------------------------------------------------------------------- eval
def cmd_eval(args) -> int:
    shared: dict = {}
    cases = load_golden(args.golden)
    configs = LADDER if not args.config else [c for c in LADDER if c.name.startswith(args.config)]
    if not configs:
        print(f"no config matching {args.config!r}")
        return 2

    n_out = sum(1 for c in cases if c.should_refuse)
    print(f"\nfixtures: {args.fixtures}   cases: {len(cases)} "
          f"({len(cases)-n_out} answerable, {n_out} out-of-scope)\n")

    results = []
    for cfg in configs:
        rag = _build(cfg, Path(args.fixtures), shared)
        r = evaluate_retrieval(rag, cases, k=K)
        r["config"], r["n_chunks"] = cfg.name, len(rag.chunks)
        if args.generation:
            r["generation"] = evaluate_generation(
                rag, cases, generator=get_generator("openrouter" if args.llm else "none"))
        results.append(r)
        print(f"  {cfg.name:34s} chunks={r['n_chunks']:5d}  hit@1={r['hit@1']:.3f}  "
              f"hit@3={r['hit@3']:.3f}  hit@{K}={r[f'hit@{K}']:.3f}  mrr={r['mrr']:.3f}  "
              f"purity={r[f'ticker_purity@{K}']:.3f}  p50={r['p50_latency_ms']:.1f}ms")

    if args.config and results:
        r = results[0]
        print(f"\n  precision@{K}={r[f'precision@{K}']:.3f}  recall@{K}={r[f'recall@{K}']:.3f}  "
              f"ndcg@{K}={r[f'ndcg@{K}']:.3f}  abstain on out-of-scope={r['abstain_rate_out_of_scope']}")
        print(f"\n  recall by tag: {json.dumps(r['recall_by_tag'])}")
        if r["failures"]:
            print(f"\n  {len(r['failures'])} FAILURES (no relevant section in top-{K}):")
            for f in r["failures"]:
                print(f"    {f['id']}  expected={f['expected']}\n         got={f['retrieved']}")
        else:
            print(f"\n  no retrieval failures at k={K}.")
        if "generation" in r:
            g = r["generation"]
            print(f"\n  generation: pass_rate={g['pass_rate']}  "
                  f"fact_in_context={g['context_contains_rate']}  "
                  f"fact_in_answer={g['contains_required_rate']}  "
                  f"numeric_grounded={g['numeric_grounded_rate']}  "
                  f"ticker_correct={g['ticker_correct_rate']}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "per_case"} for r in results], indent=2, default=str))
        print(f"\n  wrote {args.json}")
    return 0


# ------------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    shared: dict = {}
    cases = load_golden(args.golden)
    rows = []
    for cfg in LADDER:
        rag = _build(cfg, Path(args.fixtures), shared)
        rows.append((cfg.name, len(rag.chunks), evaluate_retrieval(rag, cases, k=K)))
    base, final = rows[0][2], rows[-1][2]

    print(f"\n| Configuration | Chunks | Hit@1 | Hit@3 | Hit@{K} | MRR | nDCG@{K} | Company purity | p50 ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, n, r in rows:
        print(f"| {name} | {n} | {r['hit@1']:.3f} | {r['hit@3']:.3f} | {r[f'hit@{K}']:.3f} | "
              f"{r['mrr']:.3f} | {r[f'ndcg@{K}']:.3f} | {r[f'ticker_purity@{K}']:.3f} | {r['p50_latency_ms']:.1f} |")
    print(f"\n**V0 → V4: hit@1 {base['hit@1']:.3f} → {final['hit@1']:.3f} "
          f"({final['hit@1']-base['hit@1']:+.3f}), MRR {base['mrr']:.3f} → {final['mrr']:.3f} "
          f"({final['mrr']-base['mrr']:+.3f}), company purity "
          f"{base[f'ticker_purity@{K}']:.3f} → {final[f'ticker_purity@{K}']:.3f}.**")

    def by_tag_at1(res):
        acc: dict[str, list[float]] = {}
        for p in res["per_case"]:
            if p["out_of_scope"]:
                continue
            r1 = hit_rate_at_k(p["retrieved_full"], set(p["expected"]), 1)
            for t in p["tags"]:
                acc.setdefault(t, []).append(r1)
        return acc
    b1, f1 = by_tag_at1(base), by_tag_at1(final)
    print("\n| Query type | n | V0 hit@1 | V4 hit@1 |")
    print("|---|---:|---:|---:|")
    for tag in sorted(b1):
        print(f"| {tag} | {len(b1[tag])} | {sum(b1[tag])/len(b1[tag]):.3f} | {sum(f1[tag])/len(f1[tag]):.3f} |")

    diag = [(c.name, evaluate_retrieval(_build(c, Path(args.fixtures), shared), cases, k=K))
            for c in DIAGNOSTICS]
    print("\n| Retriever (same chunking, same k, no rerank) | Recall@1 | Recall@3 | MRR |")
    print("|---|---:|---:|---:|")
    for name, r in diag:
        print(f"| {name} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['mrr']:.3f} |")

    def missed(res):
        return {p["id"] for p in res["per_case"] if not p["out_of_scope"]
                and hit_rate_at_k(p["retrieved_full"], set(p["expected"]), 1) < 1}
    md, mb = missed(diag[0][1]), missed(diag[1][1])
    print(f"\n- dense misses at rank 1: {sorted(md)}")
    print(f"- BM25  misses at rank 1: {sorted(mb)}")
    print(f"- **both** miss: {sorted(md & mb)}")
    print(f"- dense fixes for BM25: {sorted(mb - md)}   BM25 fixes for dense: {sorted(md - mb)}")
    return 0


# ----------------------------------------------------------------- calibrate
def cmd_calibrate(args) -> int:
    shared: dict = {}
    cases = load_golden(args.golden)
    cfg = replace(next(c for c in LADDER if c.name.startswith("V4")), abstain_below=None)
    rag = _build(cfg, Path(args.fixtures), shared)
    r = calibrate_abstention(rag, cases, cost_ratio=args.cost_ratio)

    n_out = sum(1 for c in cases if c.should_refuse)
    n_in = len(cases) - n_out
    print(f"\n{n_in} answerable / {n_out} out-of-scope, scored by the cross-encoder\n")
    print(f"  in-scope  top-score range: {r['in_scope_score_range']}")
    print(f"  out-scope top-score range: {r['out_scope_score_range']}")
    if r["separable"]:
        print("  the two distributions do not overlap — a clean threshold exists")
    else:
        print(f"  DISTRIBUTIONS OVERLAP by {r['overlap']:.2f}: some answerable questions score")
        print("  lower than some out-of-scope ones. No threshold separates them cleanly.")
    print(f"  cost model: 1 missed refusal == {r['cost_ratio']:g} false refusals\n")
    print("  threshold | missed refusals | false refusals")
    print("  ----------|-----------------|---------------")
    for row in [x for x in r["curve"] if x["missed_refusals"] or x["false_refusals"]][:14]:
        print(f"  {row['threshold']:9.2f} | {row['missed_refusals']:15d} | {row['false_refusals']:14d}")

    rec = r["recommended"]
    print(f"\n  RECOMMENDED abstain_below = {rec['threshold']:.2f}   (cost {rec['cost']:.0f})")
    print(f"    catches {n_out - rec['missed_refusals']}/{n_out} out-of-scope; "
          f"wrongly refuses {rec['false_refusals']}/{n_in} answerable {rec['false_ids']}")
    if not r["separable"]:
        clean = min((x["false_refusals"] for x in r["curve"] if x["missed_refusals"] == 0), default=0)
        print(f"\n  CAVEAT: a single reranker score is a weak abstention signal. Refusing every")
        print(f"  out-of-scope question costs {clean}/{n_in} false refusals. The fix is a second")
        print("  signal — rank-1-to-rank-2 margin, or an entailment check — not a finer threshold.")
    return 0


# -------------------------------------------------------------------- ingest
def cmd_ingest(args) -> int:
    sections = load_fixture_sections(Path(args.fixtures))
    print(f"\n{len(sections)} sections from {len({s.ticker for s in sections})} filings\n")
    for strat, kw in (("word_window", {"size": 500, "overlap": 50}), ("section_aware", {})):
        ch = chunk_sections(sections, strat, **kw)
        sizes = [len(c.text) for c in ch]
        print(f"  {strat:14s} {len(ch):5d} chunks   mean {sum(sizes)//len(sizes):5d} chars   "
              f"min {min(sizes):5d}   max {max(sizes):5d}")
    print()
    for s in sections:
        print(f"  {s.doc_id:34s} Item {s.item:<3} {s.title[:40]:42s} {len(s.text):>8,} chars")
    return 0


# --------------------------------------------------------------------- agent
def cmd_agent(args) -> int:
    from finkit.agent import FilingAgent
    from finkit.tools import build_registry
    shared: dict = {}
    rag = _build(LADDER[-1], Path(args.fixtures), shared)
    reg = build_registry(rag, Path(args.fixtures))
    llm = None
    if args.llm:
        from finkit.generate import OpenRouterGenerator
        llm = OpenRouterGenerator(args.model) if args.model else get_generator("openrouter")
    agent = FilingAgent(reg, llm=llm, max_steps=args.max_steps)
    result = agent.run(args.question)
    for e in result.trace:
        ms = f" {e.ms:6.1f}ms" if e.ms else " " * 9
        print(f"  step {e.step} [{e.kind:5s}]{ms}  {e.detail}")
    print(f"\n{result.output}\n")
    print(f"  steps={result.steps}  stop={result.stopped_because}  "
          f"tools={result.tools_used}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    fx, gl = str(DEFAULT_FIXTURES), str(DEFAULT_GOLDEN)

    a = sub.add_parser("ask"); a.add_argument("question")
    a.add_argument("--config", default="V4"); a.add_argument("--fixtures", default=fx)
    a.add_argument("--llm", action="store_true", help="use OpenRouter instead of extractive output")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval")
    e.add_argument("--config", default=None); e.add_argument("--fixtures", default=fx)
    e.add_argument("--golden", default=gl); e.add_argument("--generation", action="store_true")
    e.add_argument("--llm", action="store_true"); e.add_argument("--json", default=None)
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("compare")
    c.add_argument("--fixtures", default=fx); c.add_argument("--golden", default=gl)
    c.set_defaults(func=cmd_compare)

    cal = sub.add_parser("calibrate")
    cal.add_argument("--fixtures", default=fx); cal.add_argument("--golden", default=gl)
    cal.add_argument("--cost-ratio", type=float, default=5.0)
    cal.set_defaults(func=cmd_calibrate)

    i = sub.add_parser("ingest"); i.add_argument("--fixtures", default=fx)
    i.set_defaults(func=cmd_ingest)

    ag = sub.add_parser("agent"); ag.add_argument("question")
    ag.add_argument("--fixtures", default=fx); ag.add_argument("--llm", action="store_true")
    ag.add_argument("--model", default=None, help="OpenRouter model id for the planner")
    ag.add_argument("--max-steps", type=int, default=8)
    ag.set_defaults(func=cmd_agent)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
