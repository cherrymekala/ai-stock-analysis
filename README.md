# stock-analysis — grounded SEC-filing QA and a tool-calling agent, built to be measured

Retrieval and agent tooling over real SEC 10-K filings, written without a framework so every stage can be ablated, unit-tested and explained.

The original pipeline is preserved as **V0** and still runs, because the point of this repo is the measured distance between V0 and V4.

**Headline: hit@1 went 0.667 → 0.875 and MRR 0.743 → 0.925 across five configurations on a 30-question labelled set.** But the number that mattered most was one the original pipeline had no way to see: **company attribution, which the cross-encoder reranker halved to 0.500 before ticker scoping fixed it.**

Before any of that, three parsing defects meant the system had been indexing the table of contents.

---

## What was actually broken

`rag_embeddings.py` was embedding SEC filings that had been through `sec_filings.extract_key_sections`. That function returned a **135-character "MD&A"** for Apple's 10-K. Three defects compounded:

**1. The document was cut at 50,000 characters.** Apple's 10-K is 220,566 characters of text; Pfizer's is 742,216. The real MD&A starts at character 121,703. Item 1A and Item 7 were never in the indexed text at all — roughly 21% of the filing survived, and it was the cover page, the TOC and the start of Item 1.

**2. Lazy regexes bound to the table of contents.** Every item heading appears at least twice in a filing — once in the TOC, once at the real section. `(.*?)` always binds to the first. What the "MD&A section" actually contained:

```
'and Analysis of Financial Condition and Results of Operations 21 Item 7A.
 Quantitative and Qualitative Disclosures About Market Risk 27'
```

That is a table-of-contents row with page numbers in it.

**3. Entity decoding missed the numeric forms.** The code replaced `&nbsp;` but EDGAR emits `&#160;`, and `&#8217;` for apostrophes. Heading text arrived littered with entities and never matched a heading pattern cleanly.

`finkit/filings.py` fixes all three: `html.unescape` for entities, no truncation, and TOC removal **by density** — a table of contents packs many distinct item headings into a very short span, while real sections are spread across the document. Result on the same filing:

| Section | Before | After |
|---|---:|---:|
| Item 1 Business | 16,050 chars | 16,050 chars |
| Item 1A Risk Factors | *never extracted* | 68,044 chars |
| Item 7 MD&A | **135 chars** | **18,017 chars** |

---

## Results

30 labelled questions (24 answerable, 6 out-of-scope) over five real 10-Ks — AAPL, MSFT, NVDA, JPM, PFE — cached under `eval/fixtures/`. Local `all-MiniLM-L6-v2` embeddings, exact inner-product search, `ms-marco-MiniLM-L-6-v2` cross-encoder reranking. Reproduce with `python cli.py compare`.

### Retrieval

| Configuration | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | Company purity | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 baseline (original pipeline) | 319 | 0.667 | 0.833 | 0.833 | 0.743 | 0.731 | 0.875 | 5.8 |
| V1 + section-aware chunking | 1131 | 0.708 | 0.792 | 0.833 | 0.753 | 0.752 | 0.925 | 0.5 |
| V2 + hybrid BM25/dense (RRF) | 1131 | 0.708 | 0.875 | 0.917 | 0.795 | 0.823 | 0.867 | 0.5 |
| V3 + cross-encoder rerank | 1131 | 0.708 | 0.917 | 0.917 | 0.799 | 0.826 | **0.783** | 452.9 |
| V4 + ticker scoping + MMR | 1131 | **0.875** | **0.958** | **1.000** | **0.925** | **0.937** | **1.000** | 109.1 |

**hit@k is the headline, not recall@k**, and the reason is a label-semantics point: several questions are answerable from more than one section (NVIDIA names its two segments in both Item 1 and Item 7). Either grounds the answer, so the labels mean *any of these*. `recall@1` mechanically caps at 0.5 for a two-label case however good retrieval is. `hit@k` asks the question that matters: did usable evidence reach the top-k?

### End-to-end answer quality

Deterministic checks — no LLM judge, no API key, no drift.

| | V0 | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|---:|
| Answer pass rate | 0.367 | 0.467 | 0.533 | 0.333 | **0.833** |
| Fact reached the context | 0.542 | 0.750 | **0.958** | 0.875 | 0.792 |
| Numeric grounding | 1.000 | 0.792 | 0.958 | 1.000 | **1.000** |
| **Company attribution correct** | 0.792 | 0.875 | 0.708 | **0.500** | **1.000** |
| Abstains on out-of-scope | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** |

### Where the gains came from

| Query type | n | V0 hit@1 | V4 hit@1 |
|---|---:|---:|---:|
| governance | 1 | 0.000 | 1.000 |
| pfe | 5 | 0.600 | 1.000 |
| aapl | 8 | 0.375 | 0.750 |
| numeric | 17 | 0.588 | 0.824 |
| paraphrase | 5 | 0.400 | 0.600 |
| exact-term | 5 | 1.000 | 1.000 |

---

## Five things I learned that I did not expect

### 1. Better ranking made the system worse, and only one metric caught it

V3 adds a cross-encoder reranker. Every retrieval metric improves or holds: hit@3 goes 0.875 → 0.917, nDCG 0.823 → 0.826.

**Company attribution collapses from 0.708 to 0.500.** Half the answers were being assembled from the wrong company's filing.

The cause is that a cross-encoder is company-blind by construction. It scores passage-query semantic similarity, and Microsoft's discussion of its reportable segments is an *excellent* semantic match for "what are NVIDIA's operating segments" — better, sometimes, than NVIDIA's own phrasing. Widening the candidate pool to 40 to give the reranker room made it strictly more likely to surface a confident, fluent, correctly-cited answer about the wrong company.

End-to-end pass rate at V3 (0.333) is **below the original baseline** (0.367). A pipeline judged on hit@k and nDCG alone would have shipped that as an improvement.

The fix is not a better reranker. It is resolving the company from the query and filtering **before** ranking — the same control as per-tenant filtering, and for the same reason. Post-filtering would not do: it silently shrinks top-k and leaks the existence of documents outside the scope.

### 2. The parsing bug was worth more than every retrieval technique combined

Hybrid retrieval, reranking and MMR together moved hit@1 by 0.208. Fixing the extraction moved Item 7 from 135 characters to 18,017 — from "no information exists" to "the answer is in the corpus at all". No amount of retriever tuning recovers a section that was never indexed.

I would not have found it by looking at retrieval metrics, because the baseline scored a respectable 0.667 hit@1 on the questions it *could* answer. I found it by reading what the chunker actually produced. **Look at your data before you tune your model** is advice everyone gives and nobody follows.

### 3. Hybrid retrieval barely moved the average and I kept it anyway

V1 → V2 adds BM25 alongside dense, fused with RRF. hit@1 does not move at all (0.708 → 0.708), though hit@3 improves.

Running each retriever alone on identical chunking:

| Retriever (same chunking, same k, no rerank) | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| dense only (bi-encoder) | 0.646 | 0.750 | 0.753 |
| BM25 only (lexical) | 0.583 | 0.833 | 0.724 |
| hybrid RRF | 0.646 | 0.875 | 0.795 |

```
dense misses at rank 1: [q05, q12, q13, q16, q18, q21, q22]
BM25  misses at rank 1: [q02, q05, q06, q12, q13, q14, q16, q18, q20]
both  miss:             [q05, q12, q13, q16, q18]
dense fixes for BM25:   [q02, q06, q14, q20]    BM25 fixes for dense: [q21, q22]
```

They fail on **different questions**. BM25 recovers Pfizer's exclusivity-expiry and billion-dollar-product questions, where the exact figures anchor the match. Dense recovers NVIDIA's revenue growth and Microsoft Cloud, where the question and filing share little vocabulary. RRF is rank-based, so it needs no score normalisation — BM25 is unbounded and shifts with the corpus while cosine sits in [-1, 1], and any weighting tuned on today's filings is wrong after the next 10-K lands.

**RRF's job is to widen the candidate pool, not to produce the final ordering.** Judging it on aggregate hit@1 at the fusion step measures the wrong thing.

### 4. My own metric was lying to me — twice

**First:** `recall@1` capped at 0.5 for the four questions with two acceptable sections, which dragged the `cross-company-risk` tag to 0.625 in *every* configuration and made it look like nothing helped. Switching to hit@1 showed it had been 1.000 throughout. The retrieval was fine; the metric was wrong.

**Second, and worse:** the company-attribution check originally resolved the expected ticker only when ticker filtering was enabled. So V0–V3 — the configurations that do no scoping — could never fail the scoping check, and all reported 1.000. The measurement was switched off by the very setting it existed to evaluate. Fixing it is what surfaced finding #1.

Both were my bugs, in the harness, not the pipeline. It is worth stating plainly that **an eval harness needs the same scepticism as the system it measures**, and that a metric which cannot fail is not measuring anything.

### 5. The agent's tool selection was fine; its synthesis was not

Running the agent against a free-tier model, tool selection worked reliably — it called `available_tickers`, then `search_filings` scoped to the right company with a sensibly rewritten query. The final synthesis turn returned:

```json
{"answer": "<your answer>"}
```

The model had echoed the format example instead of filling it in. Then, after a prompt fix, `{"answer": "..."}`.

The loop now treats a degenerate answer the same way it treats a bad tool name — as a **correctable error returned to the model**, not a crash and not something to ship. On retry the same model produced a properly grounded answer quoting $89.3 billion of repurchases and $15.4 billion of dividends.

The asymmetry is the interesting part. **Choosing among six tools is a far easier task than writing a grounded, cited answer** — which is exactly the argument for routing with a small model and synthesising with a large one, rather than picking one model for the whole loop.

---

## Architecture

```
fetch_fixtures.py ──▶ EDGAR → full text (no truncation) → entity decode
                  ──▶ item headings → TOC removed by density → sections
                  ──▶ cached JSON  (offline, reproducible eval)

index  ──▶ chunk  ┌ word_window:   500-WORD windows (~3,000 chars)      (V0)
                  └ section_aware: paragraph packing inside a 10-K item,
                                   tables hard-split, ticker+date+item
                                   prefixed to every chunk              (V1+)
       ──▶ dense: all-MiniLM-L6-v2 → exact inner product on L2-normalised
                  vectors (== FAISS IndexFlatIP)
       ──▶ lexical: BM25 (k1=1.5, b=0.75) with an inverted index;
                    hyphenated and &-joined terms kept intact

query  ──▶ score both ──▶ RRF ──▶ resolve ticker, filter BEFORE ranking
       ──▶ MMR over 40 candidates ──▶ cross-encoder rerank ──▶ top 5
       ──▶ numbered context, best chunk first, token-budgeted
       ──▶ generate (grounded prompt, citations required, abstention)
       ──▶ check: numeric grounding · citation validity · company purity

agent  ──▶ ReAct loop over 6 tools; guards on steps, wall clock, repeat
           calls, tool errors, allowlist, and degenerate answers
```

**Why no framework.** `db.as_retriever()` hides the chunking strategy, the value of k, the absence of reranking and the absence of any evaluation. None of those are separable, ablatable or testable through it. Every table above exists because each stage is a function I can swap.

**Why exact search.** 1,131 chunks. Exact inner-product search has no training step and no recall loss; below roughly a million vectors it is the right default. HNSW is where you go when latency forces it, and then you measure the recall you traded rather than assuming none.

**Why fixtures.** EDGAR is rate-limited and filings are superseded. A benchmark that changes underneath you is not a benchmark. Every number here is measured against cached JSON, reproducible with no network round trip.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fetch_fixtures.py                 # cache 10-Ks (needs network, once)
python verify_golden.py                  # assert every label is real
python cli.py ingest                     # chunking statistics per strategy
python cli.py ask "What charge did NVIDIA take on H20 inventory?"
python cli.py eval                       # the full V0 → V4 ablation
python cli.py eval --config V4 --generation
python cli.py compare                    # the tables above
python cli.py calibrate                  # abstention threshold sweep
python cli.py agent "How did Apple return capital to shareholders in 2025?"
python -m pytest tests/ -q               # 43 tests, 0.08s, no network
```

No API key is required for anything above. `OPENROUTER_API_KEY` plus `--llm` switches the generator and the agent planner from extractive passages to a real model; every metric in this README is measured without one, because retrieval quality and the deterministic output checks do not need a model.

```
finkit/
  filings.py     EDGAR parsing; entity decoding, TOC removal by density
  chunking.py    word_window (baseline) vs section_aware
  embeddings.py  sentence-transformers → OpenAI → TF-IDF fallback
  retrieval.py   BM25 with an inverted index, dense index, RRF, MMR
  rerank.py      cross-encoder, with a lexical fallback
  generate.py    grounded prompt, citations, deterministic output checks
  evaluate.py    hit@k, recall@k, MRR, nDCG, company purity, calibration
  pipeline.py    config-driven assembly; the V0–V4 ladder lives here
  tools.py       6 agent tools; JSON schema derived from type hints
  agent.py       ReAct loop, guards, planners, parallel tool execution
eval/
  fixtures/      5 cached 10-Ks (AAPL, MSFT, NVDA, JPM, PFE)
  golden.jsonl   30 labelled questions, tagged by query type
tests/           43 unit tests over the deterministic stages
```

The original daily-report application (`main.py`, `streamlit_app.py`, `discord_bot.py`, `sec_filings.py`) is unchanged and still works.

---

## Honest limitations

- **The extractor degrades on financial-sector filings.** JPMorgan's 10-K yields only 2 of 7 sections, because bank filings incorporate most items by reference into exhibits rather than stating them inline. The density heuristic handles standard industrial and technology 10-Ks well and does not generalise to that structure. One question in the golden set covers JPM, which is not enough to measure the gap.
- **30 questions is a small evaluation set.** A 0.208 gain in hit@1 is 5 questions out of 24 — directionally solid, but I would not defend a 0.02 difference on this set, and the per-tag rows with n=1 are anecdotes, not measurements.
- **The corpus is 5 filings.** Retrieval over 1,131 chunks from 5 companies is a much easier discrimination problem than a real filings archive. Company purity in particular would be harder with 500 companies, not easier.
- **Section-level relevance labels** cannot distinguish retrieving the right paragraph from retrieving the right section's boilerplate.
- **`fact_in_answer` is measured against the extractive stub** (top passages truncated to 300 characters), so it understates what a real generator would produce. `fact_in_context` is the honest retrieval-level number; run with `--llm` for a generation number that means something.
- **No LLM-as-judge.** Everything here is deterministic on purpose. The deterministic checks catch invented figures, broken citations and cross-company contamination — they do **not** catch an unsupported claim that contains no numbers. That is what a judge is for, and it needs calibrating against human labels before it can be trusted.
- **Abstention rests on a single reranker score.** Calibration shows the in-scope and out-of-scope score distributions overlap by 1.91; catching all 6 out-of-scope questions costs 3 false refusals out of 24. That is an acceptable operating point under a 5:1 cost model and it is still a weak signal.

## Next

1. **A second abstention signal** — rank-1-to-rank-2 margin, or an entailment check of each answer sentence against its cited context. The overlap above is why this is first.
2. **LLM-as-judge for faithfulness**, binary with a rubric and reasoning-before-verdict, calibrated against human labels and reported with Cohen's kappa rather than raw agreement.
3. **A layout-aware parser** for financial-sector filings and for the tables in Item 8, which the current text-only path cannot represent.
4. **Scale the corpus** to 50+ filings and re-measure company purity, which is the metric most likely to degrade.
5. **Native tool-calling** for the agent where the provider offers it, replacing JSON-in-prose parsing.
