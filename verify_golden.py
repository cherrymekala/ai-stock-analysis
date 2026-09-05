#!/usr/bin/env python3
"""Check every golden label before trusting any metric computed from it.

A labelled set is only as good as its labels. This asserts two things for each
answerable case: the section named in `relevant_doc_ids` exists in the cached
fixtures, and every `must_include` string actually appears in that section's
text. Without this, a typo in a label silently becomes a permanent false
negative that you then spend a day "fixing" in the retriever.
"""
from __future__ import annotations

import sys

from finkit.evaluate import load_golden
from finkit.pipeline import load_fixture_sections

def main() -> int:
    sections = {s.doc_id: s for s in load_fixture_sections()}
    cases = load_golden("eval/golden.jsonl")
    bad = 0
    for c in load_golden("eval/golden.jsonl"):
        if c.should_refuse:
            continue
        missing_docs = [d for d in c.relevant_doc_ids if d not in sections]
        if missing_docs:
            print(f"  {c.id}: UNKNOWN doc_id {missing_docs}")
            bad += 1
            continue
        pool = " ".join(sections[d].text for d in c.relevant_doc_ids).lower()
        missing = [m for m in c.must_include if m.lower() not in pool]
        if missing:
            print(f"  {c.id}: must_include not found in labelled section: {missing}")
            bad += 1
    n_ans = sum(1 for c in cases if not c.should_refuse)
    print(f"\n{len(cases)} cases ({n_ans} answerable, {len(cases)-n_ans} out-of-scope); "
          f"{bad} label problem(s)")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
