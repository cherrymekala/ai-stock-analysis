#!/usr/bin/env python3
"""Cache real SEC filings to disk so evaluation is offline and reproducible.

    python fetch_fixtures.py                    # default ticker set
    python fetch_fixtures.py --tickers AAPL,JPM

Why fixtures rather than live calls in the eval: EDGAR is rate-limited, filings
are superseded, and a benchmark that changes underneath you is not a benchmark.
Every number in the README is measured against the JSON cached here, so anyone
can reproduce it without a network round trip.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from finkit.filings import extract_sections, html_to_text
from sec_filings import HEADERS, fetch_insider_trading_data, fetch_sec_filings, fetch_xbrl_financials

FIXTURES = Path(__file__).resolve().parent / "eval" / "fixtures"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "PFE"]
KEEP_ITEMS = {"1", "1A", "1C", "3", "5", "7", "7A"}   # narrative sections
SEC_DELAY = 0.4                                        # EDGAR asks for <10 req/s


def fetch_one(ticker: str) -> dict | None:
    print(f"\n{ticker}")
    filings = fetch_sec_filings(ticker, ["10-K"], count=1)
    time.sleep(SEC_DELAY)
    if not filings:
        print("  no 10-K found")
        return None
    f = filings[0]
    print(f"  {f['filing_type']} {f['date']}  {f['url']}")

    resp = requests.get(f["url"], headers=HEADERS, timeout=60)
    resp.raise_for_status()
    time.sleep(SEC_DELAY)
    text = html_to_text(resp.text)
    sections = {k: v for k, v in extract_sections(text).items() if k in KEEP_ITEMS}
    print(f"  {len(text):,} chars -> {len(sections)} sections: "
          + ", ".join(f"Item {k} ({v['n_chars']:,})" for k, v in sections.items()))

    financials = fetch_xbrl_financials(ticker) or {}
    time.sleep(SEC_DELAY)
    insiders = fetch_insider_trading_data(ticker, limit=10, days=180) or {}
    time.sleep(SEC_DELAY)

    return {
        "ticker": ticker,
        "filing_type": f["filing_type"],
        "filing_date": f["date"],
        "url": f["url"],
        "n_chars_full": len(text),
        "sections": sections,
        "financials": financials,
        "insiders": insiders,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    args = ap.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    ok = 0
    for t in [x.strip().upper() for x in args.tickers.split(",") if x.strip()]:
        try:
            data = fetch_one(t)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        if not data:
            continue
        (FIXTURES / f"{t}.json").write_text(json.dumps(data, indent=1))
        ok += 1
    print(f"\ncached {ok} filings to {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
