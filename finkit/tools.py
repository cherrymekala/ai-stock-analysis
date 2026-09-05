"""Tool registry for the filings agent.

The model never executes anything. It emits JSON matching a schema you gave it;
this module validates that JSON and your code runs the function. Keeping the
schema derived from the Python signature means the two cannot drift apart --
a hand-maintained schema eventually lies to the model.

Tool descriptions and parameter names ARE prompt engineering: the model picks
tools by reading them, so they are written for the model, not for the reader.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, get_type_hints

PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean",
              list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    schema: dict
    timeout_s: float = 20.0
    network: bool = False        # needs a live call; may fail or be slow


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, *, description: str, timeout_s: float = 20.0, network: bool = False):
        def deco(fn: Callable) -> Callable:
            hints = get_type_hints(fn)
            props, required = {}, []
            for pname, param in inspect.signature(fn).parameters.items():
                ptype = hints.get(pname, str)
                props[pname] = {"type": PY_TO_JSON.get(ptype, "string")}
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
                else:
                    props[pname]["default"] = param.default
            self._tools[fn.__name__] = Tool(
                name=fn.__name__, description=description, fn=fn,
                schema={"type": "object", "properties": props, "required": required},
                timeout_s=timeout_s, network=network,
            )
            return fn
        return deco

    def specs(self, allow: set[str] | None = None) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.schema}
                for t in self._tools.values() if allow is None or t.name in allow]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def validate(self, name: str, args: dict) -> str | None:
        """Return an error STRING, never raise.

        The error goes back to the model as a tool result so it can correct
        itself. Errors are feedback, not exceptions -- that is what makes an
        agent self-healing rather than merely crash-prone.
        """
        tool = self.get(name)
        if tool is None:
            return f"Unknown tool '{name}'. Available: {', '.join(self.names())}."
        schema = tool.schema
        missing = [r for r in schema["required"] if r not in args]
        if missing:
            return f"Missing required argument(s) {missing}. Schema: {json.dumps(schema)}"
        extra = [a for a in args if a not in schema["properties"]]
        if extra:
            return f"Unexpected argument(s) {extra}. Allowed: {list(schema['properties'])}"
        for a, v in args.items():
            want = schema["properties"][a]["type"]
            ok = {"string": str, "integer": int, "number": (int, float),
                  "boolean": bool, "array": list, "object": dict}[want]
            if not isinstance(v, ok):
                return f"Argument '{a}' must be {want}, got {type(v).__name__} ({v!r})."
        return None


def build_registry(rag, fixtures: Path) -> ToolRegistry:
    """Wire the existing project functions up as agent tools.

    Cached tools (filings, financials, insiders) read the fixtures, so the
    agent is deterministic and offline. Live tools are marked `network=True`
    and degrade to a clear error rather than an exception when unavailable --
    an agent whose tool raises loses the whole run; one that gets an error
    message can re-plan.
    """
    reg = ToolRegistry()
    cache: dict[str, dict] = {}

    def _fixture(ticker: str) -> dict | None:
        t = ticker.upper().strip()
        if t not in cache:
            p = fixtures / f"{t}.json"
            cache[t] = json.loads(p.read_text()) if p.exists() else {}
        return cache[t] or None

    @reg.register(description=(
        "Search the cached 10-K filings for passages answering a question. "
        "Returns numbered excerpts with the company, filing date and 10-K item "
        "they came from. Use this for anything the filing text would state: "
        "strategy, risk factors, segment descriptions, MD&A commentary, "
        "reported figures. ticker must be one of the available tickers."))
    def search_filings(query: str, ticker: str = "", top_k: int = 5) -> dict:
        r = rag.query(query, ticker=(ticker.upper() or None))
        if r["abstain"] or not r["sources"]:
            return {"results": [], "note": "no sufficiently relevant passage found"}
        return {"results": [
            {"n": s["n"], "ticker": s["ticker"], "item": s["item"],
             "filing_date": s["filing_date"], "title": s["title"],
             "excerpt": " ".join(h.chunk.text.split())[:700], "score": s["score"]}
            for s, h in zip(r["sources"][:top_k], r["hits"][:top_k])]}

    @reg.register(description=(
        "Get reported revenue, net income and diluted EPS with year-over-year "
        "growth, from SEC XBRL company facts. Use this for headline financials "
        "rather than searching the filing text, because these are structured "
        "and exact."))
    def get_financials(ticker: str) -> dict:
        d = _fixture(ticker)
        if not d:
            return {"error": f"No cached filing for '{ticker}'. Available: {available_tickers()['tickers']}"}
        return {"ticker": d["ticker"], "filing_date": d["filing_date"],
                "financials": d.get("financials") or {"note": "no XBRL facts cached"}}

    @reg.register(description=(
        "List recent insider transactions (SEC Form 4 filings) for a company, "
        "with dates and links. Use to check whether insiders have been filing "
        "recently. Note this returns filing metadata, not buy/sell amounts."))
    def get_insider_activity(ticker: str, limit: int = 5) -> dict:
        d = _fixture(ticker)
        if not d:
            return {"error": f"No cached filing for '{ticker}'. Available: {available_tickers()['tickers']}"}
        ins = d.get("insiders") or {}
        return {"ticker": d["ticker"], "summary": ins.get("summary"),
                "recent_transactions": (ins.get("recent_transactions") or [])[:limit]}

    @reg.register(description=(
        "List which companies and filings are available to search. Call this "
        "first if you are unsure whether a company is covered."))
    def available_tickers() -> dict:
        out = []
        for p in sorted(fixtures.glob("*.json")):
            d = json.loads(p.read_text())
            out.append({"ticker": d["ticker"], "filing_type": d["filing_type"],
                        "filing_date": d["filing_date"],
                        "items": sorted(d["sections"].keys())})
        return {"tickers": [f["ticker"] for f in out], "filings": out}

    @reg.register(description=(
        "Get live technical indicators (RSI, SMA50, SMA200) from Alpha Vantage. "
        "Requires network and an API key; returns an error if unavailable."),
        network=True, timeout_s=25.0)
    def get_technical_indicators(ticker: str) -> dict:
        try:
            from main import fetch_alpha_vantage_indicators
            data = fetch_alpha_vantage_indicators(ticker.upper())
            return data or {"error": "no indicator data returned"}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    @reg.register(description=(
        "Get a live market snapshot: last price, day range, volume and moving "
        "averages. Requires network; returns an error if unavailable."),
        network=True, timeout_s=25.0)
    def get_market_snapshot(ticker: str) -> dict:
        try:
            from main import fetch_market_snapshot
            data = fetch_market_snapshot(ticker.upper())
            return data or {"error": "no market data returned"}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    return reg
