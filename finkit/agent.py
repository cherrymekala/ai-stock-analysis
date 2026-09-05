"""A tool-calling agent over the filings corpus.

An agent is an LLM in a loop with tools and a termination condition. The model
decides WHAT to do; this code decides whether it is allowed and actually does
it. Everything below the LLM call is the part that keeps it from becoming an
unbounded invoice: step caps, a wall clock, repeat-call detection, an error
budget, and argument validation that returns a correctable message instead of
raising.

Two planners are supported so the agent is testable without a key:

  LLMPlanner       — a real ReAct loop. Tool calls are requested as JSON, which
                     is parsed leniently and validated strictly. Native
                     tool-calling APIs are better and should be preferred where
                     the provider offers them; this path exists because the
                     free OpenRouter models used here do not expose one
                     reliably.
  HeuristicPlanner — a deterministic rule-based planner. Not intelligent, but
                     it exercises the identical loop, which is what makes the
                     guards unit-testable offline.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .generate import CITE, check_numeric_grounding
from .tools import ToolRegistry

SYSTEM_PROMPT = """\
You are an equity research assistant with tools over a small set of cached SEC
10-K filings. Ground every claim in tool output.

To call a tool, reply with ONLY a JSON object:
  {"tool": "TOOL_NAME_HERE", "arguments": {"ticker": "AAPL"}}
To answer, reply with ONLY a JSON object:
  {"answer": "Apple repurchased $89.3 billion of common stock during 2025 [1]."}

The second example shows the SHAPE. Replace the text with your real answer.
Never emit a placeholder such as <your answer> or TOOL_NAME_HERE literally.

Rules:
- Call available_tickers first if you are unsure a company is covered.
- Quote figures exactly as the tool returned them. Never compute a new number.
- Attribute every figure to the company and fiscal period it came from.
- If the tools cannot answer, say so plainly in your answer. Do not guess.
- You are not a financial adviser. Do not recommend buying or selling.

TOOLS:
{tools}
"""


@dataclass
class TraceEvent:
    step: int
    kind: str          # plan | tool | guard | final
    detail: str
    ms: float = 0.0


@dataclass
class AgentResult:
    output: str
    steps: int
    trace: list[TraceEvent] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    stopped_because: str = "completed"
    evidence: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# planners
# --------------------------------------------------------------------------

# Small models frequently echo the format example back instead of filling it
# in. Detect that and treat it as a correctable error rather than shipping
# "<your answer>" to a user.
PLACEHOLDER = re.compile(r"<[a-z_ ]+>|TOOL_NAME_HERE|YOUR_ANSWER", re.IGNORECASE)


def is_degenerate(value: str) -> bool:
    """True for a non-answer: an echoed placeholder, an ellipsis, empty text.

    Weak models do this constantly, and an agent that returns "..." to a user
    has failed just as completely as one that crashed -- but silently, which is
    worse. Cheap to check, so check it.
    """
    v = value.strip()
    if not v or len(v) < 15:
        return True
    if not re.search(r"[A-Za-z]{3}", v):          # no real words
        return True
    return bool(PLACEHOLDER.fullmatch(v) or (len(v) < 60 and PLACEHOLDER.search(v)))


def parse_action(text: str) -> dict | None:
    """Lenient JSON extraction. Models wrap JSON in prose and code fences.

    The right answer in production is the provider's structured-output or
    tool-calling mode, which guarantees schema conformance instead of making
    you regex prose. This is the fallback when that is not on offer.
    """
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict) and ("tool" in obj or "answer" in obj):
                        return obj
                except json.JSONDecodeError:
                    pass
    return None


class LLMPlanner:
    """Real ReAct loop over a chat model.

    Observed limitation, worth stating plainly: on the free-tier models
    available here, TOOL SELECTION works well -- the model reliably picks
    `available_tickers`, then `search_filings` scoped to the right company with
    a sensibly rewritten query -- but the final SYNTHESIS turn frequently
    degenerates to a placeholder. That asymmetry is the whole argument for
    routing with a small model and synthesising with a large one: choosing
    among six tools is a far easier task than writing a grounded, cited answer.
    Use --model to point this at something stronger.
    """
    def __init__(self, client, registry: ToolRegistry):
        self.client = client
        # NOT str.format(): the prompt contains literal JSON braces that
        # format() would try to interpret as replacement fields.
        self.system = SYSTEM_PROMPT.replace(
            "{tools}", json.dumps(registry.specs(), indent=1))

    def __call__(self, messages: list[dict]) -> dict | None:
        raw = self.client.generate_raw(self.system, messages)
        return parse_action(raw) or {"answer": raw.strip()[:1200]}


class HeuristicPlanner:
    """Deterministic planner: enough to exercise the loop, offline and testable.

    It is a decision table, not a model. Being explicit about that matters --
    a rule-based planner is often the correct production choice for a known
    workflow, and calling it an agent does not make it smarter.
    """
    ALIASES = {"APPLE": "AAPL", "MICROSOFT": "MSFT", "NVIDIA": "NVDA",
               "JPMORGAN": "JPM", "JP MORGAN": "JPM", "PFIZER": "PFE"}

    def __init__(self, registry: ToolRegistry, known_tickers: list[str]):
        self.registry = registry
        self.known = known_tickers

    def _ticker(self, q: str) -> str:
        u = q.upper()
        for t in self.known:
            if re.search(rf"\b{t}\b", u):
                return t
        for name, t in self.ALIASES.items():
            if name in u and t in self.known:
                return t
        return ""

    def __call__(self, messages: list[dict]) -> dict | None:
        question = messages[0]["content"]
        done = {m.get("tool") for m in messages if m.get("role") == "tool"}
        tick = self._ticker(question)
        low = question.lower()

        if not tick and "available_tickers" not in done:
            return {"tool": "available_tickers", "arguments": {}}
        if not tick:
            return {"answer": ("I could not identify a company in that question that is "
                               "covered by the cached filings.")}
        wants_financials = any(w in low for w in
                               ("revenue", "net income", "eps", "earnings", "profit", "growth"))
        if wants_financials and "get_financials" not in done:
            return {"tool": "get_financials", "arguments": {"ticker": tick}}
        if any(w in low for w in ("insider", "form 4")) and "get_insider_activity" not in done:
            return {"tool": "get_insider_activity", "arguments": {"ticker": tick}}
        if "search_filings" not in done:
            return {"tool": "search_filings",
                    "arguments": {"query": question, "ticker": tick, "top_k": 4}}
        return {"answer": self._compose(messages, tick)}

    @staticmethod
    def _compose(messages: list[dict], ticker: str) -> str:
        lines: list[str] = []
        n = 0
        for m in messages:
            if m.get("role") != "tool" or m.get("is_error"):
                continue
            payload = m.get("parsed") or {}
            if m["tool"] == "get_financials":
                f = payload.get("financials", {})
                for key, label in (("revenues", "Revenue"), ("net_income", "Net income"),
                                   ("eps_diluted", "Diluted EPS")):
                    if key in f:
                        n += 1
                        v = f[key]
                        lines.append(f"{label} {v['value']:,} for period {v['period']} "
                                     f"({v['form']}), YoY growth {v['yoy_growth']}% [{n}]")
            elif m["tool"] == "search_filings":
                for r in (payload.get("results") or [])[:3]:
                    n += 1
                    lines.append(f"{r['ticker']} {r['filing_date']} Item {r['item']}: "
                                 f"{r['excerpt'][:260]} [{n}]")
            elif m["tool"] == "get_insider_activity":
                txs = payload.get("recent_transactions") or []
                n += 1
                lines.append(f"{ticker}: {len(txs)} recent Form 4 filings, most recent "
                             f"{txs[0]['date'] if txs else 'n/a'} [{n}]")
        return "\n\n".join(lines) if lines else "The tools returned no usable evidence."


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

class FilingAgent:
    def __init__(self, registry: ToolRegistry, llm=None, planner=None,
                 max_steps: int = 8, max_seconds: float = 90.0, max_tool_errors: int = 3,
                 allow: set[str] | None = None):
        self.registry = registry
        self.allow = allow
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.max_tool_errors = max_tool_errors
        if planner is not None:
            self.planner = planner
        elif llm is not None:
            self.planner = LLMPlanner(llm, registry)
        else:
            tickers = registry.get("available_tickers").fn()["tickers"]
            self.planner = HeuristicPlanner(registry, tickers)

    def run(self, question: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": question}]
        trace: list[TraceEvent] = []
        seen: set[tuple[str, str]] = set()
        tools_used: list[str] = []
        evidence: list[dict] = []
        errors = 0
        t0 = time.perf_counter()

        for step in range(1, self.max_steps + 1):
            # ---- guards checked BEFORE the model is called, every turn ----
            if time.perf_counter() - t0 > self.max_seconds:
                trace.append(TraceEvent(step, "guard", "wall-clock budget exhausted"))
                return AgentResult("Stopped: time budget exceeded.", step, trace,
                                   tools_used, "timeout", evidence)
            if errors >= self.max_tool_errors:
                trace.append(TraceEvent(step, "guard", f"tool error budget ({errors}) exhausted"))
                return AgentResult("Stopped: too many tool errors.", step, trace,
                                   tools_used, "tool_errors", evidence)

            tp = time.perf_counter()
            action = self.planner(messages)
            trace.append(TraceEvent(step, "plan", json.dumps(action)[:110],
                                    (time.perf_counter() - tp) * 1000))

            if not action:
                trace.append(TraceEvent(step, "guard", "planner returned nothing"))
                return AgentResult("Stopped: planner produced no action.", step, trace,
                                   tools_used, "no_action", evidence)

            if "answer" in action:
                answer = str(action["answer"])
                if is_degenerate(answer):
                    trace.append(TraceEvent(step, "guard", f"degenerate answer rejected: {answer[:30]!r}"))
                    messages.append({"role": "tool", "tool": "_format", "is_error": True,
                                     "content": ("Your reply was a placeholder or empty rather than a "
                                                 "real answer. Write the actual answer in full, "
                                                 "grounded in the tool results above and citing them "
                                                 "as [1], [2]. Do not reply with an ellipsis.")})
                    errors += 1
                    continue
                trace.append(TraceEvent(step, "final", answer[:90]))
                return AgentResult(answer, step, trace, tools_used, "completed", evidence)

            name = action.get("tool", "")
            args = action.get("arguments") or {}
            key = (name, json.dumps(args, sort_keys=True))

            # repeat detection: an agent re-calling the same thing is the most
            # common runaway-cost bug there is.
            if key in seen:
                trace.append(TraceEvent(step, "guard", f"repeat call blocked: {name}"))
                messages.append({"role": "tool", "tool": name, "is_error": True,
                                 "content": (f"You already called {name} with these exact arguments "
                                             f"and have the result. Do not repeat it — either use "
                                             f"what you have or answer the user.")})
                continue
            seen.add(key)

            if self.allow is not None and name not in self.allow:
                trace.append(TraceEvent(step, "guard", f"tool not permitted: {name}"))
                messages.append({"role": "tool", "tool": name, "is_error": True,
                                 "content": f"Tool '{name}' is not available to you. "
                                            f"Available: {sorted(self.allow)}"})
                errors += 1
                continue

            if err := self.registry.validate(name, args):
                why = "unknown tool" if self.registry.get(name) is None else "invalid args"
                trace.append(TraceEvent(step, "guard", f"{why}: {name} -> corrective message returned"))
                messages.append({"role": "tool", "tool": name, "is_error": True,
                                 "content": f"ERROR: {err}"})
                errors += 1
                continue

            tool = self.registry.get(name)
            ts = time.perf_counter()
            try:
                out = tool.fn(**args)
            except Exception as exc:                # a tool must never kill the loop
                trace.append(TraceEvent(step, "tool", f"{name} raised {type(exc).__name__}"))
                messages.append({"role": "tool", "tool": name, "is_error": True,
                                 "content": f"Tool '{name}' failed: {type(exc).__name__}: {exc}"})
                errors += 1
                continue
            ms = (time.perf_counter() - ts) * 1000

            tools_used.append(name)
            if isinstance(out, dict) and out.get("error"):
                errors += 1
            if name == "search_filings":
                evidence.extend(out.get("results") or [])

            trace.append(TraceEvent(step, "tool", f"{name}({args}) -> {str(out)[:70]}", ms))
            messages.append({"role": "tool", "tool": name, "is_error": False,
                             "parsed": out, "content": truncate(json.dumps(out, default=str))})

        trace.append(TraceEvent(self.max_steps, "guard", "max steps reached"))
        return AgentResult("Stopped: maximum reasoning steps reached without an answer.",
                           self.max_steps, trace, tools_used, "max_steps", evidence)


def truncate(text: str, max_chars: int = 6000) -> str:
    """Tool output re-enters the context on EVERY subsequent turn. One
    unbounded response blows the window and the budget. Always cap."""
    return text if len(text) <= max_chars else text[:max_chars] + f"\n...[truncated {len(text)-max_chars} chars]"


async def run_tools_parallel(registry: ToolRegistry, calls: list[dict]) -> list[dict]:
    """Independent tool calls in one turn should run concurrently. Sequential
    execution is the easiest latency win to leave on the table."""
    async def one(call: dict) -> dict:
        tool = registry.get(call["name"])
        if tool is None:
            return {"name": call["name"], "error": f"unknown tool {call['name']}"}
        try:
            out = await asyncio.wait_for(
                asyncio.to_thread(tool.fn, **call.get("arguments", {})), timeout=tool.timeout_s)
            return {"name": call["name"], "result": out}
        except asyncio.TimeoutError:
            return {"name": call["name"], "error": f"timed out after {tool.timeout_s}s"}
        except Exception as exc:
            return {"name": call["name"], "error": f"{type(exc).__name__}: {exc}"}
    return await asyncio.gather(*(one(c) for c in calls))


def check_agent_answer(result: AgentResult) -> dict:
    """Same deterministic checks as the RAG path, applied to the agent output."""
    ctx = " ".join(json.dumps(e, default=str) for e in result.evidence)
    return {
        "numeric": check_numeric_grounding(result.output, ctx) if ctx else
                   {"unsupported_numbers": [], "ok": True, "note": "no search evidence to check against"},
        "has_citations": bool(CITE.search(result.output)),
        "tools_used": result.tools_used,
        "stopped_because": result.stopped_because,
    }
