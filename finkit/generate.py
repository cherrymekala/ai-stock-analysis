"""Grounded generation: citations required, abstention allowed, numbers checked.

The LLM is optional. Retrieval quality is measured without one, and the
deterministic output checks below need no model either -- which is the point.
Most of what makes a filings answer trustworthy is checkable with code, and
code is cheaper, faster and more reproducible than a judge.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

SYSTEM_PROMPT = """\
You are an equity research assistant answering from SEC filings. Answer ONLY
from the numbered sources below.

Rules:
- Every factual sentence must end with a citation like [1] or [2][3].
- Quote figures, percentages and dates EXACTLY as written in the source. Never
  round, convert, restate in different units, or compute a new number.
- Attribute every figure to the company and fiscal period it came from. Do not
  mix companies or periods in a single claim.
- If the sources do not contain the answer, reply with exactly:
  "I could not find this in the filings provided."
  Do not use outside knowledge and do not guess.
- You are not a financial adviser. Do not give investment recommendations.
"""

USER_TEMPLATE = """SOURCES:
{context}

QUESTION: {question}
"""

REFUSAL = "I could not find this in the filings provided."

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    model: str = "none"

    def to_dict(self) -> dict:
        return {"answer": self.text, "sources": self.sources,
                "checks": self.checks, "model": self.model}


# --------------------------------------------------------------------------
# deterministic checks — run on every answer, in production too
# --------------------------------------------------------------------------

def _norm_num(s: str) -> str:
    return s.replace(",", "").rstrip("%").rstrip(".")


def check_numeric_grounding(answer: str, context: str) -> dict:
    """Every figure in the answer must appear in the retrieved context.

    Cheap, deterministic, and it catches the failure that matters most here:
    a plausible invented revenue number. Citation markers are stripped first --
    "[1]" is a citation, not a numeric claim.
    """
    stripped = CITE.sub(" ", answer)
    ans = {_norm_num(m) for m in NUM.findall(stripped)}
    ctx = {_norm_num(m) for m in NUM.findall(context)}
    unsupported = sorted(ans - ctx)
    return {"unsupported_numbers": unsupported, "ok": not unsupported}


def check_citations(answer: str, n_sources: int) -> dict:
    cited = {int(c) for c in CITE.findall(answer)}
    invalid = sorted(c for c in cited if c < 1 or c > n_sources)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    with_cite = sum(1 for s in sentences if CITE.search(s))
    return {
        "cited": sorted(cited),
        "invalid_citations": invalid,
        "sentence_citation_rate": round(with_cite / len(sentences), 2) if sentences else 0.0,
        "ok": not invalid and (with_cite > 0 or answer.strip() == REFUSAL),
    }


def check_ticker_consistency(sources: list[dict], expected: str | None) -> dict:
    """Did every cited source come from the company being asked about?

    A filings assistant that answers a question about NVIDIA using Microsoft's
    MD&A produces a fluent, correctly-cited, completely wrong answer. This is
    the domain's version of a tenant leak and it is worth checking explicitly.
    """
    tickers = sorted({s["ticker"] for s in sources})
    return {
        "tickers_in_context": tickers,
        "mixed": len(tickers) > 1,
        "ok": (expected is None) or (tickers == [expected]),
    }


def is_refusal(answer: str) -> bool:
    markers = ["could not find", "not in the filings", "cannot answer",
               "no information", "unable to determine", "don't have"]
    a = answer.lower()
    return any(m in a for m in markers)


# --------------------------------------------------------------------------
# generation backends
# --------------------------------------------------------------------------

class ExtractiveGenerator:
    """No-LLM fallback: the top passages verbatim with their citations.

    An evidence list rather than an answer, but it makes the pipeline runnable
    and testable with no key, and it cannot hallucinate -- a defensible
    degraded mode for a system whose whole job is grounding.
    """
    model = "extractive"

    def generate(self, question: str, context: str, sources: list[dict],
                 passages: list[str] | None = None) -> str:
        if not sources:
            return REFUSAL
        out = []
        for i, s in enumerate(sources[:3]):
            body = passages[i] if passages and i < len(passages) else ""
            out.append(f"{' '.join(body.split())[:300]} [{s['n']}]")
        return "\n\n".join(out)


class OpenRouterGenerator:  # pragma: no cover - needs a key
    """The project already used OpenRouter; keep that path.

    The default is a free-tier model, which is rate-limited and occasionally
    withdrawn. That is fine for a demo and wrong for production: pick the model
    on a task-specific eval, not on availability.
    """
    def __init__(self, model: str = "nvidia/nemotron-3.5-lightning:free"):
        from openai import OpenAI
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ["OPENROUTER_API_KEY"])
        self.model = model

    def generate(self, question: str, context: str, sources: list[dict],
                 passages: list[str] | None = None) -> str:
        if not sources:
            return REFUSAL
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,                 # 0 for eval reproducibility
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": USER_TEMPLATE.format(
                          context=context, question=question)}],
        )
        return (resp.choices[0].message.content or "").strip()

    def generate_raw(self, system: str, messages: list[dict]) -> str:
        """Free-form turn used by the agent planner. Tool-call requests come
        back as JSON in the message body because the free models here do not
        expose a native tool-calling API; prefer that API where it exists."""
        convo = [{"role": "system", "content": system}]
        for m in messages:
            if m.get("role") == "tool":
                convo.append({"role": "user",
                              "content": f"TOOL RESULT ({m['tool']}):\n{m['content'][:4000]}"})
            else:
                convo.append({"role": "user", "content": m["content"]})
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=900, messages=convo)
        return (resp.choices[0].message.content or "").strip()


def get_generator(prefer: str = "auto"):
    if prefer == "none":
        return ExtractiveGenerator()
    if prefer in ("auto", "openrouter") and os.getenv("OPENROUTER_API_KEY"):
        try:
            return OpenRouterGenerator()
        except Exception:
            pass
    return ExtractiveGenerator()


def answer_question(rag, question: str, generator=None, ticker: str | None = None) -> Answer:
    generator = generator or get_generator()
    r = rag.query(question, ticker=ticker)
    model = getattr(generator, "model", "none")
    # Resolve the expected company ALWAYS, not just when filtering is enabled.
    # Gating the check on the config would mean a configuration that does no
    # scoping can never fail the scoping check — the measurement would be
    # switched off by the very setting it is supposed to evaluate.
    expected = ticker or rag.resolve_ticker(question)

    if r["abstain"]:
        return Answer(REFUSAL, [], {
            "abstained": True,
            "top_score": round(r["hits"][0].score, 4) if r["hits"] else None,
            "numeric": {"unsupported_numbers": [], "ok": True},
            "citations": {"cited": [], "invalid_citations": [],
                          "sentence_citation_rate": 0.0, "ok": True},
            "ticker": {"tickers_in_context": [], "mixed": False, "ok": True},
            "refusal": True,
            "latency_ms": r["latency_ms"],
        }, model)

    text = generator.generate(question, r["context"], r["sources"],
                              passages=[h.chunk.text for h in r["hits"]])
    return Answer(
        text=text,
        sources=r["sources"],
        checks={
            "abstained": False,
            "top_score": round(r["hits"][0].score, 4) if r["hits"] else None,
            "numeric": check_numeric_grounding(text, r["context"]),
            "citations": check_citations(text, len(r["sources"])),
            "ticker": check_ticker_consistency(r["sources"], expected),
            "refusal": is_refusal(text),
            "latency_ms": r["latency_ms"],
        },
        model=model,
    )
