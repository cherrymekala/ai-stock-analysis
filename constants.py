SYSTEM_PROMPT = """You are a cautious equity research assistant.
- Use only the supplied market data and headlines as evidence.
- Cite evidence clearly: compare each value separately or chain only when logically continuous.
  Examples: "Price above SMA200 (264.18 > 242.50)" or "Price between SMA200 and SMA50 (242.50 < 264.18 < 265.20)".
  Do NOT chain incomparable values like "264.18 < 265.20 < 242.50" which breaks mathematically.
- Output VALID JSON only that matches the provided schema.
- If data is insufficient, say so in 'summary' and set 'confidence' <= 0.4.
- You are NOT a financial advisor. This is for educational purposes only."""

USER_TEMPLATE = """Analyze the following context for ticker {ticker}:

[MARKET_SNAPSHOT]
{snapshot_json}

[HEADLINES] (most recent first)
{headlines_bulleted}

Return JSON with keys:
summary (str),
bullish_signals (list[str]),
bearish_signals (list[str]),
price_target (object: base [float], range [low, high], time_horizon_days [int]),
confidence (float 0-1),
sources (list[str] of URLs used, if any)

Constraints:
- JSON only, no markdown.
- Be concise but specific.
"""

# Optional few-shot exemplars to steer style
FEW_SHOT_USER = """Analyze:
[MARKET_SNAPSHOT]
{"ticker":"MSFT","price":410.2,"day_high":412.0,"day_low":405.1,"volume":21834567,"sma50":402.1,"sma200":370.4}
[HEADLINES]
- "Azure growth accelerates as AI demand surges" (https://example.com/msft1)
- "PC recovery lifts Windows OEM revenue" (https://example.com/msft2)
"""
FEW_SHOT_ASSISTANT = """{
  "summary": "Momentum remains constructive: price above both moving averages (410.2 > 402.1 > 370.4) with AI tailwinds from Azure demand.",
  "bullish_signals": [
    "Price above SMA50 and SMA200 (410.2 > 402.1 > 370.4)",
    "AI-related revenue drivers (Azure growth)"
  ],
  "bearish_signals": [
    "Short-term overextension risk near day high (412.0)"
  ],
  "price_target": {"base": 430.0, "range": [400.0, 450.0], "time_horizon_days": 90},
  "confidence": 0.71,
  "sources": ["https://example.com/msft1","https://example.com/msft2"]
}"""