import os, time, json, math
import pandas as pd
import yfinance as yf
import requests
from constants import SYSTEM_PROMPT, USER_TEMPLATE, FEW_SHOT_USER, FEW_SHOT_ASSISTANT

try:
    from sec_filings import (
        fetch_sec_filings, 
        fetch_filing_text, 
        get_latest_earnings_transcript, 
        fetch_xbrl_financials,
        fetch_8k_events,
        fetch_insider_trading_data,
        fetch_earnings_transcripts
    )
    from rag_embeddings import create_filing_rag, EMBEDDINGS_AVAILABLE
    SEC_FILINGS_AVAILABLE = True
    ADVANCED_SEC_AVAILABLE = True
except ImportError:
    SEC_FILINGS_AVAILABLE = False
    ADVANCED_SEC_AVAILABLE = False
    print("⚠️  SEC filings module not available. Install: pip install sentence-transformers faiss-cpu")
    def fetch_8k_events(*args, **kwargs): return []
    def fetch_insider_trading_data(*args, **kwargs): return {}
    def fetch_earnings_transcripts(*args, **kwargs): return []

_filing_rag_cache = {}

def fetch_alpha_vantage_indicators(ticker: str) -> dict:
    """Fetch RSI, MACD, ATR from Alpha Vantage. Requires ALPHA_VANTAGE_API_KEY env var."""
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not av_key:
        return {}
    
    indicators = {}
    try:

        rsi_resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "RSI",
                "symbol": ticker.upper(),
                "interval": "daily",
                "apikey": av_key,
            },
            timeout=15,
        )
        if rsi_resp.ok:
            rsi_data = rsi_resp.json().get("Technical Analysis", {})
            if rsi_data:
                latest_date = sorted(rsi_data.keys())[-1]
                indicators["rsi"] = float(rsi_data[latest_date].get("RSI", 0))

        macd_resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "MACD",
                "symbol": ticker.upper(),
                "interval": "daily",
                "apikey": av_key,
            },
            timeout=15,
        )
        if macd_resp.ok:
            macd_data = macd_resp.json().get("Technical Analysis", {})
            if macd_data:
                latest_date = sorted(macd_data.keys())[-1]
                macd_val = macd_data[latest_date].get("MACD", 0)
                macd_signal = macd_data[latest_date].get("MACD_Signal", 0)
                macd_hist = macd_data[latest_date].get("MACD_Hist", 0)
                indicators["macd"] = {
                    "value": float(macd_val),
                    "signal": float(macd_signal),
                    "histogram": float(macd_hist),
                }

        stoch_resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "STOCH",
                "symbol": ticker.upper(),
                "interval": "daily",
                "apikey": av_key,
            },
            timeout=15,
        )
        if stoch_resp.ok:
            stoch_data = stoch_resp.json().get("Technical Analysis", {})
            if stoch_data:
                latest_date = sorted(stoch_data.keys())[-1]
                indicators["stochastic"] = {
                    "k": float(stoch_data[latest_date].get("SlowK", 0)),
                    "d": float(stoch_data[latest_date].get("SlowD", 0)),
                }
    except Exception as e:
        print(f"Warning: Alpha Vantage fetch failed: {e}")
    
    return indicators

def setup_filing_rag(ticker: str) -> bool:
    """Setup RAG system from SEC filings for a ticker"""
    global _filing_rag_cache
    
    if not SEC_FILINGS_AVAILABLE or not EMBEDDINGS_AVAILABLE:
        return False
    
    if ticker in _filing_rag_cache:
        return _filing_rag_cache[ticker] is not None
    
    try:
        print(f"📄 Fetching SEC filings for {ticker}...")
        filings_data = []
 
        latest = get_latest_earnings_transcript(ticker)
        if latest:
            filings_data.append(latest)
        
        if filings_data:
            print(f"✅ Creating RAG system from {len(filings_data)} filing(s)...")
            rag = create_filing_rag(ticker, filings_data)
            _filing_rag_cache[ticker] = rag
            return rag is not None
        else:
            _filing_rag_cache[ticker] = None
            print(f"⚠️  No SEC filings found for {ticker}")
            return False
    
    except Exception as e:
        print(f"⚠️  Error setting up RAG for {ticker}: {e}")
        _filing_rag_cache[ticker] = None
        return False

def get_filing_context(ticker: str, query: str) -> str:
    """Get RAG context from SEC filings if available"""
    global _filing_rag_cache
    
    if ticker not in _filing_rag_cache:
        setup_filing_rag(ticker)
    
    rag = _filing_rag_cache.get(ticker)
    if rag:
        try:
            return rag.generate_rag_context(query, top_k=2)
        except Exception as e:
            print(f"Error generating RAG context: {e}")
            return ""
    
    return ""

def fetch_market_snapshot(ticker: str, period="1y", interval="1d"):
    tk = yf.Ticker(ticker)
    hist = tk.history(period=period, interval=interval)

    # Compute SMA50 / SMA200
    hist["SMA50"]  = hist["Close"].rolling(window=50).mean()
    hist["SMA200"] = hist["Close"].rolling(window=200).mean()
 
    recent_hist = tk.history(period="5d")
    volume = int(recent_hist["Volume"].iloc[-1]) if len(recent_hist) > 0 else 0

    snapshot = {
        "ticker": ticker.upper(),
        "price": float(hist["Close"].iloc[-1]),
        "day_high": float(hist["High"].iloc[-1]) if len(hist) > 0 else float("nan"),
        "day_low": float(hist["Low"].iloc[-1]) if len(hist) > 0 else float("nan"),
        "volume": volume,
        "sma50": float(hist["SMA50"].iloc[-1]) if not math.isnan(hist["SMA50"].iloc[-1]) else None,
        "sma200": float(hist["SMA200"].iloc[-1]) if not math.isnan(hist["SMA200"].iloc[-1]) else None,
    }

    tech_indicators = fetch_alpha_vantage_indicators(ticker)
    if tech_indicators:
        snapshot["technical_indicators"] = tech_indicators
    
    return snapshot, hist

def fetch_news_headlines(ticker: str, limit=6):
    headlines = []
    try:
        tk = yf.Ticker(ticker)
        # Newer yfinance versions often provide a list of dicts with 'title' and 'link'
        for n in (tk.news or [])[:limit]:
            headlines.append({
                "title": n.get("title", "")[:160],
                "url": n.get("link") or n.get("providerPublishTime") or "",
            })
    except Exception:
        pass  # fallback below

    # Optional: Alpha Vantage Market News (requires ALPHA_VANTAGE_API_KEY)
    if len(headlines) == 0 and os.getenv("ALPHA_VANTAGE_API_KEY"):
        import requests
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker.upper(),
            "apikey": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "sort": "LATEST",
            "limit": limit
        }
        r = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        if r.ok:
            data = r.json().get("feed", [])[:limit]
            for item in data:
                headlines.append({
                    "title": item.get("title", "")[:160],
                    "url": item.get("url", "")
                })
    return headlines


def analyze_with_llm(ticker: str, snapshot: dict, headlines: list, model="openrouter/free", use_rag: bool = True, xbrl_financials: dict = None, advanced_sec_data: dict = None):

    if headlines:
        head_str = "\n".join([f'- "{h["title"]}" ({h.get("url","")})' for h in headlines])
    else:
        head_str = "- (no recent headlines found)"


    filing_context = ""
    if use_rag and SEC_FILINGS_AVAILABLE:
        analysis_query = f"Latest earnings, revenue growth, profitability, and strategic initiatives for {ticker}"
        filing_context = get_filing_context(ticker, analysis_query)
        if filing_context:
            filing_context = "\n\n[SEC FILING CONTEXT]\n" + filing_context

    xbrl_context = ""
    if xbrl_financials:
        xbrl_context = "\n\n[STRUCTURED FINANCIALS FROM SEC XBRL]\n"
        if "revenues" in xbrl_financials:
            rev = xbrl_financials["revenues"]
            xbrl_context += f"Revenues: ${rev['value']:,.0f} ({rev['period']})\n"
            if "yoy_growth" in rev:
                xbrl_context += f"  YoY Growth: {rev['yoy_growth']:+.2f}%\n"
        if "net_income" in xbrl_financials:
            ni = xbrl_financials["net_income"]
            xbrl_context += f"Net Income: ${ni['value']:,.0f} ({ni['period']})\n"
            if "yoy_growth" in ni:
                xbrl_context += f"  YoY Growth: {ni['yoy_growth']:+.2f}%\n"
        if "eps_diluted" in xbrl_financials:
            eps = xbrl_financials["eps_diluted"]
            xbrl_context += f"EPS (Diluted): ${eps['value']:.2f} ({eps['period']})\n"
            if "yoy_growth" in eps:
                xbrl_context += f"  YoY Growth: {eps['yoy_growth']:+.2f}%\n"

    advanced_context = ""
    if advanced_sec_data:

        if advanced_sec_data.get("8k_events"):
            advanced_context += "\n\n[RECENT 8-K CORPORATE EVENTS (Last 180 Days)]\n"
            for event in advanced_sec_data["8k_events"][:5]:
                advanced_context += f"- [{event['date']}] {event['event_type']} ({event['days_ago']}d ago)\n"

        if advanced_sec_data.get("insider_trading"):
            insider = advanced_sec_data["insider_trading"]
            if insider.get("recent_transactions"):
                advanced_context += f"\n[INSIDER TRADING (Form 4) ACTIVITY]\n"
                advanced_context += f"Recent filings: {insider['summary'].get('form_4_filings_count', 0)} in last 180 days\n"
                for filing in insider["recent_transactions"][:3]:
                    advanced_context += f"- [{filing['date']}] Filing ({filing['days_ago']}d ago)\n"
 
        if advanced_sec_data.get("earnings_filings"):
            advanced_context += f"\n[RECENT EARNINGS FILINGS]\n"
            for filing in advanced_sec_data["earnings_filings"][:2]:
                advanced_context += f"- {filing['filing_type']} ({filing['period']}, {filing['date']})\n"

    user_prompt = USER_TEMPLATE.format(
        ticker=ticker.upper(),
        snapshot_json=json.dumps(snapshot),
        headlines_bulleted=head_str
    )

    if filing_context:
        user_prompt += filing_context
    if xbrl_context:
        user_prompt += xbrl_context
    if advanced_context:
        user_prompt += advanced_context

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable")

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openrouter_api_key}",
            "HTTP-Referer": os.getenv("APP_URL", "http://localhost:3000"),
            "X-Title": os.getenv("APP_TITLE", "Stock Analysis CLI"),
        },
        json={
            "model": model,
            "temperature": 0.2,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": FEW_SHOT_USER},
                {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=90,
    )

    if not resp.ok:
        raise RuntimeError(f"OpenRouter request failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    output = data.get("choices", [{}])[0].get("message", {}).get("content")

    if not output or not isinstance(output, str):
        raise RuntimeError("Invalid OpenRouter response structure")

    cleaned = output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    return json.loads(cleaned)

def render_report(ticker: str, result: dict):
    print("\n=== AI Stock Analysis Report:", ticker.upper(), "===\n")
    print("Summary:")
    print(result.get("summary", "(no summary)"), "\n")

    print("Bullish Signals:")
    for s in result.get("bullish_signals", []):
        print("  •", s)
    print("\nBearish Signals:")
    for s in result.get("bearish_signals", []):
        print("  •", s)

    pt = result.get("price_target", {}) or {}
    base = pt.get("base")
    rng = pt.get("range")
    horizon = pt.get("time_horizon_days")
    print("\nPrice Target:")
    print(f"  Base: {base}  Range: {rng}  Horizon (days): {horizon}")

    conf = result.get("confidence")
    print(f"\nConfidence: {conf}")

    srcs = result.get("sources", [])
    if srcs:
        print("\nSources:")
        for u in srcs:
            print("  -", u)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI-Powered Stock Analysis")
    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL",
        type=str,
        help="Stock ticker symbol (e.g., AAPL). Defaults to AAPL if omitted.",
    )
    args = parser.parse_args()

    print(f"Fetching market data for {args.ticker}...")
    snapshot, hist = fetch_market_snapshot(args.ticker)

    print(f"Fetching news headlines for {args.ticker}...")
    headlines = fetch_news_headlines(args.ticker)

    xbrl_financials = None
    if SEC_FILINGS_AVAILABLE:
        print(f"Fetching XBRL financials for {args.ticker}...")
        xbrl_financials = fetch_xbrl_financials(args.ticker)
        if xbrl_financials:
            print("✅ Retrieved structured financials from SEC")

    advanced_sec_data = None
    if ADVANCED_SEC_AVAILABLE:
        print(f"Fetching advanced SEC data (8-K events, insider trading) for {args.ticker}...")
        try:
            events_8k = fetch_8k_events(args.ticker, days=180, limit=5)
            insider_data = fetch_insider_trading_data(args.ticker, limit=5, days=180)
            earnings_filings = fetch_earnings_transcripts(args.ticker, count=2)
            
            advanced_sec_data = {
                "8k_events": events_8k,
                "insider_trading": insider_data,
                "earnings_filings": earnings_filings,
            }
            
            if events_8k:
                print(f"✅ Found {len(events_8k)} recent 8-K events")
            if insider_data.get("recent_transactions"):
                print(f"✅ Found {len(insider_data['recent_transactions'])} recent Form 4 filings")
            if earnings_filings:
                print(f"✅ Found {len(earnings_filings)} recent earnings filings")
        except Exception as e:
            print(f"⚠️  Error fetching advanced SEC data: {e}")

    print("Analyzing with LLM...")
    analysis_result = analyze_with_llm(args.ticker, snapshot, headlines, xbrl_financials=xbrl_financials, advanced_sec_data=advanced_sec_data)

    render_report(args.ticker, analysis_result)