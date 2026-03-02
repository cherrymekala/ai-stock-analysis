import streamlit as st
import json
import os
from main import fetch_market_snapshot, fetch_news_headlines, analyze_with_llm, SEC_FILINGS_AVAILABLE, ADVANCED_SEC_AVAILABLE
import pandas as pd

try:
    from sec_filings import (
        fetch_xbrl_financials,
        fetch_8k_events,
        fetch_insider_trading_data,
        fetch_earnings_transcripts
    )
except ImportError:
    fetch_xbrl_financials = None
    fetch_8k_events = None
    fetch_insider_trading_data = None
    fetch_earnings_transcripts = None

st.set_page_config(page_title="📈 AI Stock Analyzer", layout="wide")

st.title("📈 AI-Powered Stock Analysis Dashboard")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, TSLA)")
    
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("⚠️ Missing OPENROUTER_API_KEY env var")
    
    analyze_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

if analyze_button or "analysis_result" in st.session_state:
    if analyze_button:
        with st.spinner("Fetching market data..."):
            snapshot, hist = fetch_market_snapshot(ticker)
            headlines = fetch_news_headlines(ticker)

        xbrl_financials = None
        if SEC_FILINGS_AVAILABLE and fetch_xbrl_financials:
            with st.spinner("Fetching SEC financials..."):
                xbrl_financials = fetch_xbrl_financials(ticker)

        advanced_sec_data = None
        if ADVANCED_SEC_AVAILABLE and fetch_8k_events and fetch_insider_trading_data and fetch_earnings_transcripts:
            with st.spinner("Fetching advanced SEC data..."):
                advanced_sec_data = {
                    "8k_events": fetch_8k_events(ticker, days=180, limit=5),
                    "insider_trading": fetch_insider_trading_data(ticker, limit=5, days=180),
                    "earnings_filings": fetch_earnings_transcripts(ticker, count=2),
                }
        
        with st.spinner("Running LLM analysis..."):
            try:
                analysis_result = analyze_with_llm(
                    ticker,
                    snapshot,
                    headlines,
                    xbrl_financials=xbrl_financials,
                    advanced_sec_data=advanced_sec_data,
                )
                st.session_state.analysis_result = analysis_result
                st.session_state.snapshot = snapshot
                st.session_state.headlines = headlines
                st.session_state.advanced_sec_data = advanced_sec_data
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.stop()
    
    result = st.session_state.get("analysis_result", {})
    snapshot = st.session_state.get("snapshot", {})
    headlines = st.session_state.get("headlines", [])
    advanced_sec_data = st.session_state.get("advanced_sec_data", {})

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${snapshot.get('price', 'N/A'):.2f}")
    with col2:
        day_high = snapshot.get('day_high', float('nan'))
        day_high_str = f"${day_high:.2f}" if not (isinstance(day_high, float) and day_high != day_high) else "N/A"
        st.metric("Day High", day_high_str)
    with col3:
        day_low = snapshot.get('day_low', float('nan'))
        day_low_str = f"${day_low:.2f}" if not (isinstance(day_low, float) and day_low != day_low) else "N/A"
        st.metric("Day Low", day_low_str)
    with col4:
        st.metric("Volume", f"{snapshot.get('volume', 0):,.0f}")

    st.subheader("📋 AI Summary")
    st.info(result.get("summary", "(No summary available)"))

    col1, col2 = st.columns(2)
    
    with col1:
        confidence = result.get("confidence", 0)
        st.metric("Confidence Score", f"{confidence:.0%}")
        st.progress(confidence, text=f"{confidence:.1%}")
    
    with col2:
        st.subheader("📊 Moving Averages")
        ma_df = pd.DataFrame({
            "Indicator": ["SMA 50", "SMA 200", "Current Price"],
            "Value": [
                snapshot.get("sma50", 0),
                snapshot.get("sma200", 0),
                snapshot.get("price", 0),
            ]
        })
        st.dataframe(ma_df, use_container_width=True)

    if "technical_indicators" in snapshot:
        st.subheader("📈 Technical Indicators")
        tech = snapshot["technical_indicators"]
        
        col1, col2, col3 = st.columns(3)
        if "rsi" in tech:
            with col1:
                rsi = tech["rsi"]
                color = "🟢" if 30 < rsi < 70 else "🔴"
                st.metric(f"{color} RSI", f"{rsi:.1f}")
        
        if "macd" in tech:
            with col2:
                macd = tech["macd"]["value"]
                signal = tech["macd"]["signal"]
                hist = tech["macd"]["histogram"]
                st.metric("MACD", f"{macd:.4f}", delta=f"Histogram: {hist:.4f}")
        
        if "stochastic" in tech:
            with col3:
                stoch_k = tech["stochastic"]["k"]
                st.metric("Stochastic %K", f"{stoch_k:.1f}")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Bullish Signals")
        bullish = result.get("bullish_signals", [])
        if bullish:
            for signal in bullish:
                st.success(f"✓ {signal}")
        else:
            st.info("No bullish signals detected")
    
    with col2:
        st.subheader("🔴 Bearish Signals")
        bearish = result.get("bearish_signals", [])
        if bearish:
            for signal in bearish:
                st.warning(f"⚠ {signal}")
        else:
            st.info("No bearish signals detected")

    st.subheader("🎯 Price Target")
    pt = result.get("price_target", {}) or {}
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Base Target", f"${pt.get('base', 0):.2f}")
    with col2:
        low = pt.get("range", [None])[0] if pt.get("range") else None
        st.metric("Range Low", f"${low:.2f}" if low else "N/A")
    with col3:
        high = pt.get("range", [None, None])[1] if pt.get("range") and len(pt.get("range", [])) > 1 else None
        st.metric("Range High", f"${high:.2f}" if high else "N/A")
    with col4:
        horizon = pt.get("time_horizon_days", 0)
        st.metric("Horizon", f"{horizon} days")

    # Filter out headlines without valid titles
    valid_headlines = [h for h in headlines if h.get("title", "").strip()]
    
    if valid_headlines:
        st.subheader("📰 Recent News")
        for h in valid_headlines[:5]:
            with st.expander(h.get("title", "Untitled")[:100]):
                url = h.get('url', '').strip()
                if url and url != '#':
                    st.write(f"🔗 [Read more]({url})")
                else:
                    st.write("📎 No link available")
    else:
        st.subheader("📰 Recent News")
        st.info("No recent headlines available")

    # Advanced SEC Data
    if advanced_sec_data:
        st.subheader("🏛️ SEC Event Intelligence")

        events_8k = advanced_sec_data.get("8k_events", [])
        insider_data = advanced_sec_data.get("insider_trading", {})
        earnings_filings = advanced_sec_data.get("earnings_filings", [])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recent 8-K Events", len(events_8k))
        with col2:
            st.metric("Form 4 Filings", len(insider_data.get("recent_transactions", [])))
        with col3:
            st.metric("Earnings Filings", len(earnings_filings))

        if events_8k:
            st.markdown("**Recent 8-K Events**")
            for event in events_8k[:5]:
                st.write(f"• {event.get('date')} — {event.get('event_type', 'Corporate Event')} ({event.get('days_ago', 'N/A')}d ago)")

        insider_filings = insider_data.get("recent_transactions", []) if insider_data else []
        if insider_filings:
            st.markdown("**Recent Form 4 (Insider Trading) Filings**")
            for filing in insider_filings[:5]:
                st.write(f"• {filing.get('date')} — Form 4 filing ({filing.get('days_ago', 'N/A')}d ago)")

        if earnings_filings:
            st.markdown("**Recent Earnings-Related Filings**")
            for filing in earnings_filings[:3]:
                st.write(f"• {filing.get('date')} — {filing.get('filing_type')} ({filing.get('period', 'N/A')})")

    st.subheader("📚 Sources")
    sources = result.get("sources", [])
    if sources:
        for src in sources:
            st.write(f"• [{src}]({src})")
    else:
        st.info("No sources cited")

    st.divider()
    st.subheader("📥 Export")
    col1, col2 = st.columns(2)
    
    with col1:
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="📄 Download Analysis (JSON)",
            data=json_str,
            file_name=f"{ticker.upper()}_analysis.json",
            mime="application/json",
        )
    
    with col2:
        csv_str = pd.DataFrame([result]).to_csv(index=False)
        st.download_button(
            label="📊 Download Analysis (CSV)",
            data=csv_str,
            file_name=f"{ticker.upper()}_analysis.csv",
            mime="text/csv",
        )

else:
    st.info("👈 Enter a ticker symbol and click 'Analyze Stock' to get started!")
