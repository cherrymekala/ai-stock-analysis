import discord
from discord.ext import commands, tasks
import os
import json
from dotenv import load_dotenv
from main import fetch_market_snapshot, fetch_news_headlines, analyze_with_llm, SEC_FILINGS_AVAILABLE, ADVANCED_SEC_AVAILABLE

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

load_dotenv()


DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
TICKERS = os.getenv("STOCK_TICKERS", "AAPL,MSFT,TSLA").split(",")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Discord bot logged in as {bot.user}")
    daily_stock_report.start()

@bot.command(name="analyze")
async def analyze_command(ctx, ticker: str = "AAPL"):
    """Command: !analyze AAPL - Run analysis for a stock"""
    async with ctx.typing():
        try:
            await ctx.send(f"🔍 Analyzing {ticker.upper()}...")
            
            snapshot, hist = fetch_market_snapshot(ticker)
            headlines = fetch_news_headlines(ticker)

            xbrl_financials = None
            if SEC_FILINGS_AVAILABLE and fetch_xbrl_financials:
                xbrl_financials = fetch_xbrl_financials(ticker)

            advanced_sec_data = None
            if ADVANCED_SEC_AVAILABLE and fetch_8k_events and fetch_insider_trading_data and fetch_earnings_transcripts:
                advanced_sec_data = {
                    "8k_events": fetch_8k_events(ticker, days=180, limit=5),
                    "insider_trading": fetch_insider_trading_data(ticker, limit=5, days=180),
                    "earnings_filings": fetch_earnings_transcripts(ticker, count=2),
                }
            
            result = analyze_with_llm(
                ticker,
                snapshot,
                headlines,
                xbrl_financials=xbrl_financials,
                advanced_sec_data=advanced_sec_data,
            )
            
            embed = build_discord_embed(ticker, result, snapshot)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

@tasks.loop(hours=24)
async def daily_stock_report():
    """Post daily stock analysis for configured tickers"""
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print(f"❌ Channel {CHANNEL_ID} not found")
        return
    
    await channel.send("📊 **Daily Stock Analysis Report** 📊")
    
    for ticker in TICKERS:
        try:
            print(f"Analyzing {ticker}...")
            snapshot, hist = fetch_market_snapshot(ticker.strip())
            headlines = fetch_news_headlines(ticker.strip())

            xbrl_financials = None
            if SEC_FILINGS_AVAILABLE and fetch_xbrl_financials:
                xbrl_financials = fetch_xbrl_financials(ticker.strip())

            advanced_sec_data = None
            if ADVANCED_SEC_AVAILABLE and fetch_8k_events and fetch_insider_trading_data and fetch_earnings_transcripts:
                advanced_sec_data = {
                    "8k_events": fetch_8k_events(ticker.strip(), days=180, limit=5),
                    "insider_trading": fetch_insider_trading_data(ticker.strip(), limit=5, days=180),
                    "earnings_filings": fetch_earnings_transcripts(ticker.strip(), count=2),
                }
            
            result = analyze_with_llm(
                ticker.strip(),
                snapshot,
                headlines,
                xbrl_financials=xbrl_financials,
                advanced_sec_data=advanced_sec_data,
            )
            
            embed = build_discord_embed(ticker.strip(), result, snapshot)
            await channel.send(embed=embed)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            await channel.send(f"⚠️ Error analyzing {ticker}: {str(e)}")

def build_discord_embed(ticker: str, result: dict, snapshot: dict) -> discord.Embed:
    """Build a Discord embed from analysis result"""
    confidence = result.get("confidence", 0)
    color = discord.Color.green() if confidence > 0.6 else discord.Color.orange() if confidence > 0.4 else discord.Color.red()
    
    embed = discord.Embed(
        title=f"📈 {ticker.upper()} Analysis",
        description=result.get("summary", "No summary")[:200],
        color=color
    )

    embed.add_field(
        name="💰 Price",
        value=f"${snapshot.get('price', 'N/A'):.2f}",
        inline=True
    )
    embed.add_field(
        name="📊 Confidence",
        value=f"{confidence:.0%}",
        inline=True
    )

    sma50 = snapshot.get("sma50")
    sma200 = snapshot.get("sma200")
    if sma50 and sma200:
        embed.add_field(
            name="📈 Moving Averages",
            value=f"SMA50: ${sma50:.2f}\nSMA200: ${sma200:.2f}",
            inline=True
        )

    if "technical_indicators" in snapshot:
        tech = snapshot["technical_indicators"]
        indicators_text = ""
        if "rsi" in tech:
            indicators_text += f"RSI: {tech['rsi']:.1f}\n"
        if "macd" in tech:
            indicators_text += f"MACD: {tech['macd']['value']:.4f}\n"
        if indicators_text:
            embed.add_field(name="🔧 Indicators", value=indicators_text.strip(), inline=True)

    bullish = result.get("bullish_signals", [])[:2]
    bearish = result.get("bearish_signals", [])[:2]
    
    if bullish:
        embed.add_field(
            name="🟢 Bullish",
            value="\n".join([f"✓ {s[:50]}" for s in bullish]),
            inline=False
        )
    
    if bearish:
        embed.add_field(
            name="🔴 Bearish",
            value="\n".join([f"⚠ {s[:50]}" for s in bearish]),
            inline=False
        )
 
    pt = result.get("price_target", {}) or {}
    if pt.get("base"):
        embed.add_field(
            name="🎯 Price Target",
            value=f"Base: ${pt.get('base', 0):.2f} (horizon: {pt.get('time_horizon_days', 0)}d)",
            inline=False
        )
    
    embed.set_footer(text="Stock Analysis")
    return embed

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ DISCORD_BOT_TOKEN not set in .env")
        exit(1)
    
    bot.run(DISCORD_TOKEN)
