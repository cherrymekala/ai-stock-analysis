import requests
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta

HEADERS = {
    "User-Agent": "Cherry Mekala cherry@email.com",
}

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

def get_cik_from_ticker(ticker: str) -> Optional[str]:
    """
    Fetch CIK (Central Index Key) for a ticker from SEC company_tickers.json
    This is the official, reliable method recommended by SEC
    """
    try:
        r = requests.get(SEC_COMPANY_TICKERS_URL, headers=HEADERS, timeout=15)
        r.raise_for_status()
        
        data = r.json()

        for company in data.values():
            if company.get("ticker", "").lower() == ticker.lower():
                cik = str(company.get("cik_str", "")).zfill(10)
                return cik
        
        print(f"❌ Ticker {ticker} not found in SEC database")
        return None
    except Exception as e:
        print(f"Error fetching CIK for {ticker}: {e}")
        return None

def get_submissions(cik: str) -> Optional[Dict]:
    """Fetch submission history from SEC JSON API"""
    try:
        url = SEC_SUBMISSIONS_URL.format(cik=cik)
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching submissions for CIK {cik}: {e}")
        return None

def build_filing_url(cik: str, accession: str, primary_doc: str) -> str:
    """
    Build correct SEC filing URL from components
    Example: https://www.sec.gov/Archives/edgar/data/320193/0001564590-23-064045-index.htm
    """
    accession_clean = accession.replace("-", "")
    cik_no_leading_zeros = str(int(cik))
    
    return (
        f"{SEC_ARCHIVES_BASE}/"
        f"{cik_no_leading_zeros}/"
        f"{accession_clean}/"
        f"{primary_doc}"
    )

def fetch_sec_filings(ticker: str, filing_types: List[str] = ["10-K", "10-Q"], count: int = 5) -> List[Dict]:
    """
    Fetch recent SEC filings for a ticker using the submissions API
    
    Args:
        ticker: Stock ticker (e.g., "AAPL")
        filing_types: Types of filings to fetch (10-K, 10-Q, 8-K, etc.)
        count: Number of recent filings to retrieve
    
    Returns:
        List of filing dictionaries with metadata
    """
    cik = get_cik_from_ticker(ticker)
    if not cik:
        print(f"Could not find CIK for {ticker}")
        return []

    submissions = get_submissions(cik)
    if not submissions:
        return []
    
    filings = []
    recent = submissions.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])

    n = min(len(forms), len(accessions), len(dates), len(docs))
    for i in range(n):
        form = forms[i]

        if not any(ft in form for ft in filing_types):
            continue

        filing_dict = {
            "ticker": ticker.upper(),
            "cik": cik,
            "filing_type": form,
            "date": dates[i],
            "accession_number": accessions[i],
            "primary_document": docs[i],
            "document_type": docs[i] or form,
        }

        filing_dict["url"] = build_filing_url(
            cik,
            filing_dict["accession_number"],
            filing_dict["primary_document"]
        )

        filings.append(filing_dict)

        if len(filings) >= count:
            break
    
    return filings

def fetch_filing_text(filing_url: str) -> Optional[str]:
    """
    Fetch the full text of a SEC filing
    Returns the main document text (first 50KB to avoid huge files)
    """
    try:

        resp = requests.get(filing_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        text = resp.text

        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)

        text = re.sub(r'<[^>]+>', ' ', text)

        text = text.replace("&nbsp;", " ")
        text = text.replace("&quot;", '"')
        text = text.replace("&apos;", "'")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")

        text = ' '.join(text.split())

        return text[:50000] if text else None
    
    except Exception as e:
        print(f"Error fetching filing text: {e}")
        return None

def get_latest_earnings_transcript(ticker: str) -> Optional[Dict]:
    """Get the latest 10-Q or 10-K filing as a proxy for earnings info"""
    filings = fetch_sec_filings(ticker, filing_types=["10-Q", "10-K"], count=1)
    
    if not filings:
        return None
    
    latest = filings[0]
    url = latest.get("url")
    if not url:
        return None

    text = fetch_filing_text(url)
    
    if text:

        sections = extract_key_sections(text)
        
        return {
            "ticker": ticker.upper(),
            "filing_type": latest["filing_type"],
            "date": latest["date"],
            "text": text,  
            "sections": sections,  
            "url": url,
            "document_type": latest.get("document_type") or latest.get("primary_document") or latest.get("filing_type"),
        }
    
    return None

def extract_key_sections(filing_text: str) -> Dict[str, str]:
    """Extract key sections from 10-K/10-Q filings
    Returns longer sections (up to 10KB each) for better embedding quality
    """
    sections = {}

    match = re.search(
        r"(?:item\s+1[.\s]?business|business\s+overview)(.*?)(?:item\s+1a|risk\s+factors|end of item)",
        filing_text,
        re.IGNORECASE | re.DOTALL
    )
    if match:
        text = match.group(1).strip()
        if len(text) > 100:  
            sections["business"] = text[:10000]

    match = re.search(
        r"(?:item\s+1a[.\s]?risk\s+factors|risk\s+factors)(.*?)(?:unresolved|financial|item\s+2|item\s+1b)",
        filing_text,
        re.IGNORECASE | re.DOTALL
    )
    if match:
        text = match.group(1).strip()
        if len(text) > 100:
            sections["risks"] = text[:10000]

    match = re.search(
        r"(?:item\s+7[.\s]?md&a|management.{0,50}discussion|md&a)(.*?)(?:item\s+8|financial\s+statements|consolidated\s+statements)",
        filing_text,
        re.IGNORECASE | re.DOTALL
    )
    if match:
        text = match.group(1).strip()
        if len(text) > 100:
            sections["md_and_a"] = text[:15000]
    
    return sections

def fetch_xbrl_financials(ticker: str) -> Optional[Dict]:
    """
    Extract structured financial data from SEC XBRL (Company Facts API)
    Returns revenues, net income, EPS, and YoY growth metrics
    """
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return None
    
    try:
        url = SEC_COMPANY_FACTS_URL.format(cik=cik)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        
        data = resp.json()
        facts = data.get("facts", {})

        gaap = facts.get("us-gaap", {}) or facts.get("ifrs-full", {})
        
        financials = {}

        for revenue_key in ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]:
            if revenue_key in gaap:
                units = gaap[revenue_key].get("units", {})
                usd_data = units.get("USD", [])
                if usd_data:

                    annual = [x for x in usd_data if x.get("form") == "10-K"]
                    if annual:
                        sorted_data = sorted(annual, key=lambda x: x.get("end", ""), reverse=True)
                        current = sorted_data[0]
                        financials["revenues"] = {
                            "value": current.get("val"),
                            "period": current.get("end"),
                            "form": current.get("form"),
                        }

                        if len(sorted_data) > 1:
                            prev = sorted_data[1]
                            yoy = ((current.get("val", 0) - prev.get("val", 0)) / prev.get("val", 1)) * 100
                            financials["revenues"]["yoy_growth"] = round(yoy, 2)
                        break

        if "NetIncomeLoss" in gaap:
            units = gaap["NetIncomeLoss"].get("units", {})
            usd_data = units.get("USD", [])
            if usd_data:
                annual = [x for x in usd_data if x.get("form") == "10-K"]
                if annual:
                    sorted_data = sorted(annual, key=lambda x: x.get("end", ""), reverse=True)
                    current = sorted_data[0]
                    financials["net_income"] = {
                        "value": current.get("val"),
                        "period": current.get("end"),
                        "form": current.get("form"),
                    }
                    if len(sorted_data) > 1:
                        prev = sorted_data[1]
                        yoy = ((current.get("val", 0) - prev.get("val", 0)) / prev.get("val", 1)) * 100
                        financials["net_income"]["yoy_growth"] = round(yoy, 2)

        if "EarningsPerShareDiluted" in gaap:
            units = gaap["EarningsPerShareDiluted"].get("units", {})
            usd_per_share = units.get("USD/shares", [])
            if usd_per_share:
                annual = [x for x in usd_per_share if x.get("form") == "10-K"]
                if annual:
                    sorted_data = sorted(annual, key=lambda x: x.get("end", ""), reverse=True)
                    current = sorted_data[0]
                    financials["eps_diluted"] = {
                        "value": current.get("val"),
                        "period": current.get("end"),
                        "form": current.get("form"),
                    }
                    if len(sorted_data) > 1:
                        prev = sorted_data[1]
                        yoy = ((current.get("val", 0) - prev.get("val", 0)) / prev.get("val", 1)) * 100
                        financials["eps_diluted"]["yoy_growth"] = round(yoy, 2)
        
        return financials if financials else None
    
    except Exception as e:
        print(f"Error fetching XBRL financials for {ticker}: {e}")
        return None

def fetch_8k_events(ticker: str, days: int = 90, limit: int = 10) -> List[Dict]:
    """
    Fetch recent 8-K (Form 8-K) filings - real-time corporate events
    8-Ks must be filed within 4 business days of triggering event
    
    Common Item codes:
    - 2.01: Asset Acquisition/Disposal
    - 2.02: Results & Financials
    - 5.02: Executive/Director Changes
    - 7.01: Major Announcement (Reg FD)
    - 8.01: Other Significant Events
    """
    
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return []
    
    submissions = get_submissions(cik)
    if not submissions:
        return []
    
    events = []
    filings_list = submissions.get("filings", {}).get("recent", {})
    
    forms = filings_list.get("form", [])
    accessions = filings_list.get("accessionNumber", [])
    dates = filings_list.get("filingDate", [])
    docs = filings_list.get("primaryDocument", [])
    items = filings_list.get("items", [])
    
    if days > 0:
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    else:
        cutoff_str = "1900-01-01"
    
    n = min(len(forms), len(accessions), len(dates), len(docs))
    
    for i in range(n):
        form = forms[i]
        filing_date = dates[i]
        
        if form != "8-K" or filing_date < cutoff_str:
            continue
        
        item_code = items[i] if i < len(items) else ""
        
        event_type_map = {
            "1.01": "Bankruptcy/Receivership",
            "2.01": "Asset Acquisition/Disposal",
            "2.02": "Results & Financials",
            "2.03": "Material Impairments",
            "3.01": "Material Agreements",
            "4.01": "Accountant Changes",
            "5.01": "Change in Control",
            "5.02": "Executive/Director Changes",
            "5.07": "Bankruptcy/Receivership",
            "7.01": "Major Announcement (Reg FD)",
            "8.01": "Other Significant Events",
        }
        
        events.append({
            "ticker": ticker.upper(),
            "filing_type": form,
            "date": filing_date,
            "item_code": item_code,
            "accession_number": accessions[i],
            "primary_document": docs[i],
            "url": build_filing_url(cik, accessions[i], docs[i]),
            "days_ago": (datetime.now() - datetime.strptime(filing_date, "%Y-%m-%d")).days,
            "event_type": event_type_map.get(item_code, "Corporate Event"),
        })
        
        if len(events) >= limit:
            break
    
    return events

def fetch_earnings_transcripts(ticker: str, count: int = 2) -> List[Dict]:
    """
    Extract earnings-related filings from quarterly (10-Q) and annual (10-K) filings
    Returns metadata for recent earnings filings
    """
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return []
    
    submissions = get_submissions(cik)
    if not submissions:
        return []
    
    earnings_data = []
    filings_list = submissions.get("filings", {}).get("recent", {})
    
    forms = filings_list.get("form", [])
    accessions = filings_list.get("accessionNumber", [])
    dates = filings_list.get("filingDate", [])
    docs = filings_list.get("primaryDocument", [])
    
    n = min(len(forms), len(accessions), len(dates), len(docs))
    
    for i in range(n):
        form = forms[i]
        
        if form not in ["10-Q", "10-K"]:
            continue
        
        earnings_data.append({
            "ticker": ticker.upper(),
            "filing_type": form,
            "date": dates[i],
            "period": "quarterly" if form == "10-Q" else "annual",
            "accession_number": accessions[i],
            "primary_document": docs[i],
            "url": build_filing_url(cik, accessions[i], docs[i]),
            "description": f"{form} filing - contains MD&A with earnings discussion"
        })
        
        if len(earnings_data) >= count:
            break
    
    return earnings_data

def fetch_insider_trading_data(ticker: str, limit: int = 10, days: int = 90) -> Dict:
    """
    Fetch Form 4 (insider trading) data
    Shows insider buying/selling activity
    
    Returns dict with recent_transactions list and summary
    """
    
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return {}
    
    submissions = get_submissions(cik)
    if not submissions:
        return {}
    
    form4_filings = []
    filings_list = submissions.get("filings", {}).get("recent", {})
    
    forms = filings_list.get("form", [])
    accessions = filings_list.get("accessionNumber", [])
    dates = filings_list.get("filingDate", [])
    docs = filings_list.get("primaryDocument", [])
    
    if days > 0:
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    else:
        cutoff_str = "1900-01-01"
    
    n = min(len(forms), len(accessions), len(dates), len(docs))
    
    for i in range(n):
        form = forms[i]
        filing_date = dates[i]
        
        if form != "4" or filing_date < cutoff_str:
            if form == "4":
                break 
            continue
        
        form4_filings.append({
            "ticker": ticker.upper(),
            "filing_type": "Form 4",
            "date": filing_date,
            "days_ago": (datetime.now() - datetime.strptime(filing_date, "%Y-%m-%d")).days,
            "accession_number": accessions[i],
            "primary_document": docs[i],
            "url": build_filing_url(cik, accessions[i], docs[i]),
        })
        
        if len(form4_filings) >= limit:
            break
    
    return {
        "recent_transactions": form4_filings,
        "summary": {
            "form_4_filings_count": len(form4_filings),
            "note": "Full insider transaction analysis requires parsing Form 4 document content"
        }
    }
