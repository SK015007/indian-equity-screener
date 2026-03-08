"""Fetch Nifty 500 / Nifty 250 constituent lists from NSE India."""

import io
import requests
import pandas as pd


NSE_NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_NIFTY250_URL = "https://archives.nseindia.com/content/indices/ind_niftylargemidcap250list.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_nifty500_tickers() -> list[str]:
    """Return list of NSE ticker symbols for Nifty 500 constituents.

    Tries NSE archive URL first; if it fails, falls back to a cached CSV
    file at ./nifty500_cache.csv (if present).
    """
    try:
        session = requests.Session()
        # Hit NSE homepage first to get cookies
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=10)
        resp = session.get(NSE_NIFTY500_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["Symbol"].dropna().str.strip().tolist()
        # Cache for future fallback
        df.to_csv("nifty500_cache.csv", index=False)
        print(f"[+] Fetched {len(symbols)} Nifty 500 tickers from NSE")
        return symbols
    except Exception as e:
        print(f"[!] NSE fetch failed: {e}")
        return _load_cached("nifty500_cache.csv", "Nifty 500")


def fetch_nifty250_tickers() -> list[str]:
    """Return list of NSE ticker symbols for Nifty LargeMidcap 250 constituents."""
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=10)
        resp = session.get(NSE_NIFTY250_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["Symbol"].dropna().str.strip().tolist()
        df.to_csv("nifty250_cache.csv", index=False)
        print(f"[+] Fetched {len(symbols)} Nifty 250 tickers from NSE")
        return symbols
    except Exception as e:
        print(f"[!] NSE Nifty 250 fetch failed: {e}")
        return _load_cached("nifty250_cache.csv", "Nifty 250")


def _load_cached(filepath: str = "nifty500_cache.csv", label: str = "Nifty 500") -> list[str]:
    """Load tickers from local cache file."""
    try:
        df = pd.read_csv(filepath)
        symbols = df["Symbol"].dropna().str.strip().tolist()
        print(f"[+] Loaded {len(symbols)} tickers from local {label} cache")
        return symbols
    except FileNotFoundError:
        print(f"[!] No cached {label} list found.")
        return []


if __name__ == "__main__":
    tickers = fetch_nifty500_tickers()
    print(f"Total tickers: {len(tickers)}")
    if tickers:
        print("Sample:", tickers[:10])
