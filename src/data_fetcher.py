"""Data fetching layer — price history and fundamental data via yfinance."""

import yfinance as yf
import pandas as pd
from src.config import HISTORY_DAYS, CRORE


def fetch_bulk_price_data(symbols: list[str], period_days: int = HISTORY_DAYS) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for multiple NSE tickers in one batch call.

    Returns dict mapping symbol → DataFrame.
    """
    nse_tickers = [f"{s}.NS" for s in symbols]
    ticker_str = " ".join(nse_tickers)

    print(f"[*] Downloading price data for {len(symbols)} tickers...")
    raw = yf.download(ticker_str, period=f"{period_days}d", group_by="ticker",
                      threads=True, progress=True)

    result = {}
    for sym, nse_sym in zip(symbols, nse_tickers):
        try:
            if len(symbols) == 1:
                df = raw.copy()
            else:
                df = raw[nse_sym].copy() if nse_sym in raw.columns.get_level_values(0) else None
            if df is not None and not df.empty:
                df = df.dropna(subset=["Close"])
                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                result[sym] = df
        except Exception:
            continue

    print(f"[+] Got price data for {len(result)} / {len(symbols)} tickers")
    return result


def fetch_fundamentals(symbol: str) -> dict | None:
    """Fetch fundamental data for a single NSE ticker.

    Returns dict with fundamental metrics, or None if data unavailable.
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info

        if not info or info.get("regularMarketPrice") is None:
            return None

        # Market Cap
        market_cap = info.get("marketCap", 0)
        market_cap_cr = market_cap / CRORE if market_cap else 0

        # ROE
        roe = info.get("returnOnEquity")
        roe_pct = (roe * 100) if roe is not None else None

        # Debt to Equity
        de_ratio = info.get("debtToEquity")
        de_ratio_val = de_ratio / 100 if de_ratio is not None else None  # yfinance reports as percentage

        # Revenue & Profit Growth
        sales_growth = _calc_growth(ticker, "revenue")
        profit_growth = _calc_growth(ticker, "profit")

        # Operating Cash Flow
        ocf = _get_operating_cashflow(ticker)

        # Promoter Holding (best-effort from major_holders)
        promoter_pct = _get_promoter_holding(ticker, info)

        # Pledged shares — not reliably available from yfinance
        pledged_pct = None

        return {
            "market_cap_cr": round(market_cap_cr, 2),
            "roe_pct": round(roe_pct, 2) if roe_pct is not None else None,
            "debt_to_equity": round(de_ratio_val, 2) if de_ratio_val is not None else None,
            "sales_growth_pct": round(sales_growth, 2) if sales_growth is not None else None,
            "profit_growth_pct": round(profit_growth, 2) if profit_growth is not None else None,
            "operating_cashflow_cr": round(ocf / CRORE, 2) if ocf is not None else None,
            "promoter_holding_pct": round(promoter_pct, 2) if promoter_pct is not None else None,
            "pledged_pct": pledged_pct,
        }
    except Exception as e:
        return None


def _calc_growth(ticker: yf.Ticker, metric: str) -> float | None:
    """Calculate YoY growth for revenue or net income."""
    try:
        if metric == "revenue":
            financials = ticker.financials
            if financials is None or financials.empty:
                return None
            row = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else None
        else:
            financials = ticker.financials
            if financials is None or financials.empty:
                return None
            row = financials.loc["Net Income"] if "Net Income" in financials.index else None

        if row is None or len(row) < 2:
            return None

        latest = row.iloc[0]
        previous = row.iloc[1]
        if previous and previous != 0:
            return ((latest - previous) / abs(previous)) * 100
        return None
    except Exception:
        return None


def _get_operating_cashflow(ticker: yf.Ticker) -> float | None:
    """Get most recent operating cash flow."""
    try:
        cf = ticker.cashflow
        if cf is None or cf.empty:
            return None
        for label in ["Operating Cash Flow", "Total Cash From Operating Activities",
                       "Cash Flow From Operating Activities"]:
            if label in cf.index:
                val = cf.loc[label].iloc[0]
                return float(val) if pd.notna(val) else None
        return None
    except Exception:
        return None


def _get_promoter_holding(ticker: yf.Ticker, info: dict) -> float | None:
    """Best-effort attempt to get promoter/insider holding percentage."""
    try:
        # yfinance 'heldPercentInsiders' is the closest proxy
        insider = info.get("heldPercentInsiders")
        if insider is not None:
            return insider * 100
        return None
    except Exception:
        return None
