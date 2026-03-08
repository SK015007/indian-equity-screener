"""
Indian Equity Swing Trading Screener
=====================================
Screens Nifty 500 stocks for swing trading opportunities using
technical (200 EMA crossover + ADX) and fundamental filters.

Usage:
    python screener.py
"""

import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

from src.config import (
    MIN_MARKET_CAP_CR, MIN_SALES_GROWTH_PCT, MIN_PROFIT_GROWTH_PCT,
    MIN_ROE_PCT, MAX_DEBT_TO_EQUITY, MIN_PROMOTER_HOLDING_PCT,
    MAX_PLEDGED_PCT, MIN_AVG_TRADED_VALUE_CR, CRORE, MAX_WORKERS,
    EMA_LONG, NIFTY_REGIME_CHECK,
)
from src.stock_universe import fetch_nifty500_tickers
from src.data_fetcher import fetch_bulk_price_data, fetch_fundamentals
from src.technicals import screen_technical, calc_ema


def _check_nifty_regime() -> tuple[bool, dict]:
    """Check if Nifty 50 is in a bullish regime (above 200 DMA).

    Returns (is_bullish, info_dict).
    """
    try:
        nifty = yf.download("^NSEI", period="365d", progress=False)
        if nifty.empty:
            print("  [!] Could not fetch Nifty 50 data, skipping regime check")
            return True, {}

        close = nifty["Close"]
        # Flatten MultiIndex if present
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        ema200 = calc_ema(close, EMA_LONG)
        current = float(close.iloc[-1])
        current_ema = float(ema200.iloc[-1])
        pct_above = round(((current - current_ema) / current_ema) * 100, 2)
        is_bullish = current > current_ema

        info = {
            "nifty_price": round(current, 2),
            "nifty_ema200": round(current_ema, 2),
            "nifty_pct_above_200": pct_above,
            "regime": "BULLISH" if is_bullish else "BEARISH",
        }
        return is_bullish, info
    except Exception as e:
        print(f"  [!] Nifty regime check failed: {e}")
        return True, {}


def run_screener():
    start_time = time.time()
    print("=" * 70)
    print("  INDIAN EQUITY SWING TRADING SCREENER (v2)")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Step 0: Market regime check
    if NIFTY_REGIME_CHECK:
        print("\n[Step 0] Checking Nifty 50 market regime...")
        is_bullish, regime_info = _check_nifty_regime()
        if regime_info:
            print(f"  Nifty 50: {regime_info['nifty_price']} | "
                  f"200 EMA: {regime_info['nifty_ema200']} | "
                  f"{regime_info['nifty_pct_above_200']}% above | "
                  f"Regime: {regime_info['regime']}")
        if not is_bullish:
            print("\n  [!] WARNING: Nifty 50 is BELOW its 200 EMA.")
            print("  Market regime is BEARISH - swing long trades carry higher risk.")
            print("  Proceeding with screening but exercise extra caution.\n")

    # Step 1: Get stock universe
    symbols = fetch_nifty500_tickers()
    if not symbols:
        print("[!] No tickers to screen. Exiting.")
        sys.exit(1)

    total = len(symbols)
    print(f"\n[Step 1/4] Stock universe: {total} tickers")

    # Step 2: Fetch price data in bulk
    print(f"\n[Step 2/4] Fetching price/volume data...")
    price_data = fetch_bulk_price_data(symbols)
    print(f"  Price data available for {len(price_data)} stocks")

    # Step 3: Apply technical filters
    print(f"\n[Step 3/4] Applying technical filters (EMA crossover + ADX + RSI)...")
    tech_passed = {}
    for sym, df in price_data.items():
        result = screen_technical(df)
        if result is not None:
            avg_tv_cr = result["avg_traded_value"] / CRORE
            if avg_tv_cr >= MIN_AVG_TRADED_VALUE_CR:
                result["avg_traded_value_cr"] = round(avg_tv_cr, 2)
                tech_passed[sym] = result

    print(f"  Technical filter: {len(tech_passed)} / {len(price_data)} stocks passed")
    if not tech_passed:
        print("\n[!] No stocks passed technical screening.")
        _print_elapsed(start_time)
        return

    # Step 4: Apply fundamental filters (only on technical survivors)
    print(f"\n[Step 4/4] Fetching fundamentals for {len(tech_passed)} stocks...")
    final_results = []
    near_miss = []  # Stocks that passed technical but failed fundamental

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_fundamentals, sym): sym for sym in tech_passed}
        done_count = 0
        for future in as_completed(futures):
            sym = futures[future]
            done_count += 1
            print(f"  [{done_count}/{len(tech_passed)}] {sym}", end="")

            fund = future.result()
            if fund is None:
                print(" - fundamental data unavailable, skipped")
                miss_row = {"symbol": sym, "fail_reason": "No fundamental data"}
                miss_row.update(tech_passed[sym])
                near_miss.append(miss_row)
                continue

            passed, reason = _check_fundamentals(fund)
            if passed:
                print(" -> PASSED")
                row = {"symbol": sym}
                row.update(tech_passed[sym])
                row.update(fund)
                final_results.append(row)
            else:
                print(f" - failed: {reason}")
                miss_row = {"symbol": sym, "fail_reason": reason}
                miss_row.update(tech_passed[sym])
                miss_row.update(fund)
                near_miss.append(miss_row)

    # Output results
    _print_results(final_results, near_miss, total, len(price_data),
                   len(tech_passed), start_time)


def _check_fundamentals(fund: dict) -> tuple[bool, str]:
    """Check all fundamental conditions. Returns (passed, failure_reason)."""

    mcap = fund.get("market_cap_cr", 0)
    if mcap < MIN_MARKET_CAP_CR:
        return False, f"Market cap Rs.{mcap} Cr < Rs.{MIN_MARKET_CAP_CR} Cr"

    roe = fund.get("roe_pct")
    if roe is not None and roe < MIN_ROE_PCT:
        return False, f"ROE {roe}% < {MIN_ROE_PCT}%"

    de = fund.get("debt_to_equity")
    if de is not None and de > MAX_DEBT_TO_EQUITY:
        return False, f"D/E {de} > {MAX_DEBT_TO_EQUITY}"

    sg = fund.get("sales_growth_pct")
    if sg is not None and sg < MIN_SALES_GROWTH_PCT:
        return False, f"Sales growth {sg}% < {MIN_SALES_GROWTH_PCT}%"

    pg = fund.get("profit_growth_pct")
    if pg is not None and pg < MIN_PROFIT_GROWTH_PCT:
        return False, f"Profit growth {pg}% < {MIN_PROFIT_GROWTH_PCT}%"

    ocf = fund.get("operating_cashflow_cr")
    if ocf is not None and ocf <= 0:
        return False, f"Operating cash flow Rs.{ocf} Cr <= 0"

    promo = fund.get("promoter_holding_pct")
    if promo is not None and promo < MIN_PROMOTER_HOLDING_PCT:
        return False, f"Promoter holding {promo}% < {MIN_PROMOTER_HOLDING_PCT}%"

    pledged = fund.get("pledged_pct")
    if pledged is not None and pledged > MAX_PLEDGED_PCT:
        return False, f"Pledged shares {pledged}% > {MAX_PLEDGED_PCT}%"

    return True, ""


def _print_results(results: list[dict], near_miss: list[dict], total: int,
                   price_count: int, tech_count: int, start_time: float):
    """Print formatted results to console and save CSV."""
    print("\n" + "=" * 70)
    print("  SCREENING RESULTS")
    print("=" * 70)
    print(f"  Universe scanned : {total}")
    print(f"  Price data found : {price_count}")
    print(f"  Technical pass   : {tech_count}")
    print(f"  Final matches    : {len(results)}")
    print("-" * 70)

    # --- Main results ---
    if not results:
        print("\n  No stocks matched all criteria today.")
        print("  Consider relaxing filters (RSI range, ADX, volume multiplier, or growth thresholds).")
    else:
        df = _build_display_df(results)
        print("\n  === STOCKS PASSING ALL FILTERS ===\n")
        print(df.to_string(index=False))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = f"screener_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[+] Results saved to: {csv_path}")

    # --- Near-miss report ---
    if near_miss:
        print("\n" + "-" * 70)
        print("  === NEAR MISS: Technical pass, fundamental fail ===")
        print("  (These stocks have strong charts but failed a fundamental filter)\n")
        nm_df = _build_near_miss_df(near_miss)
        print(nm_df.to_string(index=False))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        nm_csv = f"near_miss_{timestamp}.csv"
        nm_df.to_csv(nm_csv, index=False)
        print(f"\n[+] Near-miss saved to: {nm_csv}")

    _print_elapsed(start_time)


def _build_display_df(results: list[dict]) -> pd.DataFrame:
    """Build the main results display DataFrame."""
    df = pd.DataFrame(results)

    display_cols = [
        "symbol", "price", "ema200", "ema50", "pct_above_ema200",
        "rsi", "adx", "crossover_date", "crossover_vol_ratio",
        "stop_loss", "sl_pct", "pct_from_52w_high",
        "avg_traded_value_cr", "market_cap_cr", "sales_growth_pct",
        "profit_growth_pct", "roe_pct", "debt_to_equity",
        "operating_cashflow_cr", "promoter_holding_pct",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols]

    col_names = {
        "symbol": "Symbol",
        "price": "CMP(Rs)",
        "ema200": "EMA200",
        "ema50": "EMA50",
        "pct_above_ema200": "%>EMA200",
        "rsi": "RSI(14)",
        "adx": "ADX",
        "crossover_date": "Crossover",
        "crossover_vol_ratio": "Vol Ratio",
        "stop_loss": "Stop Loss",
        "sl_pct": "SL %",
        "pct_from_52w_high": "%<52wH",
        "avg_traded_value_cr": "ADTV(Cr)",
        "market_cap_cr": "MCap(Cr)",
        "sales_growth_pct": "Sales G%",
        "profit_growth_pct": "Profit G%",
        "roe_pct": "ROE%",
        "debt_to_equity": "D/E",
        "operating_cashflow_cr": "OCF(Cr)",
        "promoter_holding_pct": "Promo%",
    }
    df = df.rename(columns=col_names)
    df = df.sort_values("Crossover", ascending=False)
    return df


def _build_near_miss_df(near_miss: list[dict]) -> pd.DataFrame:
    """Build the near-miss display DataFrame."""
    df = pd.DataFrame(near_miss)

    display_cols = [
        "symbol", "price", "rsi", "adx", "crossover_date",
        "crossover_vol_ratio", "stop_loss", "sl_pct", "fail_reason",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols]

    col_names = {
        "symbol": "Symbol",
        "price": "CMP(Rs)",
        "rsi": "RSI(14)",
        "adx": "ADX",
        "crossover_date": "Crossover",
        "crossover_vol_ratio": "Vol Ratio",
        "stop_loss": "Stop Loss",
        "sl_pct": "SL %",
        "fail_reason": "Failed Because",
    }
    df = df.rename(columns=col_names)
    return df


def _print_elapsed(start_time: float):
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  Total time: {mins}m {secs}s")
    print("=" * 70)


if __name__ == "__main__":
    run_screener()
