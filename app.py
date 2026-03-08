"""
Streamlit Dashboard for Indian Equity Swing Trading Screener
=============================================================
Run:  streamlit run app.py
"""

import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    EMA_LONG, EMA_SHORT, CROSSOVER_LOOKBACK, VOLUME_MULTIPLIER,
    VOLUME_AVG_PERIOD, RSI_PERIOD, RSI_LOWER, RSI_UPPER,
    MAX_ABOVE_EMA_PCT, ADX_MIN, ATR_SL_MULTIPLIER,
    MIN_MARKET_CAP_CR, MIN_SALES_GROWTH_PCT, MIN_PROFIT_GROWTH_PCT,
    MIN_ROE_PCT, MAX_DEBT_TO_EQUITY, MIN_PROMOTER_HOLDING_PCT,
    MAX_PLEDGED_PCT, MIN_AVG_TRADED_VALUE_CR, CRORE, MAX_WORKERS,
)
from src.stock_universe import fetch_nifty500_tickers, fetch_nifty250_tickers
from src.data_fetcher import fetch_bulk_price_data, fetch_fundamentals
from src.technicals import screen_technical, screen_ema200_breakout, calc_ema, calc_rsi, calc_atr, calc_adx

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swing Screener - Indian Equities",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.title("Screener Filters")

# Strategy selector
STRATEGY_SWING = "Swing Trade (EMA Crossover + ADX)"
STRATEGY_BREAKOUT = "EMA 200 Breakout (Nifty 250)"
strategy = st.sidebar.radio(
    "Strategy",
    [STRATEGY_SWING, STRATEGY_BREAKOUT],
    help="**Swing Trade**: Nifty 500, recent 200 EMA crossover with volume spike, RSI 50-70, ADX > 20.\n\n"
         "**EMA 200 Breakout**: Nifty 250, recent 200 EMA crossover + good fundamentals. "
         "Less strict than swing (no RSI/ADX filters). Catches NMDC-type breakout moves.",
)

is_swing = (strategy == STRATEGY_SWING)

# Technical filters (only for Swing strategy)
if is_swing:
    st.sidebar.subheader("Technical")
    f_rsi_lower = st.sidebar.slider("RSI Lower", 30, 60, RSI_LOWER)
    f_rsi_upper = st.sidebar.slider("RSI Upper", 60, 85, RSI_UPPER)
    f_adx_min = st.sidebar.slider("Min ADX", 10, 40, ADX_MIN)
    f_vol_mult = st.sidebar.slider("Volume Multiplier", 1.0, 5.0, VOLUME_MULTIPLIER, 0.1)
    f_max_above_ema = st.sidebar.slider("Max % above EMA200", 3.0, 15.0, MAX_ABOVE_EMA_PCT, 0.5)
    f_lookback = st.sidebar.slider("Crossover Lookback (days)", 2, 30, CROSSOVER_LOOKBACK)
else:
    st.sidebar.subheader("Technical")
    f_max_above_ema = st.sidebar.slider("Max % above EMA200", 3.0, 25.0, 15.0, 0.5)
    f_lookback = st.sidebar.slider("Crossover Lookback (days)", 0, 7, 7)

st.sidebar.subheader("Fundamental")
f_mcap = st.sidebar.number_input("Min Market Cap (Cr)", 100, 50000, int(MIN_MARKET_CAP_CR), 100)
f_sales_g = st.sidebar.slider("Min Sales Growth %", 0.0, 30.0, MIN_SALES_GROWTH_PCT, 1.0)
f_profit_g = st.sidebar.slider("Min Profit Growth %", 0.0, 30.0, MIN_PROFIT_GROWTH_PCT, 1.0)
f_roe = st.sidebar.slider("Min ROE %", 0.0, 30.0, MIN_ROE_PCT, 1.0)
f_de = st.sidebar.slider("Max D/E", 0.0, 2.0, MAX_DEBT_TO_EQUITY, 0.1)
f_promo = st.sidebar.slider("Min Promoter Holding %", 20.0, 75.0, MIN_PROMOTER_HOLDING_PCT, 1.0)

st.sidebar.subheader("Liquidity")
f_adtv = st.sidebar.number_input("Min ADTV (Cr)", 1.0, 50.0, MIN_AVG_TRADED_VALUE_CR, 0.5)


# ── Helper: check fundamentals with sidebar values ───────────────────────────
def check_fundamentals(fund: dict) -> tuple[bool, str]:
    mcap = fund.get("market_cap_cr", 0)
    if mcap < f_mcap:
        return False, f"MCap Rs.{mcap} Cr < {f_mcap}"
    roe = fund.get("roe_pct")
    if roe is not None and roe < f_roe:
        return False, f"ROE {roe}% < {f_roe}%"
    de = fund.get("debt_to_equity")
    if de is not None and de > f_de:
        return False, f"D/E {de} > {f_de}"
    sg = fund.get("sales_growth_pct")
    if sg is not None and sg < f_sales_g:
        return False, f"Sales growth {sg}% < {f_sales_g}%"
    pg = fund.get("profit_growth_pct")
    if pg is not None and pg < f_profit_g:
        return False, f"Profit growth {pg}% < {f_profit_g}%"
    ocf = fund.get("operating_cashflow_cr")
    if ocf is not None and ocf <= 0:
        return False, f"OCF Rs.{ocf} Cr <= 0"
    promo = fund.get("promoter_holding_pct")
    if promo is not None and promo < f_promo:
        return False, f"Promoter {promo}% < {f_promo}%"
    pledged = fund.get("pledged_pct")
    if pledged is not None and pledged > MAX_PLEDGED_PCT:
        return False, f"Pledged {pledged}% > {MAX_PLEDGED_PCT}%"
    return True, ""


# ── Helper: screen with sidebar values ───────────────────────────────────────
def screen_with_params(df: pd.DataFrame) -> dict | None:
    """screen_technical using sidebar filter values."""
    if df is None or len(df) < EMA_LONG + 10:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ema200 = calc_ema(close, EMA_LONG)
    ema50 = calc_ema(close, EMA_SHORT)
    rsi = calc_rsi(close, RSI_PERIOD)
    avg_vol_20 = volume.rolling(VOLUME_AVG_PERIOD).mean()
    atr = calc_atr(high, low, close)
    adx = calc_adx(high, low, close)

    cp = close.iloc[-1]
    ce200 = ema200.iloc[-1]
    ce50 = ema50.iloc[-1]
    cr = rsi.iloc[-1]
    ca = atr.iloc[-1]
    cadx = adx.iloc[-1]

    if cp <= ce200 or cp <= ce50:
        return None
    pct = ((cp - ce200) / ce200) * 100
    if pct > f_max_above_ema:
        return None
    if np.isnan(cr) or cr < f_rsi_lower or cr > f_rsi_upper:
        return None
    if np.isnan(cadx) or cadx < f_adx_min:
        return None

    crossover_found = False
    crossover_date = None
    crossover_vol_ratio = 0.0
    days_since_crossover = None
    lookback_start = max(len(df) - f_lookback, 1)
    for i in range(lookback_start, len(df)):
        if close.iloc[i] > ema200.iloc[i] and close.iloc[i - 1] <= ema200.iloc[i - 1]:
            vol_on_day = volume.iloc[i]
            avg_vol = avg_vol_20.iloc[i]
            if not np.isnan(avg_vol) and avg_vol > 0:
                ratio = vol_on_day / avg_vol
                if ratio >= f_vol_mult:
                    crossover_found = True
                    crossover_date = df.index[i]
                    crossover_vol_ratio = round(ratio, 2)
                    days_since_crossover = len(df) - 1 - i
                    break
    if not crossover_found:
        return None

    recent_20 = df.tail(VOLUME_AVG_PERIOD)
    avg_traded_value = (recent_20["Close"] * recent_20["Volume"]).mean()

    atr_sl = round(cp - ATR_SL_MULTIPLIER * ca, 2)
    swing_sl = round(float(low.iloc[-10:].min()), 2)
    stop_loss = max(atr_sl, swing_sl)
    sl_pct = round(((cp - stop_loss) / cp) * 100, 2)
    high_52w = high.max()
    pct_52w = round(((high_52w - cp) / high_52w) * 100, 2)

    return {
        "price": round(cp, 2), "ema200": round(ce200, 2), "ema50": round(ce50, 2),
        "pct_above_ema200": round(pct, 2), "rsi": round(cr, 2), "adx": round(cadx, 2),
        "atr": round(ca, 2),
        "crossover_date": crossover_date.strftime("%Y-%m-%d") if hasattr(crossover_date, "strftime") else str(crossover_date),
        "crossover_vol_ratio": crossover_vol_ratio, "days_since_crossover": days_since_crossover,
        "avg_traded_value": avg_traded_value,
        "stop_loss": stop_loss, "sl_pct": sl_pct, "pct_from_52w_high": pct_52w,
    }


# ── Helper: breakout screen with sidebar values ──────────────────────────────
def screen_breakout_with_params(df: pd.DataFrame) -> dict | None:
    """EMA 200 breakout screen using sidebar filter values."""
    return screen_ema200_breakout(df, max_above_ema_pct=f_max_above_ema, lookback=f_lookback)


# ── Helper: candlestick chart ────────────────────────────────────────────────
def make_chart(symbol: str, df: pd.DataFrame) -> go.Figure:
    """Build candlestick + EMA + volume chart for a stock."""
    close = df["Close"]
    ema200 = calc_ema(close, EMA_LONG)
    ema50 = calc_ema(close, EMA_SHORT)
    ema20 = calc_ema(close, 20)
    rsi = calc_rsi(close)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{symbol} - Daily Chart", "Volume", "RSI (14)"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # EMAs
    for ema, name, color in [
        (ema20, "EMA 20", "#ffab40"),
        (ema50, "EMA 50", "#42a5f5"),
        (ema200, "EMA 200", "#ab47bc"),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=ema, name=name, line=dict(width=1.5, color=color),
        ), row=1, col=1)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.7,
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI", line=dict(color="#ff7043", width=1.5),
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=20),
        template="plotly_dark",
    )
    fig.update_xaxes(type="category", nticks=20, row=3, col=1)
    return fig


# ── Nifty Regime ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def get_nifty_regime():
    try:
        nifty = yf.download("^NSEI", period="365d", progress=False)
        if nifty.empty:
            return None
        close = nifty["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        ema200 = calc_ema(close, EMA_LONG)
        cur = float(close.iloc[-1])
        ema_val = float(ema200.iloc[-1])
        pct = round(((cur - ema_val) / ema_val) * 100, 2)
        return {
            "price": round(cur, 2), "ema200": round(ema_val, 2),
            "pct": pct, "bullish": cur > ema_val,
        }
    except Exception:
        return None


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Indian Equity Screener")
universe_label = "Nifty 500" if is_swing else "Nifty 250"
strategy_label = "EMA Crossover + ADX" if is_swing else "EMA 200 Breakout"
st.caption(f"Scan date: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
           f"{universe_label} universe  |  {strategy_label} strategy")

# Regime banner
regime = get_nifty_regime()
if regime:
    color = "green" if regime["bullish"] else "red"
    label = "BULLISH" if regime["bullish"] else "BEARISH"
    st.markdown(
        f"**Nifty 50 Regime:** :{'green' if regime['bullish'] else 'red'}[{label}] "
        f"&mdash; {regime['price']} ({regime['pct']:+.2f}% vs 200 EMA at {regime['ema200']})"
    )
    if not regime["bullish"]:
        st.warning("Nifty 50 is below its 200 EMA. Swing longs carry higher risk. "
                   "Consider smaller position sizes.")

st.divider()

# Run button
if st.button("Run Screener", type="primary", use_container_width=True):
    # ── Step 1: Tickers ──────────────────────────────────────────────────
    with st.spinner(f"Fetching {universe_label} ticker list..."):
        symbols = fetch_nifty500_tickers() if is_swing else fetch_nifty250_tickers()
    if not symbols:
        st.error(f"Could not fetch {universe_label} list. Check your connection.")
        st.stop()

    # ── Step 2: Price data ───────────────────────────────────────────────
    progress = st.progress(0, text="Downloading price data...")
    price_data = fetch_bulk_price_data(symbols)
    progress.progress(50, text=f"Price data for {len(price_data)} stocks. Applying technical filters...")

    # ── Step 3: Technical screening ──────────────────────────────────────
    tech_passed = {}
    for sym, df in price_data.items():
        result = screen_with_params(df) if is_swing else screen_breakout_with_params(df)
        if result is not None:
            avg_tv_cr = result["avg_traded_value"] / CRORE
            if avg_tv_cr >= f_adtv:
                result["avg_traded_value_cr"] = round(avg_tv_cr, 2)
                tech_passed[sym] = result

    progress.progress(70, text=f"Technical pass: {len(tech_passed)} stocks. Fetching fundamentals...")

    # ── Step 4: Fundamental screening ────────────────────────────────────
    final_results = []
    near_miss = []

    if tech_passed:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_fundamentals, sym): sym for sym in tech_passed}
            done = 0
            for future in as_completed(futures):
                sym = futures[future]
                done += 1
                pct_done = 70 + int(30 * done / len(tech_passed))
                progress.progress(pct_done, text=f"Fundamentals: {done}/{len(tech_passed)} ({sym})")

                fund = future.result()
                if fund is None:
                    miss_row = {"symbol": sym, "fail_reason": "No data"}
                    miss_row.update(tech_passed[sym])
                    near_miss.append(miss_row)
                    continue

                passed, reason = check_fundamentals(fund)
                row = {"symbol": sym}
                row.update(tech_passed[sym])
                row.update(fund)
                if passed:
                    final_results.append(row)
                else:
                    row["fail_reason"] = reason
                    near_miss.append(row)

    progress.progress(100, text="Done!")
    time.sleep(0.3)
    progress.empty()

    # ── Store results in session state ────────────────────────────────────
    st.session_state["results"] = final_results
    st.session_state["near_miss"] = near_miss
    st.session_state["price_data"] = price_data
    st.session_state["strategy"] = strategy
    st.session_state["stats"] = {
        "universe": len(symbols),
        "price_ok": len(price_data),
        "tech_pass": len(tech_passed),
        "final": len(final_results),
    }

# ── Display Results ──────────────────────────────────────────────────────────
if "stats" in st.session_state:
    s = st.session_state["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe", s["universe"])
    c2.metric("Price Data", s["price_ok"])
    c3.metric("Technical Pass", s["tech_pass"])
    c4.metric("Final Matches", s["final"])
    st.divider()

if "results" in st.session_state:
    results = st.session_state["results"]
    near_miss = st.session_state["near_miss"]
    price_data = st.session_state["price_data"]
    last_strategy = st.session_state.get("strategy", STRATEGY_SWING)
    last_is_swing = (last_strategy == STRATEGY_SWING)

    # ── Main results table ───────────────────────────────────────────────
    if results:
        st.subheader("Stocks Passing All Filters")
        res_df = pd.DataFrame(results)

        # Columns vary by strategy
        if last_is_swing:
            display_cols = [
                "symbol", "price", "ema200", "ema50", "pct_above_ema200",
                "rsi", "adx", "crossover_date", "days_since_crossover",
                "crossover_vol_ratio", "stop_loss", "sl_pct",
                "pct_from_52w_high", "avg_traded_value_cr",
                "market_cap_cr", "sales_growth_pct", "profit_growth_pct",
                "roe_pct", "debt_to_equity", "operating_cashflow_cr",
                "promoter_holding_pct",
            ]
        else:
            display_cols = [
                "symbol", "price", "ema200", "ema50", "pct_above_ema200",
                "rsi", "crossover_date", "days_since_crossover",
                "crossover_vol_ratio", "stop_loss", "sl_pct",
                "pct_from_52w_high", "avg_traded_value_cr",
                "market_cap_cr", "sales_growth_pct", "profit_growth_pct",
                "roe_pct", "debt_to_equity", "operating_cashflow_cr",
                "promoter_holding_pct",
            ]

        display_cols = [c for c in display_cols if c in res_df.columns]
        col_rename = {
            "symbol": "Symbol", "price": "CMP", "ema200": "EMA200",
            "ema50": "EMA50", "pct_above_ema200": "%>EMA200",
            "rsi": "RSI", "adx": "ADX", "crossover_date": "Crossover",
            "days_since_crossover": "Days", "crossover_vol_ratio": "VolRatio",
            "stop_loss": "SL", "sl_pct": "SL%", "pct_from_52w_high": "%<52wH",
            "avg_traded_value_cr": "ADTV(Cr)", "market_cap_cr": "MCap(Cr)",
            "sales_growth_pct": "SalesG%", "profit_growth_pct": "ProfitG%",
            "roe_pct": "ROE%", "debt_to_equity": "D/E",
            "operating_cashflow_cr": "OCF(Cr)", "promoter_holding_pct": "Promo%",
        }
        show_df = res_df[display_cols].rename(columns=col_rename)

        # Sort by days since crossover (freshest breakouts first)
        if "Days" in show_df.columns:
            show_df = show_df.sort_values("Days", ascending=True)
        elif "%>EMA200" in show_df.columns:
            show_df = show_df.sort_values("%>EMA200", ascending=True)

        st.dataframe(show_df, use_container_width=True, hide_index=True)

        # CSV download
        csv = show_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "screener_results.csv",
                           "text/csv", use_container_width=True)

        # ── Charts for each result ───────────────────────────────────────
        st.subheader("Charts")
        for row in results:
            sym = row["symbol"]
            if sym in price_data:
                days = row.get('days_since_crossover', '?')
                if last_is_swing:
                    label = (f"{sym} - Rs.{row['price']} | RSI {row['rsi']} | "
                             f"ADX {row.get('adx', 'N/A')} | Crossover {days}d ago | "
                             f"SL Rs.{row['stop_loss']}")
                else:
                    label = (f"{sym} - Rs.{row['price']} | {row['pct_above_ema200']}% > EMA200 | "
                             f"Crossover {days}d ago | VolRatio {row.get('crossover_vol_ratio', 'N/A')} | "
                             f"SL Rs.{row['stop_loss']}")
                with st.expander(label, expanded=True):
                    fig = make_chart(sym, price_data[sym].tail(120))
                    st.plotly_chart(fig, use_container_width=True)

                    # Trade setup box
                    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                    tc1.metric("Entry", f"Rs.{row['price']}")
                    tc2.metric("Stop Loss", f"Rs.{row['stop_loss']}", f"-{row['sl_pct']}%")
                    tc3.metric("ATR(14)", f"Rs.{row['atr']}")
                    tc4.metric("Crossover", f"{row.get('days_since_crossover', '?')}d ago",
                               f"Vol {row.get('crossover_vol_ratio', 'N/A')}x")
                    tc5.metric("% from 52W High", f"{row['pct_from_52w_high']}%")

    else:
        st.info("No stocks matched all criteria. Try relaxing filters in the sidebar.")

    # ── Near-miss table ──────────────────────────────────────────────────
    if near_miss:
        st.divider()
        st.subheader("Near Miss (Technical pass, Fundamental fail)")
        st.caption("Strong charts but failed a fundamental filter")
        nm_df = pd.DataFrame(near_miss)
        nm_cols = ["symbol", "price", "rsi", "adx", "crossover_date",
                   "days_since_crossover", "crossover_vol_ratio",
                   "stop_loss", "sl_pct", "fail_reason"]
        nm_cols = [c for c in nm_cols if c in nm_df.columns]
        nm_show = nm_df[nm_cols].rename(columns={
            "symbol": "Symbol", "price": "CMP", "rsi": "RSI", "adx": "ADX",
            "crossover_date": "Crossover", "days_since_crossover": "Days",
            "crossover_vol_ratio": "VolRatio",
            "stop_loss": "SL", "sl_pct": "SL%", "fail_reason": "Failed Because",
        })
        st.dataframe(nm_show, use_container_width=True, hide_index=True)

        # Charts for near-miss
        for row in near_miss:
            sym = row["symbol"]
            if sym in price_data:
                with st.expander(f"{sym} - Rs.{row['price']} | "
                                 f"Failed: {row.get('fail_reason', 'N/A')}"):
                    fig = make_chart(sym, price_data[sym].tail(120))
                    st.plotly_chart(fig, use_container_width=True)

elif "stats" not in st.session_state:
    st.info("Configure filters in the sidebar, then click **Run Screener** to start.")
