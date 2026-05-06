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
    VCP_MAX_FROM_52W_HIGH_PCT, VCP_MIN_ABOVE_52W_LOW_PCT, VCP_BASE_LENGTH,
    VCP_MIN_CONTRACTIONS, VCP_VOL_CONTRACTION_RATIO, VCP_BREAKOUT_VOL_MULT,
    VCP_PIVOT_PROXIMITY_PCT,
)
from src.stock_universe import fetch_nifty500_tickers, fetch_nifty250_tickers
from src.data_fetcher import fetch_bulk_price_data, fetch_fundamentals
from src.technicals import screen_technical, screen_ema200_breakout, screen_vcp, calc_ema, calc_rsi, calc_atr, calc_adx
from src.signal_tracker import (
    record_signal, check_and_update_signals,
    get_active_signals, get_active_signals_with_live_prices,
    get_closed_signals, get_performance_stats, manually_close_signal,
    get_backend_info, export_signals_json, import_signals_json,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swing Screener - Indian Equities",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* --- Global --- */
    .block-container { padding-top: 1.5rem; }

    /* --- Metric cards --- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8892b0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6f1ff !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stRadio > label,
    section[data-testid="stSidebar"] .stSlider > label,
    section[data-testid="stSidebar"] .stNumberInput > label {
        color: #c9d1d9 !important;
    }

    /* --- Tables --- */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* --- Buttons --- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.03em;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00b4d8 0%, #90e0ef 100%);
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3);
    }

    /* --- Expander headers --- */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* --- Section headers --- */
    h2 {
        color: #e6f1ff !important;
        border-bottom: 2px solid #0077b6;
        padding-bottom: 0.3rem;
    }
    h3 {
        color: #ccd6f6 !important;
    }

    /* --- Dividers --- */
    hr {
        border-color: #21262d !important;
    }

    /* --- Download button --- */
    .stDownloadButton > button {
        background: transparent;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #58a6ff !important;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background: rgba(88, 166, 255, 0.1);
        border-color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.title("Screener Filters")

# Strategy selector
STRATEGY_SWING = "Swing Trade (EMA Crossover + ADX)"
STRATEGY_BREAKOUT = "EMA 200 Breakout (Nifty 250)"
STRATEGY_VCP = "VCP - Volatility Contraction"
strategy = st.sidebar.radio(
    "Strategy",
    [STRATEGY_SWING, STRATEGY_BREAKOUT, STRATEGY_VCP],
    help="**Swing Trade**: Nifty 500, recent 200 EMA crossover with volume spike, RSI 50-70, ADX > 20.\n\n"
         "**EMA 200 Breakout**: Nifty 250, recent 200 EMA crossover + good fundamentals. "
         "Less strict than swing (no RSI/ADX filters). Catches NMDC-type breakout moves.\n\n"
         "**VCP**: Minervini-style pattern. Stocks near 52W highs with tightening "
         "price contractions and volume dry-up, ready to break out.",
)

is_swing = (strategy == STRATEGY_SWING)
is_vcp = (strategy == STRATEGY_VCP)

# Technical filters vary by strategy
st.sidebar.subheader("Technical")
if is_swing:
    f_rsi_lower = st.sidebar.slider("RSI Lower", 30, 60, RSI_LOWER)
    f_rsi_upper = st.sidebar.slider("RSI Upper", 60, 85, RSI_UPPER)
    f_adx_min = st.sidebar.slider("Min ADX", 10, 40, ADX_MIN)
    f_vol_mult = st.sidebar.slider("Volume Multiplier", 1.0, 5.0, VOLUME_MULTIPLIER, 0.1)
    f_max_above_ema = st.sidebar.slider("Max % above EMA200", 3.0, 15.0, MAX_ABOVE_EMA_PCT, 0.5)
    f_lookback = st.sidebar.slider("Crossover Lookback (days)", 2, 30, CROSSOVER_LOOKBACK)
elif is_vcp:
    f_vcp_max_from_high = st.sidebar.slider("Max % from 52W High", 5.0, 40.0, VCP_MAX_FROM_52W_HIGH_PCT, 1.0)
    f_vcp_min_above_low = st.sidebar.slider("Min % above 52W Low", 10.0, 50.0, VCP_MIN_ABOVE_52W_LOW_PCT, 5.0)
    f_vcp_base_length = st.sidebar.slider("Base Length (days)", 30, 180, VCP_BASE_LENGTH, 10)
    f_vcp_vol_contraction = st.sidebar.slider("Vol Contraction Ratio", 0.3, 1.0, VCP_VOL_CONTRACTION_RATIO, 0.05)
    f_vcp_pivot_proximity = st.sidebar.slider("Max % below Pivot", 1.0, 10.0, VCP_PIVOT_PROXIMITY_PCT, 0.5)
    f_vcp_breakout_vol = st.sidebar.slider("Breakout Vol Multiplier", 1.0, 4.0, VCP_BREAKOUT_VOL_MULT, 0.1)
else:
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
    reasons = []
    mcap = fund.get("market_cap_cr", 0)
    if mcap < f_mcap:
        reasons.append(f"MCap Rs.{mcap} Cr < {f_mcap}")
    roe = fund.get("roe_pct")
    if roe is not None and roe < f_roe:
        reasons.append(f"ROE {roe}% < {f_roe}%")
    de = fund.get("debt_to_equity")
    if de is not None and de > f_de:
        reasons.append(f"D/E {de} > {f_de}")
    sg = fund.get("sales_growth_pct")
    if sg is not None and sg < f_sales_g:
        reasons.append(f"Sales growth {sg}% < {f_sales_g}%")
    pg = fund.get("profit_growth_pct")
    if pg is not None and pg < f_profit_g:
        reasons.append(f"Profit growth {pg}% < {f_profit_g}%")
    ocf = fund.get("operating_cashflow_cr")
    if ocf is not None and ocf <= 0:
        reasons.append(f"OCF Rs.{ocf} Cr <= 0")
    promo = fund.get("promoter_holding_pct")
    if promo is not None and promo < f_promo:
        reasons.append(f"Promoter {promo}% < {f_promo}%")
    pledged = fund.get("pledged_pct")
    if pledged is not None and pledged > MAX_PLEDGED_PCT:
        reasons.append(f"Pledged {pledged}% > {MAX_PLEDGED_PCT}%")
    if reasons:
        return False, "; ".join(reasons)
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


# ── Helper: VCP screen with sidebar values ───────────────────────────────────
def screen_vcp_with_params(df: pd.DataFrame) -> dict | None:
    """VCP screen using sidebar filter values."""
    return screen_vcp(
        df,
        max_from_high_pct=f_vcp_max_from_high,
        min_above_low_pct=f_vcp_min_above_low,
        base_length=f_vcp_base_length,
        vol_contraction_ratio=f_vcp_vol_contraction,
        breakout_vol_mult=f_vcp_breakout_vol,
        pivot_proximity_pct=f_vcp_pivot_proximity,
    )


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
        paper_bgcolor="rgba(13,17,23,0.8)",
        plot_bgcolor="rgba(22,27,34,0.9)",
        font=dict(color="#c9d1d9"),
    )
    fig.update_xaxes(gridcolor="rgba(48,54,61,0.5)", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="rgba(48,54,61,0.5)", zerolinecolor="#30363d")
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
if is_swing:
    universe_label, strategy_label = "Nifty 500", "EMA Crossover + ADX"
elif is_vcp:
    universe_label, strategy_label = "Nifty 500", "VCP (Volatility Contraction)"
else:
    universe_label, strategy_label = "Nifty 250", "EMA 200 Breakout"

# Header
st.markdown(f"""
<div style="padding: 0.5rem 0 0.8rem 0;">
    <h1 style="margin:0; color:#e6f1ff; font-size:2rem; font-weight:800; letter-spacing:-0.02em;">
        Indian Equity Screener
    </h1>
    <p style="margin:0.3rem 0 0 0; color:#8892b0; font-size:0.9rem;">
        {datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;&bull;&nbsp;
        {universe_label} Universe &nbsp;&bull;&nbsp;
        {strategy_label}
    </p>
</div>
""", unsafe_allow_html=True)

# Regime banner
regime = get_nifty_regime()
if regime:
    if regime["bullish"]:
        regime_color = "#26a69a"
        regime_bg = "rgba(38,166,154,0.08)"
        regime_border = "rgba(38,166,154,0.3)"
        regime_label = "BULLISH"
        regime_icon = "trending_up"
    else:
        regime_color = "#ef5350"
        regime_bg = "rgba(239,83,80,0.08)"
        regime_border = "rgba(239,83,80,0.3)"
        regime_label = "BEARISH"
        regime_icon = "trending_down"
    st.markdown(f"""
    <div style="background:{regime_bg}; border:1px solid {regime_border}; border-radius:8px;
                padding:12px 18px; margin-bottom:0.5rem; display:flex; align-items:center; gap:12px;">
        <span style="background:{regime_color}; color:#fff; font-weight:700; font-size:0.75rem;
                     padding:4px 10px; border-radius:4px; letter-spacing:0.08em;">
            {regime_label}
        </span>
        <span style="color:#c9d1d9; font-size:0.9rem;">
            <strong>Nifty 50</strong> &nbsp;{regime['price']}
            &nbsp;<span style="color:{regime_color};">({regime['pct']:+.2f}%)</span>
            &nbsp;vs 200 EMA at {regime['ema200']}
        </span>
    </div>
    """, unsafe_allow_html=True)
    if not regime["bullish"]:
        st.warning("Nifty 50 is below its 200 EMA. Swing longs carry higher risk. "
                   "Consider smaller position sizes.")

st.divider()

# Run button
if st.button("Run Screener", type="primary", use_container_width=True):
    # ── Step 1: Tickers ──────────────────────────────────────────────────
    with st.spinner(f"Fetching {universe_label} ticker list..."):
        if is_swing or is_vcp:
            symbols = fetch_nifty500_tickers()
        else:
            symbols = fetch_nifty250_tickers()
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
        if is_swing:
            result = screen_with_params(df)
        elif is_vcp:
            result = screen_vcp_with_params(df)
        else:
            result = screen_breakout_with_params(df)
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
                    # Fundamentals unavailable — still include with a note
                    row = {"symbol": sym}
                    row.update(tech_passed[sym])
                    row["fund_note"] = "Fundamental data unavailable"
                    final_results.append(row)
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

    # ── Auto-save new signals to tracker ─────────────────────────────────
    new_signals = 0
    for row in final_results:
        saved = record_signal(
            symbol=row["symbol"],
            strategy=strategy,
            entry_price=row["price"],
            stop_loss=row["stop_loss"],
            sl_pct=row["sl_pct"],
        )
        if saved:
            new_signals += 1
    if new_signals:
        st.toast(f"{new_signals} new signal(s) saved to tracker")
        # Mark that the cache needs refresh
        st.session_state["_signals_cache_bust"] = st.session_state.get("_signals_cache_bust", 0) + 1

# ── Display Results ──────────────────────────────────────────────────────────
if "stats" in st.session_state:
    s = st.session_state["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe", s["universe"])
    c2.metric("Price Data", s["price_ok"], f"{s['price_ok']}/{s['universe']}")
    c3.metric("Technical Pass", s["tech_pass"],
              f"{round(100 * s['tech_pass'] / max(s['price_ok'], 1), 1)}% pass rate")
    c4.metric("Final Matches", s["final"],
              f"{round(100 * s['final'] / max(s['tech_pass'], 1), 1)}% of tech pass" if s["tech_pass"] else "0")
    st.divider()

if "results" in st.session_state:
    results = st.session_state["results"]
    near_miss = st.session_state["near_miss"]
    price_data = st.session_state["price_data"]
    last_strategy = st.session_state.get("strategy", STRATEGY_SWING)
    last_is_swing = (last_strategy == STRATEGY_SWING)
    last_is_vcp = (last_strategy == STRATEGY_VCP)

    # ── Main results table ───────────────────────────────────────────────
    if results:
        st.subheader(f"Final Matches ({len(results)})")
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
        elif last_is_vcp:
            display_cols = [
                "symbol", "price", "vcp_status", "pivot_high", "pct_below_pivot",
                "pct_from_52w_high", "rsi",
                "contraction_ratio", "vol_contraction", "breakout_vol_ratio",
                "base_depth_pct", "stop_loss", "sl_pct",
                "avg_traded_value_cr",
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
            # VCP-specific
            "vcp_status": "Status", "pivot_high": "Pivot", "pct_below_pivot": "%<Pivot",
            "contraction_ratio": "Contraction", "vol_contraction": "VolDryUp",
            "breakout_vol_ratio": "BrkoutVol", "base_depth_pct": "BaseDepth%",
        }
        show_df = res_df[display_cols].rename(columns=col_rename)

        # Sort
        if last_is_vcp:
            if "%<Pivot" in show_df.columns:
                show_df = show_df.sort_values("%<Pivot", ascending=True)
        elif "Days" in show_df.columns:
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
                if last_is_swing:
                    days = row.get('days_since_crossover', '?')
                    label = (f"{sym} - Rs.{row['price']} | RSI {row['rsi']} | "
                             f"ADX {row.get('adx', 'N/A')} | Crossover {days}d ago | "
                             f"SL Rs.{row['stop_loss']}")
                elif last_is_vcp:
                    label = (f"{sym} - Rs.{row['price']} | {row['vcp_status']} | "
                             f"Pivot Rs.{row['pivot_high']} ({row['pct_below_pivot']}% below) | "
                             f"SL Rs.{row['stop_loss']}")
                else:
                    days = row.get('days_since_crossover', '?')
                    label = (f"{sym} - Rs.{row['price']} | {row.get('pct_above_ema200', '?')}% > EMA200 | "
                             f"Crossover {days}d ago | VolRatio {row.get('crossover_vol_ratio', 'N/A')} | "
                             f"SL Rs.{row['stop_loss']}")
                with st.expander(label, expanded=True):
                    fig = make_chart(sym, price_data[sym].tail(120))
                    st.plotly_chart(fig, use_container_width=True)

                    # Trade setup box - varies by strategy
                    if last_is_vcp:
                        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                        tc1.metric("Entry", f"Rs.{row['price']}")
                        tc2.metric("Stop Loss", f"Rs.{row['stop_loss']}", f"-{row['sl_pct']}%")
                        tc3.metric("Pivot", f"Rs.{row['pivot_high']}",
                                   f"{row['pct_below_pivot']}% below")
                        tc4.metric("Vol Dry-Up", f"{row['vol_contraction']}x",
                                   f"Brkout {row['breakout_vol_ratio']}x")
                        tc5.metric("% from 52W High", f"{row['pct_from_52w_high']}%")
                    else:
                        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                        tc1.metric("Entry", f"Rs.{row['price']}")
                        tc2.metric("Stop Loss", f"Rs.{row['stop_loss']}", f"-{row['sl_pct']}%")
                        tc3.metric("ATR(14)", f"Rs.{row['atr']}")
                        tc4.metric("Crossover", f"{row.get('days_since_crossover', '?')}d ago",
                                   f"Vol {row.get('crossover_vol_ratio', 'N/A')}x")
                        tc5.metric("% from 52W High", f"{row['pct_from_52w_high']}%")

    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem; background:rgba(239,83,80,0.06);
                    border:1px solid rgba(239,83,80,0.2); border-radius:8px; margin:1rem 0;">
            <p style="font-size:1.1rem; color:#ef5350; font-weight:600; margin:0;">
                No stocks matched all criteria
            </p>
            <p style="color:#8892b0; margin:0.4rem 0 0 0; font-size:0.9rem;">
                Try relaxing the fundamental or technical filters in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Near-miss table ──────────────────────────────────────────────────
    if near_miss:
        st.divider()
        st.subheader(f"Near Miss ({len(near_miss)})")
        st.caption("Passed technical screening but failed one or more fundamental filters")
        nm_df = pd.DataFrame(near_miss)

        if last_is_vcp:
            nm_cols = ["symbol", "price", "vcp_status", "pivot_high", "pct_below_pivot",
                       "pct_from_52w_high", "rsi", "contraction_ratio",
                       "vol_contraction", "stop_loss", "sl_pct", "fail_reason"]
        else:
            nm_cols = ["symbol", "price", "pct_above_ema200", "rsi", "crossover_date",
                       "days_since_crossover", "crossover_vol_ratio",
                       "stop_loss", "sl_pct", "fail_reason"]
            if last_is_swing:
                nm_cols.insert(4, "adx")

        nm_cols = [c for c in nm_cols if c in nm_df.columns]
        nm_rename = {
            "symbol": "Symbol", "price": "CMP", "pct_above_ema200": "%>EMA200",
            "rsi": "RSI", "adx": "ADX",
            "crossover_date": "Crossover", "days_since_crossover": "Days",
            "crossover_vol_ratio": "VolRatio",
            "stop_loss": "SL", "sl_pct": "SL%", "fail_reason": "Failed Because",
            "vcp_status": "Status", "pivot_high": "Pivot", "pct_below_pivot": "%<Pivot",
            "pct_from_52w_high": "%<52wH", "contraction_ratio": "Contraction",
            "vol_contraction": "VolDryUp",
        }
        nm_show = nm_df[nm_cols].rename(columns=nm_rename)
        if last_is_vcp and "%<Pivot" in nm_show.columns:
            nm_show = nm_show.sort_values("%<Pivot", ascending=True)
        elif "Days" in nm_show.columns:
            nm_show = nm_show.sort_values("Days", ascending=True)
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
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#8892b0;">
        <p style="font-size:3rem; margin:0;">&#x1F50D;</p>
        <h3 style="color:#ccd6f6; margin:0.5rem 0;">Ready to Scan</h3>
        <p>Configure filters in the sidebar, then click <strong>Run Screener</strong> to start.</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL JOURNAL — persistent tracking of all signals until TP/SL hit
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<h2 style="margin-top:0.5rem;">Signal Journal</h2>
""", unsafe_allow_html=True)
st.caption("Signals are auto-saved when the screener runs. They are tracked until target or stop-loss is hit.")

# ── Backend status banner ──────────────────────────────────────────────────
backend = get_backend_info()
if backend["persistent"]:
    st.markdown(f"""
    <div style="background:rgba(38,166,154,0.08); border:1px solid rgba(38,166,154,0.3);
                border-radius:6px; padding:8px 14px; margin-bottom:0.5rem; font-size:0.85rem;">
        <span style="background:#26a69a; color:#fff; font-weight:700; font-size:0.7rem;
                     padding:2px 8px; border-radius:3px; letter-spacing:0.05em;">PERSISTENT</span>
        &nbsp; <strong>{backend['backend']}</strong> — {backend['info']}
    </div>
    """, unsafe_allow_html=True)
else:
    with st.expander("⚠️ Signals may be lost on Streamlit Cloud restarts — click for fix"):
        st.warning(
            f"**Backend:** {backend['backend']}\n\n"
            f"**Issue:** {backend['info']}\n\n"
            "**Permanent Fix (5 min):**\n\n"
            "1. Sign up at [supabase.com](https://supabase.com) (free)\n"
            "2. Create a new project, go to **Project Settings > Database**\n"
            "3. Copy the **Connection String (URI)** under 'Connection Pooling'\n"
            "4. Go to your Streamlit Cloud app → **Settings → Secrets**\n"
            "5. Add: `DATABASE_URL = \"postgresql://...your-string...\"`\n"
            "6. Save — your app will redeploy and use Postgres permanently\n\n"
            "**Quick Fix:** Use the Backup/Restore section below to manually save your signals."
        )

# ── Backup / Restore ────────────────────────────────────────────────────────
with st.expander("Backup / Restore Signals (JSON)"):
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("**Download Backup**")
        try:
            backup_json = export_signals_json()
            count_in_backup = backup_json.count('"id"') if backup_json != "[]" else 0
            st.download_button(
                f"Download {count_in_backup} signal(s)",
                backup_json,
                f"signals_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Backup failed: {e}")
    with bcol2:
        st.markdown("**Upload Backup**")
        uploaded = st.file_uploader("Choose a backup JSON", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                content = uploaded.read().decode("utf-8")
                imported = import_signals_json(content)
                st.success(f"Imported {imported} signal(s) (duplicates skipped)")
                st.rerun()
            except Exception as e:
                st.error(f"Restore failed: {e}")

# Check signals button
jcol1, jcol2 = st.columns([1, 3])
with jcol1:
    if st.button("Update Signals", use_container_width=True,
                 help="Check current prices for all active signals and update TP/SL status"):
        with st.spinner("Checking active signals against current prices..."):
            result = check_and_update_signals()
        if result["tp1_hits"]:
            st.success(f"TP1 Hit: {', '.join(result['tp1_hits'])}")
        if result["tp2_hits"]:
            st.success(f"TP2 Hit: {', '.join(result['tp2_hits'])}")
        if result["sl_hits"]:
            st.error(f"SL Hit: {', '.join(result['sl_hits'])}")
        if not result["tp1_hits"] and not result["tp2_hits"] and not result["sl_hits"]:
            st.info(f"Checked {result['updated']} signal(s). No TP/SL triggers.")

# Performance stats
try:
    stats = get_performance_stats()
    if stats["total_closed"] > 0 or stats["total_active"] > 0:
        st.markdown("#### Performance")
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("Active", stats["total_active"])
        pc2.metric("Closed", stats["total_closed"])
        pc3.metric("Win Rate", f"{stats['win_rate']}%",
                   f"{stats['wins']}W / {stats['losses']}L")
        pc4.metric("Avg Win", f"{stats['avg_win_pct']:+.1f}%")
        pc5.metric("Avg Loss", f"{stats['avg_loss_pct']:+.1f}%")
except Exception as e:
    st.warning(f"Could not load performance stats: {e}")

# Active signals table — fetches LIVE CMP for real-time unrealized P&L
ac1, ac2, ac3 = st.columns([2, 1, 1])
with ac1:
    st.markdown("#### Active Signals")
with ac2:
    show_live = st.toggle("Live CMP", value=True,
                          help="Fetch live prices from yfinance (slower)")
with ac3:
    refresh_live = st.button("Refresh", use_container_width=True,
                             help="Force refresh of live prices")

@st.cache_data(ttl=60, show_spinner=False)
def _cached_active_with_live(nonce: int = 0):
    try:
        return get_active_signals_with_live_prices()
    except Exception as e:
        st.warning(f"Could not fetch live prices: {e}. Showing entry data only.")
        return get_active_signals()

if refresh_live:
    _cached_active_with_live.clear()

# Bust cache when new signals are added
nonce = st.session_state.get("_signals_cache_bust", 0)

try:
    if show_live:
        with st.spinner("Fetching live prices..."):
            active_df = _cached_active_with_live(nonce)
    else:
        active_df = get_active_signals()
        # Add empty live columns so display logic works
        if not active_df.empty:
            active_df["cmp"] = active_df["entry_price"]
            active_df["live_pnl_pct"] = 0.0
            active_df["dist_to_sl_pct"] = active_df["sl_pct"]
            active_df["dist_to_tp1_pct"] = round(
                ((active_df["target_1"] - active_df["entry_price"]) / active_df["entry_price"]) * 100, 2
            )
except Exception as e:
    st.error(f"Error loading active signals: {e}")
    active_df = pd.DataFrame()

if not active_df.empty:
    try:
        # Compute days since signal date
        today = datetime.now().date()
        active_df["signal_dt"] = pd.to_datetime(active_df["signal_date"]).dt.date
        active_df["age"] = active_df["signal_dt"].apply(
            lambda d: "TODAY" if d == today else f"{(today - d).days}d ago"
        )

        # Strategy short labels (before reordering)
        strategy_short = active_df["strategy"].astype(str).replace({
            "Swing Trade (EMA Crossover + ADX)": "Swing",
            "EMA 200 Breakout (Nifty 250)": "Breakout",
            "VCP - Volatility Contraction": "VCP",
        })
        active_df["strategy_short"] = strategy_short

        # Display columns - prioritize live data
        display_cols = [
            "age", "symbol", "strategy_short", "entry_price", "cmp", "live_pnl_pct",
            "stop_loss", "dist_to_sl_pct", "target_1", "dist_to_tp1_pct", "target_2",
            "high_since_entry", "mfe_pct", "low_since_entry", "mae_pct",
            "signal_date",
        ]
        # Only keep columns that exist
        display_cols = [c for c in display_cols if c in active_df.columns]
        active_display = active_df[display_cols].copy()

        rename_map = {
            "age": "Age", "symbol": "Symbol", "strategy_short": "Strategy",
            "entry_price": "Entry", "cmp": "CMP", "live_pnl_pct": "Live P&L%",
            "stop_loss": "SL", "dist_to_sl_pct": "% to SL",
            "target_1": "TP1", "dist_to_tp1_pct": "% to TP1", "target_2": "TP2",
            "high_since_entry": "High Since", "mfe_pct": "MFE%",
            "low_since_entry": "Low Since", "mae_pct": "MAE%",
            "signal_date": "Signal Date",
        }
        active_display = active_display.rename(columns=rename_map)

        # Color-code with pandas Styler (use .map for newer pandas, fall back to .applymap)
        def _color_pnl(val):
            try:
                v = float(val)
                if v > 0:
                    return "color: #26a69a; font-weight: 600;"
                elif v < 0:
                    return "color: #ef5350; font-weight: 600;"
            except (ValueError, TypeError):
                pass
            return ""

        def _color_age(val):
            if val == "TODAY":
                return "background-color: rgba(38,166,154,0.15); color: #26a69a; font-weight: 700;"
            return ""

        try:
            styler = active_display.style
            # pandas 2.1+ uses .map, older uses .applymap
            if hasattr(styler, "map"):
                styled = styler.map(_color_pnl, subset=["Live P&L%"]) \
                               .map(_color_age, subset=["Age"])
            else:
                styled = styler.applymap(_color_pnl, subset=["Live P&L%"]) \
                               .applymap(_color_age, subset=["Age"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            # Fall back to plain dataframe if styling fails
            st.dataframe(active_display, use_container_width=True, hide_index=True)

        if show_live:
            st.caption(f"CMP cached for 60s. Showing {len(active_df)} active signal(s). "
                       f"Live P&L = (CMP - Entry) / Entry %. Click 'Refresh' for fresh prices.")
        else:
            st.caption(f"Showing {len(active_df)} active signal(s). "
                       f"Toggle 'Live CMP' on for real-time P&L.")

        # Manual close option
        with st.expander("Manually close a signal"):
            signal_options = {f"{r['symbol']} (Entry: Rs.{r['entry_price']} on {r['signal_date']})": r['id']
                              for _, r in active_df.iterrows()}
            if signal_options:
                selected = st.selectbox("Select signal to close", list(signal_options.keys()))
                close_reason = st.text_input("Reason", "Manual close")
                if st.button("Close Signal"):
                    manually_close_signal(signal_options[selected], close_reason)
                    _cached_active_with_live.clear()
                    st.success(f"Closed signal for {selected}")
                    st.rerun()
    except Exception as e:
        st.error(f"Error displaying active signals: {e}")
else:
    st.info("No active signals. Run the screener to generate new signals.")

# Closed signals table
try:
    closed_df = get_closed_signals()
    if not closed_df.empty:
        st.markdown("#### Closed Signals (History)")
        closed_display = closed_df[[
            "symbol", "strategy", "entry_price", "exit_price", "pnl_pct",
            "status", "exit_reason", "signal_date", "exit_date",
        ]].copy()
        closed_display.columns = [
            "Symbol", "Strategy", "Entry", "Exit", "P&L%",
            "Result", "Reason", "Signal Date", "Exit Date",
        ]
        st.dataframe(closed_display, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Could not load closed signals: {e}")
