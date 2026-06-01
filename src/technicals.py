"""Technical indicator calculations for the stock screener."""

import numpy as np
import pandas as pd

from src.config import (
    EMA_LONG, EMA_SHORT, CROSSOVER_LOOKBACK, VOLUME_MULTIPLIER,
    VOLUME_AVG_PERIOD, RSI_PERIOD, RSI_LOWER, RSI_UPPER, MAX_ABOVE_EMA_PCT,
    ADX_PERIOD, ADX_MIN, ATR_PERIOD, ATR_SL_MULTIPLIER,
    VCP_MAX_FROM_52W_HIGH_PCT, VCP_MIN_ABOVE_52W_LOW_PCT, VCP_BASE_LENGTH,
    VCP_MIN_CONTRACTIONS, VCP_VOL_CONTRACTION_RATIO, VCP_BREAKOUT_VOL_MULT,
    VCP_PIVOT_PROXIMITY_PCT,
    BB_PERIOD, BB_STD_DEV, BB_SQUEEZE_LOOKBACK_DAILY, BB_SQUEEZE_LOOKBACK_WEEKLY,
    BB_SQUEEZE_PERCENTILE, BB_BREAKOUT_LOOKBACK, BB_BREAKOUT_VOL_MULT,
    BB_MIN_TREND_FILTER,
)


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = ATR_PERIOD) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = ADX_PERIOD) -> pd.Series:
    """Calculate Average Directional Index (ADX)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)

    # Zero out whichever is smaller
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    atr = calc_atr(high, low, close, period)

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def screen_ema200_breakout(df: pd.DataFrame, max_above_ema_pct: float = MAX_ABOVE_EMA_PCT,
                           lookback: int = CROSSOVER_LOOKBACK) -> dict | None:
    """EMA 200 breakout screen: price above 200 EMA with recent crossover detection.

    Catches stocks that have recently broken above 200 EMA (like NMDC-type moves).
    Less strict than swing strategy — no RSI/ADX range requirements.
    Returns dict with key metrics if stock qualifies, else None.
    """
    if df is None or len(df) < EMA_LONG + 10:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ema200 = calc_ema(close, EMA_LONG)
    ema50 = calc_ema(close, EMA_SHORT)
    rsi = calc_rsi(close, RSI_PERIOD)
    atr = calc_atr(high, low, close, ATR_PERIOD)
    avg_vol_20 = volume.rolling(VOLUME_AVG_PERIOD).mean()

    current_price = close.iloc[-1]
    current_ema200 = ema200.iloc[-1]
    current_ema50 = ema50.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_atr = atr.iloc[-1]

    # Must be above 200 EMA
    if current_price <= current_ema200:
        return None

    # Not too extended above 200 EMA
    pct_above_ema200 = ((current_price - current_ema200) / current_ema200) * 100
    if pct_above_ema200 > max_above_ema_pct:
        return None

    # Find the most recent 200 EMA crossover (scan full history, not just lookback)
    crossover_date = None
    crossover_vol_ratio = 0.0
    days_since_crossover = None

    for i in range(len(df) - 1, 0, -1):
        if close.iloc[i] > ema200.iloc[i] and close.iloc[i - 1] <= ema200.iloc[i - 1]:
            crossover_date = df.index[i]
            days_since_crossover = len(df) - 1 - i
            vol_on_day = volume.iloc[i]
            avg_vol = avg_vol_20.iloc[i]
            if not np.isnan(avg_vol) and avg_vol > 0:
                crossover_vol_ratio = round(vol_on_day / avg_vol, 2)
            break

    # Require crossover within lookback window
    if crossover_date is None or days_since_crossover is None:
        return None
    if days_since_crossover > lookback:
        return None

    # Average daily traded value
    recent_20 = df.tail(VOLUME_AVG_PERIOD)
    avg_traded_value = (recent_20["Close"] * recent_20["Volume"]).mean()

    # Stop-loss calculations
    atr_stop_loss = round(current_price - (ATR_SL_MULTIPLIER * current_atr), 2)
    swing_low = low.iloc[-10:].min()
    swing_sl = round(float(swing_low), 2)
    stop_loss = max(atr_stop_loss, swing_sl)
    sl_pct = round(((current_price - stop_loss) / current_price) * 100, 2)

    # 52-week high proximity
    high_52w = high.max()
    pct_from_52w_high = round(((high_52w - current_price) / high_52w) * 100, 2)

    return {
        "price": round(current_price, 2),
        "ema200": round(current_ema200, 2),
        "ema50": round(current_ema50, 2),
        "pct_above_ema200": round(pct_above_ema200, 2),
        "rsi": round(current_rsi, 2),
        "atr": round(current_atr, 2),
        "crossover_date": crossover_date.strftime("%Y-%m-%d") if hasattr(crossover_date, "strftime") else str(crossover_date),
        "crossover_vol_ratio": crossover_vol_ratio,
        "days_since_crossover": days_since_crossover,
        "avg_traded_value": avg_traded_value,
        "stop_loss": stop_loss,
        "sl_pct": sl_pct,
        "pct_from_52w_high": pct_from_52w_high,
    }


# ── VCP (Volatility Contraction Pattern) ────────────────────────────────────

def _find_pivot_highs(high: pd.Series, order: int = 5) -> list[tuple[int, float]]:
    """Find local pivot highs. Returns list of (index_position, price)."""
    pivots = []
    for i in range(order, len(high) - order):
        if all(high.iloc[i] >= high.iloc[i - j] for j in range(1, order + 1)) and \
           all(high.iloc[i] >= high.iloc[i + j] for j in range(1, order + 1)):
            pivots.append((i, float(high.iloc[i])))
    return pivots


def screen_vcp(df: pd.DataFrame,
               max_from_high_pct: float = VCP_MAX_FROM_52W_HIGH_PCT,
               min_above_low_pct: float = VCP_MIN_ABOVE_52W_LOW_PCT,
               base_length: int = VCP_BASE_LENGTH,
               min_contractions: int = VCP_MIN_CONTRACTIONS,
               vol_contraction_ratio: float = VCP_VOL_CONTRACTION_RATIO,
               breakout_vol_mult: float = VCP_BREAKOUT_VOL_MULT,
               pivot_proximity_pct: float = VCP_PIVOT_PROXIMITY_PCT) -> dict | None:
    """VCP (Volatility Contraction Pattern) screener - Mark Minervini style.

    Detects stocks consolidating near highs with tightening price contractions
    and declining volume, ready for or just beginning a breakout.

    Returns dict with key metrics if stock qualifies, else None.
    """
    if df is None or len(df) < EMA_LONG + base_length:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ema200 = calc_ema(close, EMA_LONG)
    ema150 = calc_ema(close, 150)
    ema50 = calc_ema(close, EMA_SHORT)
    rsi = calc_rsi(close, RSI_PERIOD)
    atr = calc_atr(high, low, close, ATR_PERIOD)

    cp = float(close.iloc[-1])
    ce200 = float(ema200.iloc[-1])
    ce150 = float(ema150.iloc[-1])
    ce50 = float(ema50.iloc[-1])
    cr = float(rsi.iloc[-1])
    ca = float(atr.iloc[-1])

    # ── Minervini Trend Template ────────────────────────────────────────
    # Price must be above 150 EMA and 200 EMA
    if cp <= ce200 or cp <= ce150:
        return None

    # 150 EMA must be above 200 EMA (stacked EMAs)
    if ce150 <= ce200:
        return None

    # 200 EMA must be rising (current > 1 month ago)
    if len(ema200) < 22 or float(ema200.iloc[-1]) <= float(ema200.iloc[-22]):
        return None

    # Price within max_from_high_pct of 52W high
    high_52w = float(high.max())
    pct_from_52w_high = round(((high_52w - cp) / high_52w) * 100, 2)
    if pct_from_52w_high > max_from_high_pct:
        return None

    # Price at least min_above_low_pct above 52W low
    low_52w = float(low.min())
    pct_above_52w_low = ((cp - low_52w) / low_52w) * 100
    if pct_above_52w_low < min_above_low_pct:
        return None

    # ── Volatility Contraction Detection ────────────────────────────────
    base = df.tail(base_length)
    base_high = base["High"]

    # Split base into segments and check tightening ranges
    seg_len = base_length // (min_contractions + 1)
    if seg_len < 10:
        seg_len = 10

    ranges = []
    for i in range(min_contractions + 1):
        start = i * seg_len
        end = min(start + seg_len, len(base))
        if end <= start:
            break
        seg = base.iloc[start:end]
        seg_range = (float(seg["High"].max()) - float(seg["Low"].min()))
        seg_pct = (seg_range / float(seg["Low"].min())) * 100
        ranges.append(seg_pct)

    if len(ranges) < min_contractions + 1:
        return None

    # Check that later segments are tighter
    contracting_count = sum(1 for i in range(len(ranges) - 1) if ranges[i] > ranges[i + 1])
    if contracting_count < min_contractions:
        return None

    contraction_ratio = round(ranges[-1] / ranges[0], 2) if ranges[0] > 0 else 1.0

    # ATR contraction confirmation
    atr_recent = float(atr.iloc[-10:].mean())
    atr_prior = float(atr.iloc[-base_length:-base_length // 2].mean())
    atr_contraction = round(atr_recent / atr_prior, 2) if atr_prior > 0 else 1.0

    # ── Volume Dry-up ───────────────────────────────────────────────────
    vol_recent_20 = float(volume.iloc[-20:].mean())
    vol_prior_50 = float(volume.iloc[-50:-20].mean()) if len(volume) >= 50 else float(volume.mean())
    vol_ratio = round(vol_recent_20 / vol_prior_50, 2) if vol_prior_50 > 0 else 1.0

    if vol_ratio > vol_contraction_ratio:
        return None

    # ── Pivot / Breakout Zone ───────────────────────────────────────────
    pivot_high = float(base_high.max())
    pct_below_pivot = ((pivot_high - cp) / pivot_high) * 100

    breakout_active = cp >= pivot_high
    near_pivot = pct_below_pivot <= pivot_proximity_pct

    if not breakout_active and not near_pivot:
        return None

    # Breakout volume (last 3 days max vol vs 50-day avg)
    avg_vol_50 = float(volume.iloc[-50:].mean()) if len(volume) >= 50 else float(volume.mean())
    recent_max_vol = float(volume.iloc[-3:].max())
    breakout_vol_ratio = round(recent_max_vol / avg_vol_50, 2) if avg_vol_50 > 0 else 0

    # Determine status
    if breakout_active and breakout_vol_ratio >= breakout_vol_mult:
        vcp_status = "BREAKING OUT"
    elif breakout_active:
        vcp_status = "ABOVE PIVOT"
    else:
        vcp_status = "NEAR PIVOT"

    # ── Standard metrics ────────────────────────────────────────────────
    recent_20_df = df.tail(VOLUME_AVG_PERIOD)
    avg_traded_value = float((recent_20_df["Close"] * recent_20_df["Volume"]).mean())

    # Stop loss: below the last contraction low or ATR-based
    last_seg_low = float(base.iloc[-seg_len:]["Low"].min())
    atr_sl = round(cp - ATR_SL_MULTIPLIER * ca, 2)
    stop_loss = max(atr_sl, round(last_seg_low, 2))
    sl_pct = round(((cp - stop_loss) / cp) * 100, 2)

    return {
        "price": round(cp, 2),
        "ema200": round(ce200, 2),
        "ema150": round(ce150, 2),
        "ema50": round(ce50, 2),
        "rsi": round(cr, 2),
        "atr": round(ca, 2),
        "pct_from_52w_high": pct_from_52w_high,
        "pct_above_52w_low": round(pct_above_52w_low, 2),
        "pivot_high": round(pivot_high, 2),
        "pct_below_pivot": round(pct_below_pivot, 2),
        "vcp_status": vcp_status,
        "contraction_ratio": contraction_ratio,
        "atr_contraction": atr_contraction,
        "vol_contraction": vol_ratio,
        "breakout_vol_ratio": breakout_vol_ratio,
        "base_depth_pct": round(ranges[0], 2),
        "avg_traded_value": avg_traded_value,
        "stop_loss": stop_loss,
        "sl_pct": sl_pct,
    }


def screen_technical(df: pd.DataFrame) -> dict | None:
    """Apply all technical screening conditions to a single stock's OHLCV DataFrame.

    Returns dict with screening results if stock passes all conditions, else None.
    """
    if df is None or len(df) < EMA_LONG + 10:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Calculate indicators
    ema200 = calc_ema(close, EMA_LONG)
    ema50 = calc_ema(close, EMA_SHORT)
    rsi = calc_rsi(close, RSI_PERIOD)
    avg_vol_20 = volume.rolling(VOLUME_AVG_PERIOD).mean()
    atr = calc_atr(high, low, close, ATR_PERIOD)
    adx = calc_adx(high, low, close, ADX_PERIOD)

    current_price = close.iloc[-1]
    current_ema200 = ema200.iloc[-1]
    current_ema50 = ema50.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_atr = atr.iloc[-1]
    current_adx = adx.iloc[-1]

    # --- Condition: Current price above 200 EMA ---
    if current_price <= current_ema200:
        return None

    # --- Condition: Price above 50 EMA ---
    if current_price <= current_ema50:
        return None

    # --- Condition: Not more than 8% above 200 EMA ---
    pct_above_ema200 = ((current_price - current_ema200) / current_ema200) * 100
    if pct_above_ema200 > MAX_ABOVE_EMA_PCT:
        return None

    # --- Condition: RSI between 50 and 70 ---
    if np.isnan(current_rsi) or current_rsi < RSI_LOWER or current_rsi > RSI_UPPER:
        return None

    # --- Condition: ADX >= threshold (trend strength) ---
    if np.isnan(current_adx) or current_adx < ADX_MIN:
        return None

    # --- Condition: 200 EMA crossover in last N sessions with volume spike ---
    crossover_found = False
    crossover_date = None
    crossover_vol_ratio = 0.0

    lookback_start = max(len(df) - CROSSOVER_LOOKBACK, 1)
    days_since_crossover = None
    for i in range(lookback_start, len(df)):
        if close.iloc[i] > ema200.iloc[i] and close.iloc[i - 1] <= ema200.iloc[i - 1]:
            vol_on_day = volume.iloc[i]
            avg_vol = avg_vol_20.iloc[i]
            if not np.isnan(avg_vol) and avg_vol > 0:
                ratio = vol_on_day / avg_vol
                if ratio >= VOLUME_MULTIPLIER:
                    crossover_found = True
                    crossover_date = df.index[i]
                    crossover_vol_ratio = round(ratio, 2)
                    days_since_crossover = len(df) - 1 - i
                    break

    if not crossover_found:
        return None

    # --- Average daily traded value ---
    recent_20 = df.tail(VOLUME_AVG_PERIOD)
    avg_traded_value = (recent_20["Close"] * recent_20["Volume"]).mean()

    # --- Stop-loss calculations ---
    # Method 1: ATR-based
    atr_stop_loss = round(current_price - (ATR_SL_MULTIPLIER * current_atr), 2)
    atr_sl_pct = round(((current_price - atr_stop_loss) / current_price) * 100, 2)

    # Method 2: Recent swing low (lowest low in last 10 sessions)
    swing_low = low.iloc[-10:].min()
    swing_sl = round(float(swing_low), 2)
    swing_sl_pct = round(((current_price - swing_sl) / current_price) * 100, 2)

    # Use the tighter (higher) stop loss
    stop_loss = max(atr_stop_loss, swing_sl)
    sl_pct = round(((current_price - stop_loss) / current_price) * 100, 2)

    # Proximity to 52-week high
    high_52w = high.max()
    pct_from_52w_high = round(((high_52w - current_price) / high_52w) * 100, 2)

    return {
        "price": round(current_price, 2),
        "ema200": round(current_ema200, 2),
        "ema50": round(current_ema50, 2),
        "pct_above_ema200": round(pct_above_ema200, 2),
        "rsi": round(current_rsi, 2),
        "adx": round(current_adx, 2),
        "atr": round(current_atr, 2),
        "crossover_date": crossover_date.strftime("%Y-%m-%d") if hasattr(crossover_date, "strftime") else str(crossover_date),
        "crossover_vol_ratio": crossover_vol_ratio,
        "days_since_crossover": days_since_crossover,
        "avg_traded_value": avg_traded_value,
        "stop_loss": stop_loss,
        "sl_pct": sl_pct,
        "pct_from_52w_high": pct_from_52w_high,
    }


# ── Bollinger Band Squeeze ──────────────────────────────────────────────────

def calc_bollinger_bands(close: pd.Series, period: int = BB_PERIOD,
                         std_dev: float = BB_STD_DEV):
    """Calculate Bollinger Bands. Returns (middle, upper, lower)."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly bars (week ending Friday)."""
    if df is None or df.empty:
        return df
    weekly = pd.DataFrame()
    weekly["Open"] = df["Open"].resample("W-FRI").first()
    weekly["High"] = df["High"].resample("W-FRI").max()
    weekly["Low"] = df["Low"].resample("W-FRI").min()
    weekly["Close"] = df["Close"].resample("W-FRI").last()
    weekly["Volume"] = df["Volume"].resample("W-FRI").sum()
    return weekly.dropna(subset=["Close"])


def screen_bb_squeeze(df: pd.DataFrame, timeframe: str = "daily",
                      bb_period: int = BB_PERIOD,
                      bb_std: float = BB_STD_DEV,
                      squeeze_lookback=None,
                      squeeze_percentile: float = BB_SQUEEZE_PERCENTILE,
                      breakout_lookback: int = BB_BREAKOUT_LOOKBACK,
                      vol_mult: float = BB_BREAKOUT_VOL_MULT,
                      require_trend: bool = BB_MIN_TREND_FILTER):
    """Bollinger Band Squeeze & Breakout screener (daily or weekly)."""
    if df is None or df.empty:
        return None

    # Resample to weekly if needed
    if timeframe == "weekly":
        work_df = _resample_weekly(df)
        if squeeze_lookback is None:
            squeeze_lookback = BB_SQUEEZE_LOOKBACK_WEEKLY
        trend_ema_period = 20
        min_bars_needed = max(bb_period, trend_ema_period, squeeze_lookback) + 5
    else:
        work_df = df
        if squeeze_lookback is None:
            squeeze_lookback = BB_SQUEEZE_LOOKBACK_DAILY
        trend_ema_period = EMA_LONG
        min_bars_needed = max(bb_period, trend_ema_period, squeeze_lookback) + 5

    if work_df is None or len(work_df) < min_bars_needed:
        return None

    close = work_df["Close"]
    high = work_df["High"]
    low = work_df["Low"]
    volume = work_df["Volume"]

    # Bollinger Bands
    bb_mid, bb_upper, bb_lower = calc_bollinger_bands(close, bb_period, bb_std)
    bb_width_pct = ((bb_upper - bb_lower) / bb_mid) * 100

    # Trend filter
    trend_ema = calc_ema(close, trend_ema_period)
    cp = float(close.iloc[-1])
    ce = float(trend_ema.iloc[-1])

    if require_trend and cp <= ce:
        return None

    # Latest BB values
    cur_mid = float(bb_mid.iloc[-1])
    cur_upper = float(bb_upper.iloc[-1])
    cur_lower = float(bb_lower.iloc[-1])
    cur_width = float(bb_width_pct.iloc[-1])

    if np.isnan(cur_width):
        return None

    # Squeeze detection
    lookback_widths = bb_width_pct.tail(squeeze_lookback).dropna()
    if len(lookback_widths) < squeeze_lookback // 2:
        return None

    pct_threshold = float(np.percentile(lookback_widths, squeeze_percentile))
    min_width = float(lookback_widths.min())
    max_width = float(lookback_widths.max())

    is_squeezed = cur_width <= pct_threshold

    if max_width > min_width:
        squeeze_intensity = round(
            (1 - (cur_width - min_width) / (max_width - min_width)) * 100, 1
        )
    else:
        squeeze_intensity = 100.0

    # Breakout detection - scan last N bars
    breakout_bar_idx = None
    breakout_vol_ratio = 0.0
    days_since_breakout = None

    avg_vol = volume.rolling(VOLUME_AVG_PERIOD).mean()
    scan_start = max(len(work_df) - 1 - breakout_lookback, 1)
    for i in range(len(work_df) - 1, scan_start, -1):
        if close.iloc[i] > bb_upper.iloc[i] and close.iloc[i - 1] <= bb_upper.iloc[i - 1]:
            breakout_bar_idx = i
            days_since_breakout = len(work_df) - 1 - i
            v = volume.iloc[i]
            av = avg_vol.iloc[i]
            if not np.isnan(av) and av > 0:
                breakout_vol_ratio = round(float(v / av), 2)
            break

    breakout_active = breakout_bar_idx is not None
    above_upper = cp > cur_upper

    # Status
    if breakout_active and breakout_vol_ratio >= vol_mult:
        status = "BREAKING OUT"
    elif breakout_active:
        status = "BREAKOUT (low vol)"
    elif above_upper:
        status = "ABOVE UPPER"
    elif is_squeezed:
        status = "IN SQUEEZE"
    else:
        return None

    # Only return IN SQUEEZE if intensity is high
    if status == "IN SQUEEZE" and squeeze_intensity < 70:
        return None

    # Standard metrics
    atr_series = calc_atr(high, low, close)
    cur_atr = float(atr_series.iloc[-1])

    rsi_series = calc_rsi(close)
    cur_rsi = float(rsi_series.iloc[-1])

    daily_recent = df.tail(VOLUME_AVG_PERIOD)
    avg_traded_value = float((daily_recent["Close"] * daily_recent["Volume"]).mean())

    # Stop loss: max of BB middle and ATR-based SL
    atr_sl = round(cp - ATR_SL_MULTIPLIER * cur_atr, 2)
    bb_mid_sl = round(cur_mid, 2)
    stop_loss = max(atr_sl, bb_mid_sl)
    sl_pct = round(((cp - stop_loss) / cp) * 100, 2)

    high_52w = float(df["High"].max())
    pct_from_52w_high = round(((high_52w - cp) / high_52w) * 100, 2)

    breakout_date_str = ""
    if breakout_bar_idx is not None:
        bd = work_df.index[breakout_bar_idx]
        breakout_date_str = bd.strftime("%Y-%m-%d") if hasattr(bd, "strftime") else str(bd)

    return {
        "price": round(cp, 2),
        "bb_upper": round(cur_upper, 2),
        "bb_middle": round(cur_mid, 2),
        "bb_lower": round(cur_lower, 2),
        "bb_width_pct": round(cur_width, 2),
        "squeeze_intensity": squeeze_intensity,
        "bb_status": status,
        "rsi": round(cur_rsi, 2),
        "atr": round(cur_atr, 2),
        "breakout_date": breakout_date_str,
        "days_since_breakout": days_since_breakout if days_since_breakout is not None else "",
        "breakout_vol_ratio": breakout_vol_ratio,
        "trend_ema": round(ce, 2),
        "avg_traded_value": avg_traded_value,
        "stop_loss": stop_loss,
        "sl_pct": sl_pct,
        "pct_from_52w_high": pct_from_52w_high,
        "timeframe": timeframe,
    }
