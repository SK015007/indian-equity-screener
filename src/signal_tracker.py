"""Persistent signal tracker — records screener signals and tracks them until TP/SL hit.

Supports two storage backends:
- SQLite (default, local) — file at project root
- Postgres (cloud-persistent) — auto-detected from DATABASE_URL env var or
  st.secrets["DATABASE_URL"]. Use with Supabase, Neon, Railway, etc.
"""

import sqlite3
import os
import json
from datetime import datetime, date

import yfinance as yf
import pandas as pd

# Try to import streamlit for secrets access (optional)
try:
    import streamlit as st
    _has_streamlit = True
except ImportError:
    _has_streamlit = False

# Try to import psycopg2 (only needed for Postgres)
try:
    import psycopg2
    import psycopg2.extras
    _has_psycopg2 = True
except ImportError:
    _has_psycopg2 = False


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals.db")


def _get_database_url() -> str | None:
    """Get DATABASE_URL from Streamlit secrets or environment."""
    # Try Streamlit secrets first (preferred for cloud deployment)
    if _has_streamlit:
        try:
            url = st.secrets.get("DATABASE_URL")
            if url:
                return url
        except Exception:
            pass
    # Fall back to env var
    return os.environ.get("DATABASE_URL")


def _is_postgres() -> bool:
    """Check if we should use Postgres backend."""
    url = _get_database_url()
    return bool(url and _has_psycopg2)


def get_backend_info() -> dict:
    """Return info about the active backend (for UI display)."""
    if _is_postgres():
        return {"backend": "Postgres", "persistent": True,
                "info": "Signals persist forever in cloud database."}
    elif _get_database_url() and not _has_psycopg2:
        return {"backend": "SQLite (Postgres URL set but psycopg2 missing)",
                "persistent": False,
                "info": "Install psycopg2-binary to enable Postgres."}
    else:
        return {"backend": "SQLite (local file)", "persistent": False,
                "info": "Signals may be lost on Streamlit Cloud restarts. "
                        "Set DATABASE_URL in Streamlit secrets to use Postgres."}


# ── Postgres connection helpers ──────────────────────────────────────────────

def _pg_conn():
    url = _get_database_url()
    conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def _init_pg():
    """Create the signals table if it doesn't exist (Postgres)."""
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    sl_pct REAL NOT NULL,
                    target_1 REAL NOT NULL,
                    target_2 REAL NOT NULL,
                    rr_1 REAL NOT NULL DEFAULT 2.0,
                    rr_2 REAL NOT NULL DEFAULT 3.0,
                    signal_date TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    exit_price REAL,
                    exit_date TEXT,
                    exit_reason TEXT,
                    pnl_pct REAL,
                    high_since_entry REAL,
                    low_since_entry REAL,
                    last_checked TEXT,
                    extra_data TEXT,
                    UNIQUE(symbol, strategy, signal_date)
                )
            """)
        conn.commit()
    finally:
        conn.close()


def _init_sqlite():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            sl_pct REAL NOT NULL,
            target_1 REAL NOT NULL,
            target_2 REAL NOT NULL,
            rr_1 REAL NOT NULL DEFAULT 2.0,
            rr_2 REAL NOT NULL DEFAULT 3.0,
            signal_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            exit_price REAL,
            exit_date TEXT,
            exit_reason TEXT,
            pnl_pct REAL,
            high_since_entry REAL,
            low_since_entry REAL,
            last_checked TEXT,
            extra_data TEXT,
            UNIQUE(symbol, strategy, signal_date)
        )
    """)
    conn.commit()
    return conn


def _get_conn():
    """Get a database connection for the active backend."""
    if _is_postgres():
        _init_pg()
        return _pg_conn()
    else:
        return _init_sqlite()


def _placeholder() -> str:
    return "%s" if _is_postgres() else "?"


def _execute(conn, sql: str, params: tuple = ()):
    """Execute SQL on either backend with appropriate placeholder style."""
    ph = _placeholder()
    # Convert ? to %s for Postgres
    if ph == "%s":
        sql = sql.replace("?", "%s")
    if _is_postgres():
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur
    else:
        return conn.execute(sql, params)


def _fetchone(conn, sql: str, params: tuple = ()):
    cur = _execute(conn, sql, params)
    row = cur.fetchone()
    if _is_postgres():
        cur.close()
    return row


def _fetchall(conn, sql: str, params: tuple = ()):
    cur = _execute(conn, sql, params)
    rows = cur.fetchall()
    if _is_postgres():
        cur.close()
    return rows


def _read_sql_to_df(sql: str) -> pd.DataFrame:
    """Read a query result into a pandas DataFrame."""
    conn = _get_conn()
    try:
        if _is_postgres():
            return pd.read_sql_query(sql, conn)
        else:
            return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


# ── Public API ───────────────────────────────────────────────────────────────

def record_signal(symbol: str, strategy: str, entry_price: float,
                  stop_loss: float, sl_pct: float,
                  rr_1: float = 2.0, rr_2: float = 3.0,
                  extra_data: str = "") -> bool:
    """Record a new signal. Returns True if new, False if duplicate."""
    risk = entry_price - stop_loss
    target_1 = round(entry_price + risk * rr_1, 2)
    target_2 = round(entry_price + risk * rr_2, 2)
    signal_date = date.today().isoformat()

    conn = _get_conn()
    try:
        existing = _fetchone(conn,
            "SELECT id, status FROM signals WHERE symbol=? AND strategy=? AND status='ACTIVE'",
            (symbol, strategy)
        )
        if existing:
            return False

        _execute(conn, """
            INSERT INTO signals (symbol, strategy, entry_price, stop_loss, sl_pct,
                                 target_1, target_2, rr_1, rr_2, signal_date,
                                 status, high_since_entry, low_since_entry,
                                 last_checked, extra_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?)
        """, (symbol, strategy, entry_price, stop_loss, sl_pct,
              target_1, target_2, rr_1, rr_2, signal_date,
              entry_price, entry_price, signal_date, extra_data))
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        conn.close()


def check_and_update_signals() -> dict:
    """Check all ACTIVE signals against current prices. Update TP/SL hits."""
    conn = _get_conn()
    try:
        active = _fetchall(conn, "SELECT * FROM signals WHERE status='ACTIVE'")
    except Exception:
        conn.close()
        return {"updated": 0, "tp1_hits": [], "tp2_hits": [], "sl_hits": []}

    if not active:
        conn.close()
        return {"updated": 0, "tp1_hits": [], "tp2_hits": [], "sl_hits": []}

    # Convert rows to dicts for uniform access
    active = [dict(row) for row in active]
    symbols = [row["symbol"] for row in active]
    nse_symbols = [f"{s}.NS" for s in symbols]

    try:
        data = yf.download(nse_symbols, period="5d", progress=False, group_by="ticker")
    except Exception:
        conn.close()
        return {"updated": 0, "tp1_hits": [], "tp2_hits": [], "sl_hits": []}

    today = date.today().isoformat()
    tp1_hits, tp2_hits, sl_hits = [], [], []
    updated = 0

    for row in active:
        sym = row["symbol"]
        nse_sym = f"{sym}.NS"

        try:
            if len(symbols) == 1:
                ticker_data = data
            else:
                ticker_data = data[nse_sym] if nse_sym in data.columns.get_level_values(0) else None

            if ticker_data is None or ticker_data.empty:
                continue

            recent_high = float(ticker_data["High"].max())
            recent_low = float(ticker_data["Low"].min())
            current_price = float(ticker_data["Close"].iloc[-1])

            track_high = max(row["high_since_entry"] or row["entry_price"], recent_high)
            track_low = min(row["low_since_entry"] or row["entry_price"], recent_low)

            if recent_low <= row["stop_loss"]:
                pnl = round(((row["stop_loss"] - row["entry_price"]) / row["entry_price"]) * 100, 2)
                _execute(conn, """
                    UPDATE signals SET status='SL HIT', exit_price=?, exit_date=?,
                    exit_reason='Stop loss hit', pnl_pct=?,
                    high_since_entry=?, low_since_entry=?, last_checked=?
                    WHERE id=?
                """, (row["stop_loss"], today, pnl, track_high, track_low, today, row["id"]))
                sl_hits.append(sym)
                updated += 1
                continue

            if recent_high >= row["target_2"]:
                pnl = round(((row["target_2"] - row["entry_price"]) / row["entry_price"]) * 100, 2)
                _execute(conn, """
                    UPDATE signals SET status='TP2 HIT', exit_price=?, exit_date=?,
                    exit_reason='Target 2 hit', pnl_pct=?,
                    high_since_entry=?, low_since_entry=?, last_checked=?
                    WHERE id=?
                """, (row["target_2"], today, pnl, track_high, track_low, today, row["id"]))
                tp2_hits.append(sym)
                updated += 1
                continue

            if recent_high >= row["target_1"]:
                pnl = round(((row["target_1"] - row["entry_price"]) / row["entry_price"]) * 100, 2)
                _execute(conn, """
                    UPDATE signals SET status='TP1 HIT', exit_price=?, exit_date=?,
                    exit_reason='Target 1 hit', pnl_pct=?,
                    high_since_entry=?, low_since_entry=?, last_checked=?
                    WHERE id=?
                """, (row["target_1"], today, pnl, track_high, track_low, today, row["id"]))
                tp1_hits.append(sym)
                updated += 1
                continue

            curr_pnl = round(((current_price - row["entry_price"]) / row["entry_price"]) * 100, 2)
            _execute(conn, """
                UPDATE signals SET high_since_entry=?, low_since_entry=?,
                last_checked=?, pnl_pct=?
                WHERE id=?
            """, (track_high, track_low, today, curr_pnl, row["id"]))
            updated += 1

        except Exception:
            continue

    conn.commit()
    conn.close()
    return {"updated": updated, "tp1_hits": tp1_hits, "tp2_hits": tp2_hits, "sl_hits": sl_hits}


def get_active_signals() -> pd.DataFrame:
    """Get all ACTIVE signals as a DataFrame."""
    return _read_sql_to_df(
        "SELECT * FROM signals WHERE status='ACTIVE' ORDER BY signal_date DESC"
    )


def get_closed_signals() -> pd.DataFrame:
    return _read_sql_to_df(
        "SELECT * FROM signals WHERE status != 'ACTIVE' ORDER BY exit_date DESC"
    )


def get_all_signals() -> pd.DataFrame:
    return _read_sql_to_df("SELECT * FROM signals ORDER BY signal_date DESC")


def get_performance_stats() -> dict:
    """Calculate overall performance statistics."""
    conn = _get_conn()
    try:
        rows = _fetchall(conn, "SELECT * FROM signals WHERE status != 'ACTIVE'")
        rows = [dict(r) for r in rows]
        active_row = _fetchone(conn, "SELECT COUNT(*) AS c FROM signals WHERE status='ACTIVE'")
        active_count = (dict(active_row) if not isinstance(active_row, dict) else active_row).get("c", 0) \
                       if active_row else 0
    finally:
        conn.close()

    if not rows:
        return {
            "total_closed": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_win_pct": 0, "avg_loss_pct": 0, "avg_pnl_pct": 0,
            "best_trade": 0, "worst_trade": 0, "total_active": active_count,
        }

    wins = [r for r in rows if r.get("pnl_pct") and r["pnl_pct"] > 0]
    losses = [r for r in rows if r.get("pnl_pct") is not None and r["pnl_pct"] <= 0]
    all_pnl = [r["pnl_pct"] for r in rows if r.get("pnl_pct") is not None]

    return {
        "total_closed": len(rows),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(rows) * 100, 1) if rows else 0,
        "avg_win_pct": round(sum(r["pnl_pct"] for r in wins) / len(wins), 2) if wins else 0,
        "avg_loss_pct": round(sum(r["pnl_pct"] for r in losses) / len(losses), 2) if losses else 0,
        "avg_pnl_pct": round(sum(all_pnl) / len(all_pnl), 2) if all_pnl else 0,
        "best_trade": max(all_pnl) if all_pnl else 0,
        "worst_trade": min(all_pnl) if all_pnl else 0,
        "total_active": active_count,
    }


def fetch_ohlc_history(symbols: list, days: int = 90) -> dict:
    if not symbols:
        return {}

    days = max(days, 5)
    nse_symbols = [f"{s}.NS" for s in symbols]
    result = {}

    try:
        data = yf.download(
            " ".join(nse_symbols), period=f"{days}d", progress=False,
            group_by="ticker", threads=True,
        )

        for sym, nse_sym in zip(symbols, nse_symbols):
            try:
                if len(symbols) == 1:
                    df = data
                else:
                    if nse_sym not in data.columns.get_level_values(0):
                        continue
                    df = data[nse_sym]

                if df is None or df.empty:
                    continue
                df = df.dropna(subset=["Close"])
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                if len(df) > 0:
                    result[sym] = df
            except Exception:
                continue
    except Exception:
        pass

    return result


def fetch_live_prices(symbols: list) -> dict:
    ohlc = fetch_ohlc_history(symbols, days=2)
    return {sym: round(float(df["Close"].iloc[-1]), 2)
            for sym, df in ohlc.items() if not df.empty}


def _fetch_intraday(symbol: str):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period="1d", interval="5m")
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        return df if len(df) > 0 else None
    except Exception:
        return None


def _compute_high_low_close(ohlc, signal_dt: date, entry_price: float,
                             symbol: str, today: date):
    cmp_val = entry_price
    high_val = entry_price
    low_val = entry_price

    if ohlc is None or ohlc.empty:
        return cmp_val, high_val, low_val

    try:
        idx_dates = pd.to_datetime(ohlc.index).date
        mask = idx_dates >= signal_dt
        since_signal = ohlc[mask]
    except Exception:
        since_signal = ohlc.tail((today - signal_dt).days + 1)

    if since_signal.empty or len(since_signal) == 0:
        intraday = _fetch_intraday(symbol)
        if intraday is not None and not intraday.empty:
            since_signal = intraday
        else:
            since_signal = ohlc.tail(1)

    try:
        close_series = since_signal["Close"].dropna()
        if len(close_series) > 0:
            cmp_val = round(float(close_series.iloc[-1]), 2)
    except Exception:
        pass

    try:
        high_series = since_signal["High"].dropna()
        if len(high_series) > 0:
            actual_high = float(high_series.max())
            high_val = round(max(actual_high, entry_price, cmp_val), 2)
    except Exception:
        pass

    try:
        low_series = since_signal["Low"].dropna()
        if len(low_series) > 0:
            actual_low = float(low_series.min())
            low_val = round(min(actual_low, entry_price, cmp_val), 2)
    except Exception:
        pass

    return cmp_val, high_val, low_val


def get_active_signals_with_live_prices() -> pd.DataFrame:
    df = get_active_signals()
    if df.empty:
        return df

    today = date.today()
    symbols = df["symbol"].unique().tolist()

    df["_signal_dt"] = pd.to_datetime(df["signal_date"]).dt.date
    max_age_days = max((today - sd).days for sd in df["_signal_dt"]) + 5
    max_age_days = max(max_age_days, 5)

    ohlc_data = fetch_ohlc_history(symbols, days=max_age_days)

    cmps, highs, lows = {}, {}, {}
    for _, row in df.iterrows():
        sym = row["symbol"]
        signal_dt = row["_signal_dt"]
        entry = float(row["entry_price"])
        ohlc = ohlc_data.get(sym)
        cmp_val, high_val, low_val = _compute_high_low_close(
            ohlc, signal_dt, entry, sym, today
        )
        cmps[sym] = cmp_val
        highs[sym] = high_val
        lows[sym] = low_val

    df["cmp"] = df["symbol"].map(cmps).fillna(df["entry_price"])
    df["high_since_entry"] = df["symbol"].map(highs).fillna(df["entry_price"])
    df["low_since_entry"] = df["symbol"].map(lows).fillna(df["entry_price"])

    df["live_pnl_pct"] = round(((df["cmp"] - df["entry_price"]) / df["entry_price"]) * 100, 2)
    df["mfe_pct"] = round(((df["high_since_entry"] - df["entry_price"]) / df["entry_price"]) * 100, 2)
    df["mae_pct"] = round(((df["low_since_entry"] - df["entry_price"]) / df["entry_price"]) * 100, 2)
    df["dist_to_sl_pct"] = round(((df["cmp"] - df["stop_loss"]) / df["cmp"]) * 100, 2)
    df["dist_to_tp1_pct"] = round(((df["target_1"] - df["cmp"]) / df["cmp"]) * 100, 2)

    df = df.drop(columns=["_signal_dt"])
    return df


def manually_close_signal(signal_id: int, reason: str = "Manual close"):
    conn = _get_conn()
    try:
        row = _fetchone(conn, "SELECT * FROM signals WHERE id=?", (signal_id,))
        if row and dict(row).get("status") == "ACTIVE":
            row = dict(row)
            try:
                ticker = yf.Ticker(f"{row['symbol']}.NS")
                current_price = ticker.info.get("currentPrice") or ticker.info.get("regularMarketPrice", 0)
            except Exception:
                current_price = row["entry_price"]

            pnl = round(((current_price - row["entry_price"]) / row["entry_price"]) * 100, 2)
            _execute(conn, """
                UPDATE signals SET status='CLOSED', exit_price=?, exit_date=?,
                exit_reason=?, pnl_pct=?, last_checked=?
                WHERE id=?
            """, (current_price, date.today().isoformat(), reason, pnl,
                  date.today().isoformat(), signal_id))
            conn.commit()
    finally:
        conn.close()


# ── Backup / Restore ─────────────────────────────────────────────────────────

def export_signals_json() -> str:
    """Export all signals as JSON string."""
    df = get_all_signals()
    if df.empty:
        return "[]"
    return df.to_json(orient="records", date_format="iso")


def import_signals_json(json_str: str) -> int:
    """Import signals from JSON string. Skips duplicates. Returns count imported."""
    try:
        records = json.loads(json_str)
    except Exception:
        return 0

    if not isinstance(records, list):
        return 0

    conn = _get_conn()
    imported = 0
    try:
        for r in records:
            try:
                # Check for duplicate by symbol+strategy+signal_date
                existing = _fetchone(conn,
                    "SELECT id FROM signals WHERE symbol=? AND strategy=? AND signal_date=?",
                    (r.get("symbol"), r.get("strategy"), r.get("signal_date"))
                )
                if existing:
                    continue

                _execute(conn, """
                    INSERT INTO signals (
                        symbol, strategy, entry_price, stop_loss, sl_pct,
                        target_1, target_2, rr_1, rr_2, signal_date, status,
                        exit_price, exit_date, exit_reason, pnl_pct,
                        high_since_entry, low_since_entry, last_checked, extra_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r.get("symbol"), r.get("strategy"), r.get("entry_price"),
                    r.get("stop_loss"), r.get("sl_pct"), r.get("target_1"),
                    r.get("target_2"), r.get("rr_1", 2.0), r.get("rr_2", 3.0),
                    r.get("signal_date"), r.get("status", "ACTIVE"),
                    r.get("exit_price"), r.get("exit_date"), r.get("exit_reason"),
                    r.get("pnl_pct"), r.get("high_since_entry"),
                    r.get("low_since_entry"), r.get("last_checked"),
                    r.get("extra_data", "")
                ))
                imported += 1
            except Exception:
                continue
        conn.commit()
    finally:
        conn.close()
    return imported
