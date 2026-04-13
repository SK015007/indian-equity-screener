"""Persistent signal tracker — records screener signals and tracks them until TP/SL hit."""

import sqlite3
import os
from datetime import datetime, date

import yfinance as yf
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals.db")


def _get_conn() -> sqlite3.Connection:
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
        # Check if this signal already exists (any status)
        existing = conn.execute(
            "SELECT id, status FROM signals WHERE symbol=? AND strategy=? AND status='ACTIVE'",
            (symbol, strategy)
        ).fetchone()
        if existing:
            return False  # Already tracking this signal

        conn.execute("""
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
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def check_and_update_signals() -> dict:
    """Check all ACTIVE signals against current prices. Update TP/SL hits.

    Returns summary: {updated: int, tp1_hits: list, tp2_hits: list, sl_hits: list}
    """
    conn = _get_conn()
    active = conn.execute(
        "SELECT * FROM signals WHERE status='ACTIVE'"
    ).fetchall()

    if not active:
        conn.close()
        return {"updated": 0, "tp1_hits": [], "tp2_hits": [], "sl_hits": []}

    # Batch fetch current prices
    symbols = [row["symbol"] for row in active]
    nse_symbols = [f"{s}.NS" for s in symbols]

    try:
        data = yf.download(nse_symbols, period="5d", progress=False, group_by="ticker")
    except Exception:
        conn.close()
        return {"updated": 0, "tp1_hits": [], "tp2_hits": [], "sl_hits": []}

    today = date.today().isoformat()
    tp1_hits = []
    tp2_hits = []
    sl_hits = []
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

            # Get the high and low since we last checked
            recent_high = float(ticker_data["High"].max())
            recent_low = float(ticker_data["Low"].min())
            current_price = float(ticker_data["Close"].iloc[-1])

            # Update tracking highs/lows
            track_high = max(row["high_since_entry"] or row["entry_price"], recent_high)
            track_low = min(row["low_since_entry"] or row["entry_price"], recent_low)

            # Check SL hit (use low to detect intraday SL breach)
            if recent_low <= row["stop_loss"]:
                pnl = round(((row["stop_loss"] - row["entry_price"]) / row["entry_price"]) * 100, 2)
                conn.execute("""
                    UPDATE signals SET status='SL HIT', exit_price=?, exit_date=?,
                    exit_reason='Stop loss hit', pnl_pct=?,
                    high_since_entry=?, low_since_entry=?, last_checked=?
                    WHERE id=?
                """, (row["stop_loss"], today, pnl, track_high, track_low, today, row["id"]))
                sl_hits.append(sym)
                updated += 1
                continue

            # Check TP2 hit first (higher target)
            if recent_high >= row["target_2"]:
                pnl = round(((row["target_2"] - row["entry_price"]) / row["entry_price"]) * 100, 2)
                conn.execute("""
                    UPDATE signals SET status='TP2 HIT', exit_price=?, exit_date=?,
                    exit_reason='Target 2 hit', pnl_pct=?,
                    high_since_entry=?, low_since_entry=?, last_checked=?
                    WHERE id=?
                """, (row["target_2"], today, pnl, track_high, track_low, today, row["id"]))
                tp2_hits.append(sym)
                updated += 1
                continue

            # Check TP1 hit
            if recent_high >= row["target_1"]:
                pnl = round(((row["target_1"] - row["entry_price"]) / row["entry_price"]) * 100, 2)
                conn.execute("""
                    UPDATE signals SET status='TP1 HIT', exit_price=?, exit_date=?,
                    exit_reason='Target 1 hit', pnl_pct=?,
                    high_since_entry=?, low_since_entry=?, last_checked=?
                    WHERE id=?
                """, (row["target_1"], today, pnl, track_high, track_low, today, row["id"]))
                tp1_hits.append(sym)
                updated += 1
                continue

            # Still active — update tracking data
            curr_pnl = round(((current_price - row["entry_price"]) / row["entry_price"]) * 100, 2)
            conn.execute("""
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
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM signals WHERE status='ACTIVE' ORDER BY signal_date DESC", conn
    )
    conn.close()
    return df


def get_closed_signals() -> pd.DataFrame:
    """Get all closed signals (TP/SL hit) as a DataFrame."""
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM signals WHERE status != 'ACTIVE' ORDER BY exit_date DESC", conn
    )
    conn.close()
    return df


def get_all_signals() -> pd.DataFrame:
    """Get all signals as a DataFrame."""
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM signals ORDER BY signal_date DESC", conn
    )
    conn.close()
    return df


def get_performance_stats() -> dict:
    """Calculate overall performance statistics."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM signals WHERE status != 'ACTIVE'").fetchall()
    conn.close()

    if not rows:
        return {
            "total_closed": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_win_pct": 0, "avg_loss_pct": 0, "avg_pnl_pct": 0,
            "best_trade": 0, "worst_trade": 0, "total_active": 0,
        }

    wins = [r for r in rows if r["pnl_pct"] and r["pnl_pct"] > 0]
    losses = [r for r in rows if r["pnl_pct"] and r["pnl_pct"] <= 0]

    active_count = _get_conn().execute(
        "SELECT COUNT(*) FROM signals WHERE status='ACTIVE'"
    ).fetchone()[0]

    all_pnl = [r["pnl_pct"] for r in rows if r["pnl_pct"] is not None]

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


def manually_close_signal(signal_id: int, reason: str = "Manual close"):
    """Manually close an active signal."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM signals WHERE id=?", (signal_id,)).fetchone()
    if row and row["status"] == "ACTIVE":
        # Fetch current price
        try:
            ticker = yf.Ticker(f"{row['symbol']}.NS")
            current_price = ticker.info.get("currentPrice") or ticker.info.get("regularMarketPrice", 0)
        except Exception:
            current_price = row["entry_price"]

        pnl = round(((current_price - row["entry_price"]) / row["entry_price"]) * 100, 2)
        conn.execute("""
            UPDATE signals SET status='CLOSED', exit_price=?, exit_date=?,
            exit_reason=?, pnl_pct=?, last_checked=?
            WHERE id=?
        """, (current_price, date.today().isoformat(), reason, pnl,
              date.today().isoformat(), signal_id))
        conn.commit()
    conn.close()
