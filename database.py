"""SQLite metadata-only history. Raw submitted email, sender and URL text are never persisted."""
from __future__ import annotations
import sqlite3
import re
from pathlib import Path
from datetime import datetime, timezone
from config import DATABASE_PATH, LABELS

def connect(path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p); conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db(path: str | Path = DATABASE_PATH) -> None:
    with connect(path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            category TEXT NOT NULL, score REAL NOT NULL,
            sender_domain TEXT NOT NULL, url_count INTEGER NOT NULL,
            signals TEXT NOT NULL, reviewed_category TEXT, review_note TEXT,
            reviewed_at TEXT
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at)")

def save_analysis(result: dict, path: str | Path = DATABASE_PATH) -> int:
    init_db(path)
    with connect(path) as con:
        cur = con.execute("INSERT INTO analyses(created_at,category,score,sender_domain,url_count,signals) VALUES(?,?,?,?,?,?)", (
            datetime.now(timezone.utc).isoformat(timespec="seconds"), result["category"], float(result["score"]),
            result.get("sender", {}).get("domain", "not provided")[:253], int(result.get("urls", {}).get("count", 0)),
            " | ".join(result.get("signals", []))[:2000]))
        return int(cur.lastrowid)

def get_history(path: str | Path = DATABASE_PATH) -> list[dict]:
    init_db(path)
    with connect(path) as con:
        return [dict(r) for r in con.execute("SELECT * FROM analyses ORDER BY id DESC")]

def save_review(analysis_id: int, category: str, note: str = "", path: str | Path = DATABASE_PATH) -> None:
    if category not in LABELS: raise ValueError(f"Review category must be one of {LABELS}")
    note = (note or "").replace("\x00", " ")
    note = re.sub(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+", "[URL redacted]", note)
    note = re.sub(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[email redacted]", note)
    note = re.sub(r"(?i)\b(password|passcode|otp|verification code|security code)\s*(?:is|:|=)\s*[^\s,;]+", r"\1 [value redacted]", note)
    init_db(path)
    with connect(path) as con:
        cur = con.execute("UPDATE analyses SET reviewed_category=?, review_note=?, reviewed_at=? WHERE id=?", (category, note[:1000], datetime.now(timezone.utc).isoformat(timespec="seconds"), int(analysis_id)))
        if not cur.rowcount: raise ValueError("Analysis record was not found")

def delete_history(path: str | Path = DATABASE_PATH) -> int:
    init_db(path)
    with connect(path) as con:
        cur = con.execute("DELETE FROM analyses")
        return int(cur.rowcount)
