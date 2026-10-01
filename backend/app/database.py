import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone

from .config import DB_PATH

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id          TEXT PRIMARY KEY,
    repo_id     TEXT NOT NULL,
    task        TEXT,
    source      TEXT NOT NULL DEFAULT 'hub',
    base_model  TEXT,
    local_path  TEXT NOT NULL,
    status      TEXT NOT NULL,
    progress    REAL NOT NULL DEFAULT 0,
    total_bytes INTEGER NOT NULL DEFAULT 0,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    model_id      TEXT,
    system_prompt TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    model_id        TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);

CREATE TABLE IF NOT EXISTS finetune_jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    base_model   TEXT NOT NULL,
    output_model TEXT NOT NULL,
    task         TEXT NOT NULL,
    dataset      TEXT NOT NULL,
    params       TEXT NOT NULL,
    status       TEXT NOT NULL,
    progress     REAL NOT NULL DEFAULT 0,
    epoch        REAL,
    loss         REAL,
    logs         TEXT NOT NULL DEFAULT '[]',
    error        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _lock, closing(_connect()) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        # Anything left in-flight from a previous run can't be resumed.
        conn.execute(
            "UPDATE models SET status='error', error='Tải xuống bị gián đoạn' WHERE status='downloading'"
        )
        conn.execute(
            "UPDATE finetune_jobs SET status='failed', error='Server khởi động lại' "
            "WHERE status IN ('queued','running')"
        )
        conn.commit()


def query(sql: str, params: tuple = ()) -> list[dict]:
    with _lock, closing(_connect()) as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> int:
    with _lock, closing(_connect()) as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid


def get_setting(key: str) -> str | None:
    row = query_one("SELECT value FROM settings WHERE key=?", (key,))
    return row["value"] if row else None


def set_setting(key: str, value: str | None) -> None:
    if value is None:
        execute("DELETE FROM settings WHERE key=?", (key,))
    else:
        execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
