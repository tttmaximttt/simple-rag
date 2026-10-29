"""SQLite metadata: documents, ingestion jobs, query log, event log."""
import sqlite3
import threading
import time
from contextlib import contextmanager

from .config import DATA_DIR

_DB = DATA_DIR / "meta.sqlite"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, filename TEXT, size INTEGER, pages INTEGER, chunks INTEGER,
  status TEXT, error TEXT, created REAL, updated REAL, duration REAL
);
CREATE TABLE IF NOT EXISTS queries (
  id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, ts REAL,
  retrieve_ms REAL, generate_ms REAL, total_ms REAL, ok INTEGER, error TEXT
);
CREATE TABLE IF NOT EXISTS api_keys (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, prefix TEXT, hash TEXT UNIQUE,
  created REAL, last_used REAL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, level TEXT, source TEXT, message TEXT
);
"""


@contextmanager
def conn():
    with _lock:
        c = sqlite3.connect(_DB)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(documents)")}
        if "sha256" not in cols:  # added after the first release
            c.execute("ALTER TABLE documents ADD COLUMN sha256 TEXT")
        c.execute("CREATE INDEX IF NOT EXISTS documents_sha ON documents (sha256)")


def rows(sql, args=()):
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def execute(sql, args=()):
    with conn() as c:
        c.execute(sql, args)


def log(level: str, source: str, message: str):
    execute("INSERT INTO events (ts, level, source, message) VALUES (?,?,?,?)", (time.time(), level, source, message))
    execute("DELETE FROM events WHERE id < (SELECT MAX(id) - 2000 FROM events)")
