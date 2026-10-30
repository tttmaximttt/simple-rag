"""SQLite metadata: documents, ingestion jobs, query log, event log."""
import sqlite3
import threading
import time
from contextlib import contextmanager

from .config import DATA_DIR

_DB = DATA_DIR / "meta.sqlite"
_lock = threading.Lock()
KEEP_EVENTS = 2000  # per partition: auth events, and all others
KEEP_QUERIES = 5000

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
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT,
  must_change INTEGER DEFAULT 0, created REAL, last_login REAL
);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id INTEGER, created REAL, expires REAL, ip TEXT
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
        if "language" not in cols:  # set at ingest, for searching in the documents' languages
            c.execute("ALTER TABLE documents ADD COLUMN language TEXT")
        # Lets log() trim each partition with an index seek instead of scanning the table.
        c.execute("CREATE INDEX IF NOT EXISTS events_part ON events ((source IS 'auth'), id)")
        if "scope" not in {r["name"] for r in c.execute("PRAGMA table_info(api_keys)")}:
            # "read" (search and ask) or "write" (also upload and import). Keys from before scopes stay read-only.
            c.execute("ALTER TABLE api_keys ADD COLUMN scope TEXT NOT NULL DEFAULT 'read'")
        c.execute("CREATE INDEX IF NOT EXISTS queries_ts ON queries (ts)")
        # Keep the newest KEEP_QUERIES rows (ids only grow); recreated so a changed limit takes effect.
        c.execute("DROP TRIGGER IF EXISTS queries_cap")
        c.execute(f"CREATE TRIGGER queries_cap AFTER INSERT ON queries BEGIN "
                  f"DELETE FROM queries WHERE id <= NEW.id - {int(KEEP_QUERIES)}; END")


def rows(sql, args=()):
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def execute(sql, args=()) -> int:
    """Run one statement and return the number of rows it changed."""
    with conn() as c:
        return c.execute(sql, args).rowcount


def log(level: str, source: str, message: str):
    execute("INSERT INTO events (ts, level, source, message) VALUES (?,?,?,?)", (time.time(), level, source, message))
    # Auth events and everything else are capped separately, so neither can push the other out.
    part = f"(source IS 'auth') = {int(source == 'auth')}"  # must match the events_part index expression
    execute(f"DELETE FROM events WHERE {part} AND id <= "
            f"(SELECT id FROM events WHERE {part} ORDER BY id DESC LIMIT 1 OFFSET {KEEP_EVENTS})")
