import hashlib
import re

import psycopg
from pgvector.psycopg import register_vector

from . import dimension_mismatch


MAX_IDENT = 63  # Postgres truncates identifiers at 63 bytes
INDEX_SUFFIX = "_hnsw"
HNSW_VECTOR_MAX = 2000  # vector HNSW limit; larger dims use halfvec (limit 4000)
TSV_SUFFIX = "_tsv"  # shorter than INDEX_SUFFIX, so it fits wherever that does

# Postgres text-search configurations for the languages pdftext.detect_language reports. "simple" only
# lowercases, for a document whose language wasn't detected.
TS_CONFIGS = {"English": "english", "Spanish": "spanish", "French": "french", "German": "german",
              "Italian": "italian", "Portuguese": "portuguese", "Dutch": "dutch"}


def ts_config(language):
    """The text-search configuration for a chunk of a document in `language` (None when not detected)."""
    return TS_CONFIGS.get(language or "", "simple")


def table_name(collection):
    """Short names keep their historical table name; long ones get a hash so
    the `_hnsw` index name fits and distinct collections never collide."""
    name = "rag_" + re.sub(r"[^a-z0-9_]", "_", collection.lower())
    if len((name + INDEX_SUFFIX).encode()) <= MAX_IDENT:
        return name
    digest = hashlib.sha1(collection.encode()).hexdigest()[:10]
    return name[:MAX_IDENT - len(INDEX_SUFFIX) - len(digest) - 1] + "_" + digest


class PgVectorStore:
    def __init__(self, dsn, collection):
        self.dsn = dsn
        self.collection = collection
        self.table = table_name(collection)

    def _conn(self):
        c = psycopg.connect(self.dsn, autocommit=True, connect_timeout=5)
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(c)
        return c

    def ping(self):
        with self._conn() as c:
            c.execute("SELECT 1")

    def _exists(self, c):
        return c.execute("SELECT to_regclass(%s)", (self.table,)).fetchone()[0] is not None

    def _dim(self, c):
        row = c.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = %s::regclass AND attname = 'embedding'", (self.table,)).fetchone()
        m = re.search(r"\((\d+)\)", row[0]) if row else None
        return int(m.group(1)) if m else None

    def ensure(self, dim):
        dim = int(dim)
        with self._conn() as c:
            if self._exists(c):
                have = self._dim(c)
                if have is not None and have != dim:
                    n = c.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
                    if n:
                        raise dimension_mismatch(self.collection, n, have, dim)
                    c.execute(f"DROP TABLE {self.table}")
            half = dim > HNSW_VECTOR_MAX
            col, ops = (f"halfvec({dim})", "halfvec_cosine_ops") if half else (f"vector({dim})", "vector_cosine_ops")
            c.execute(f"""CREATE TABLE IF NOT EXISTS {self.table} (
                id TEXT PRIMARY KEY, doc_id TEXT, filename TEXT, page INT, chunk INT,
                text TEXT, embedding {col}, ts_config TEXT, tsv tsvector)""")
            self._migrate(c)
            c.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_doc ON {self.table} (doc_id)")
            c.execute(f"CREATE INDEX IF NOT EXISTS {self.table}{INDEX_SUFFIX} ON {self.table} "
                      f"USING hnsw (embedding {ops})")

    def _has_keywords(self, c):
        return c.execute("SELECT 1 FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'tsv' "
                         "AND NOT attisdropped", (self.table,)).fetchone() is not None

    def _migrate(self, c):
        """Add the keyword columns and index to a table made before hybrid search. Its rows have no keyword
        data (tsv is NULL) until they're re-indexed; keyword_missing() counts them."""
        if not self._has_keywords(c):
            c.execute(f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS ts_config TEXT, "
                      "ADD COLUMN IF NOT EXISTS tsv tsvector")
        c.execute(f"CREATE INDEX IF NOT EXISTS {self.table}{TSV_SUFFIX} ON {self.table} USING gin (tsv)")

    def upsert(self, items):
        import numpy as np
        with self._conn() as c, c.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table} (id, doc_id, filename, page, chunk, text, embedding, ts_config, tsv)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s, to_tsvector(%s::regconfig, %s))
                    ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding,
                        ts_config = EXCLUDED.ts_config, tsv = EXCLUDED.tsv""",
                [(i, p["doc_id"], p["filename"], p["page"], p["chunk"], p["text"], np.array(v, dtype=np.float32),
                  cfg, cfg, p["text"]) for i, v, p in items for cfg in [ts_config(p.get("language"))]])

    def search(self, vector, k):
        import numpy as np
        v = np.array(vector, dtype=np.float32)
        cast = "::halfvec" if len(v) > HNSW_VECTOR_MAX else ""
        with self._conn() as c:
            if not self._exists(c):
                return []
            with c.transaction():
                c.execute(f"SET LOCAL hnsw.ef_search = {max(40, int(k))}")
                rows = c.execute(
                    f"""SELECT doc_id, filename, page, chunk, text, 1 - (embedding <=> %s{cast}) AS score
                        FROM {self.table} ORDER BY embedding <=> %s{cast} LIMIT %s""",
                    (v, v, k)).fetchall()
        keys = ("doc_id", "filename", "page", "chunk", "text", "score")
        return [dict(zip(keys, r)) for r in rows]

    def keyword_search(self, query, k):
        """Full-text search: the query is parsed with each text-search configuration and matched against the
        chunks stored with that configuration, any of its words counting (OR), ranked by ts_rank_cd."""
        with self._conn() as c:
            if not self._exists(c):
                return []
            if not self._has_keywords(c):
                self._migrate(c)
            rows = c.execute(
                f"""WITH q AS (
                        SELECT cfg, replace(plainto_tsquery(cfg::regconfig, %s)::text, ' & ', ' | ') AS qt
                        FROM unnest(%s::text[]) AS cfg)
                    SELECT t.doc_id, t.filename, t.page, t.chunk, t.text, ts_rank_cd(t.tsv, q.qt::tsquery) AS score
                    FROM {self.table} t JOIN q ON t.ts_config = q.cfg
                    WHERE q.qt <> '' AND t.tsv @@ q.qt::tsquery
                    ORDER BY score DESC, t.id LIMIT %s""", (query, [*TS_CONFIGS.values(), "simple"], k)).fetchall()
        keys = ("doc_id", "filename", "page", "chunk", "text", "score")
        return [dict(zip(keys, r)) for r in rows]

    def keyword_missing(self):
        """Chunks stored before hybrid search, which keyword search can't find until they're re-indexed."""
        with self._conn() as c:
            if not self._exists(c):
                return 0
            if not self._has_keywords(c):
                return c.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
            return c.execute(f"SELECT COUNT(*) FROM {self.table} WHERE tsv IS NULL").fetchone()[0]

    def delete_doc(self, doc_id):
        with self._conn() as c:
            # Ingestion clears old chunks before the first upsert creates the table.
            if self._exists(c):
                c.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))

    def count(self):
        with self._conn() as c:
            return c.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0] if self._exists(c) else 0
