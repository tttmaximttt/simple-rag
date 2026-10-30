import hashlib
import re

import psycopg
from pgvector.psycopg import register_vector

from . import dimension_mismatch


MAX_IDENT = 63  # Postgres truncates identifiers at 63 bytes
INDEX_SUFFIX = "_hnsw"
HNSW_VECTOR_MAX = 2000  # vector HNSW limit; larger dims use halfvec (limit 4000)


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
                text TEXT, embedding {col})""")
            c.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_doc ON {self.table} (doc_id)")
            c.execute(f"CREATE INDEX IF NOT EXISTS {self.table}{INDEX_SUFFIX} ON {self.table} "
                      f"USING hnsw (embedding {ops})")

    def upsert(self, items):
        import numpy as np
        with self._conn() as c, c.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table} (id, doc_id, filename, page, chunk, text, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                [(i, p["doc_id"], p["filename"], p["page"], p["chunk"], p["text"], np.array(v, dtype=np.float32))
                 for i, v, p in items])

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

    def delete_doc(self, doc_id):
        with self._conn() as c:
            # Ingestion clears old chunks before the first upsert creates the table.
            if self._exists(c):
                c.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))

    def count(self):
        with self._conn() as c:
            return c.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0] if self._exists(c) else 0
