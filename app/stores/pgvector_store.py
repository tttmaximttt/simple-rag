import re

import psycopg
from pgvector.psycopg import register_vector


class PgVectorStore:
    def __init__(self, dsn, collection):
        self.dsn = dsn
        self.table = "rag_" + re.sub(r"[^a-z0-9_]", "_", collection.lower())

    def _conn(self):
        c = psycopg.connect(self.dsn, autocommit=True, connect_timeout=5)
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(c)
        return c

    def ping(self):
        with self._conn() as c:
            c.execute("SELECT 1")

    def ensure(self, dim):
        with self._conn() as c:
            c.execute(f"""CREATE TABLE IF NOT EXISTS {self.table} (
                id TEXT PRIMARY KEY, doc_id TEXT, filename TEXT, page INT, chunk INT,
                text TEXT, embedding vector({int(dim)}))""")
            c.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_doc ON {self.table} (doc_id)")
            c.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
                      "USING hnsw (embedding vector_cosine_ops)")

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
        with self._conn() as c:
            rows = c.execute(
                f"""SELECT doc_id, filename, page, chunk, text, 1 - (embedding <=> %s) AS score
                    FROM {self.table} ORDER BY embedding <=> %s LIMIT %s""",
                (np.array(vector, dtype=np.float32), np.array(vector, dtype=np.float32), k)).fetchall()
        keys = ("doc_id", "filename", "page", "chunk", "text", "score")
        return [dict(zip(keys, r)) for r in rows]

    def delete_doc(self, doc_id):
        with self._conn() as c:
            # Ingestion clears old chunks before the first upsert creates the table.
            if c.execute("SELECT to_regclass(%s)", (self.table,)).fetchone()[0]:
                c.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))

    def count(self):
        with self._conn() as c:
            exists = c.execute("SELECT to_regclass(%s)", (self.table,)).fetchone()[0]
            return c.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0] if exists else 0
