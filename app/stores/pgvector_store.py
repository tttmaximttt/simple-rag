import hashlib
import math
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
        with c.transaction():
            # Two ingests (or app instances) migrating at once would collide on CREATE INDEX IF NOT EXISTS.
            c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (self.table,))
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
        chunks stored with that configuration, any of its words counting (OR).

        Matches are scored like BM25's IDF part: each query word a chunk contains adds log(1 + (N - df + 0.5) /
        (df + 0.5)), with N and df counted among the chunks of that configuration. ts_rank_cd alone weighs
        every word the same, so a chunk full of common words beat the one with a rare code or name, which is
        what keyword search is for. It still breaks ties."""
        configs = [*TS_CONFIGS.values(), "simple"]
        with self._conn() as c:
            # A table made before hybrid search gets its keyword columns from ensure() at the next (re-)index;
            # a search doesn't run DDL.
            if not self._exists(c) or not self._has_keywords(c):
                return []
            # Each configuration's lexemes of the query, with how many chunks of that configuration hold each
            # (GIN lookups) and how many chunks it has in all.
            stats = c.execute(
                f"""WITH words AS (
                        SELECT DISTINCT cfg, lex
                        FROM unnest(%s::text[]) AS cfg,
                             unnest(tsvector_to_array(to_tsvector(cfg::regconfig, %s))) AS lex)
                    SELECT w.cfg, w.lex,
                           (SELECT COUNT(*) FROM {self.table} t
                            WHERE t.ts_config = w.cfg AND t.tsv @@ quote_literal(w.lex)::tsquery) AS df,
                           (SELECT COUNT(*) FROM {self.table} t WHERE t.ts_config = w.cfg) AS n
                    FROM words w""", (configs, query)).fetchall()
            stats = [(cfg, lex, math.log(1 + (n - df + 0.5) / (df + 0.5))) for cfg, lex, df, n in stats if df]
            if not stats:
                return []
            cfgs, lexes, weights = (list(x) for x in zip(*stats))
            rows = c.execute(
                f"""WITH w AS (SELECT * FROM unnest(%s::text[], %s::text[], %s::float8[]) AS w(cfg, lex, idf)),
                    q AS (SELECT cfg, string_agg(quote_literal(lex), ' | ')::tsquery AS qt FROM w GROUP BY cfg)
                    SELECT t.doc_id, t.filename, t.page, t.chunk, t.text,
                           (SELECT SUM(w.idf) FROM w WHERE w.cfg = t.ts_config
                            AND t.tsv @@ quote_literal(w.lex)::tsquery) AS score
                    FROM {self.table} t JOIN q ON t.ts_config = q.cfg
                    WHERE t.tsv @@ q.qt
                    ORDER BY score DESC, ts_rank_cd(t.tsv, q.qt) DESC, t.id LIMIT %s""",
                (cfgs, lexes, weights, k)).fetchall()
        keys = ("doc_id", "filename", "page", "chunk", "text", "score")
        return [dict(zip(keys, r)) for r in rows]

    def keyword_missing(self, doc_ids=None):
        """Chunks stored before hybrid search, which keyword search can't find until they're re-indexed. With
        doc_ids, only those documents' chunks count: orphans of deleted documents can't be re-indexed."""
        with self._conn() as c:
            if not self._exists(c):
                return 0
            where = "tsv IS NULL" if self._has_keywords(c) else "TRUE"  # no keyword columns yet: every row
            args = ()
            if doc_ids is not None:
                where, args = f"{where} AND doc_id = ANY(%s)", (list(doc_ids),)
            return c.execute(f"SELECT COUNT(*) FROM {self.table} WHERE {where}", args).fetchone()[0]

    def delete_doc(self, doc_id):
        with self._conn() as c:
            # Ingestion clears old chunks before the first upsert creates the table.
            if self._exists(c):
                c.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))

    def count(self):
        with self._conn() as c:
            return c.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0] if self._exists(c) else 0
