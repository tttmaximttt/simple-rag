"""Hybrid search: keyword search in each store, its fusion with vector search, and the search mode."""
import os
import time
import uuid

import pytest
from qdrant_client import QdrantClient

from app import db, rag
from app.stores import bm25, chroma_store, pgvector_store, qdrant_store


def chunk(doc, n, text, vec=(1.0, 0.0, 0.0), language="English", page=1):
    return (f"{doc}:{n}", list(vec), {"doc_id": doc, "filename": f"{doc}.pdf", "page": page, "chunk": n,
                                      "text": text, "language": language})


@pytest.fixture(autouse=True)
def fresh_bm25():
    bm25.clear()
    yield
    bm25.clear()


@pytest.fixture
def chroma(tmp_path):
    return chroma_store.ChromaStore(str(tmp_path / "chroma"), "docs")


@pytest.fixture
def qdrant(monkeypatch):
    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_store, "QdrantClient", lambda **kw: client)
    return qdrant_store.QdrantStore("http://unused", None, "docs")


@pytest.fixture
def pgvector():
    dsn = os.environ.get("PGVECTOR_TEST_DSN")
    if not dsn:
        pytest.skip("PGVECTOR_TEST_DSN not set")
    store = pgvector_store.PgVectorStore(dsn, "t_" + uuid.uuid4().hex[:8])
    yield store
    with store._conn() as c:
        c.execute(f"DROP TABLE IF EXISTS {store.table}")


@pytest.fixture(params=["chroma", "qdrant", "pgvector"])
def store(request):
    return request.getfixturevalue(request.param)


# ---------- keyword search in every store ----------

def test_results_have_the_same_keys_in_every_store(store):
    store.ensure(3)
    store.upsert([chunk("a", 0, "alpha bravo")])
    keys = {"doc_id", "filename", "page", "chunk", "text", "score"}
    assert set(store.search([1.0, 0.0, 0.0], 1)[0]) == keys  # no "language", which only pgvector uses
    assert set(store.keyword_search("alpha", 1)[0]) == keys


def test_keyword_search_finds_exact_codes(store):
    assert store.keyword_search("a52082", 5) == []  # no collection yet
    store.ensure(3)
    store.upsert([chunk("a", 0, "The revision code a52082 identifies the board."),
                  chunk("a", 1, "Power supplies must provide 5V at 3A for the board."),
                  chunk("b", 0, "Set dvfs=2 in config.txt to lower the voltage.")])
    hits = store.keyword_search("which board has revision code a52082?", 5)
    assert hits[0]["doc_id"] == "a" and hits[0]["chunk"] == 0
    assert {"doc_id", "filename", "page", "chunk", "text", "score"} <= set(hits[0])
    assert [h["doc_id"] for h in store.keyword_search("dvfs=2", 5)][:1] == ["b"]
    assert store.keyword_missing() == 0


def test_keyword_search_forgets_deleted_documents(store):
    store.ensure(3)
    store.upsert([chunk("a", 0, "alpha bravo"), chunk("b", 0, "alpha charlie")])
    assert {h["doc_id"] for h in store.keyword_search("alpha", 5)} == {"a", "b"}
    store.delete_doc("a")
    assert [h["doc_id"] for h in store.keyword_search("alpha", 5)] == ["b"]


# ---------- pgvector: tsvector per document language ----------

@pytest.mark.parametrize("language, config", [
    ("English", "english"), ("Spanish", "spanish"), ("German", "german"), (None, "simple"), ("Klingon", "simple"),
])
def test_ts_config_follows_the_document_language(language, config):
    assert pgvector_store.ts_config(language) == config


def test_pgvector_stems_with_each_chunks_language(pgvector):
    pgvector.ensure(3)
    pgvector.upsert([chunk("es", 0, "Los códigos de estado indican el resultado de la petición.", language="Spanish"),
                     chunk("en", 0, "The servers were running out of memory.", language="English"),
                     chunk("xx", 0, "códigos sin idioma detectado", language=None)])
    with pgvector._conn() as c:
        assert dict(c.execute(f"SELECT doc_id, ts_config FROM {pgvector.table}").fetchall()) == {
            "es": "spanish", "en": "english", "xx": "simple"}
    # Spanish stemming matches "código" to "códigos"; the undetected document only matches the exact word.
    assert [h["doc_id"] for h in pgvector.keyword_search("código", 5)] == ["es"]
    assert [h["doc_id"] for h in pgvector.keyword_search("server run", 5)] == ["en"]
    assert {h["doc_id"] for h in pgvector.keyword_search("códigos", 5)} == {"es", "xx"}
    assert pgvector.keyword_search("the of and", 5) == []  # stop words only


def test_pgvector_migrates_a_table_made_before_hybrid_search(pgvector):
    with pgvector._conn() as c:  # the schema before this change
        c.execute(f"CREATE TABLE {pgvector.table} (id TEXT PRIMARY KEY, doc_id TEXT, filename TEXT, page INT, "
                  "chunk INT, text TEXT, embedding vector(3))")
        c.execute(f"INSERT INTO {pgvector.table} VALUES ('a:0', 'a', 'a.pdf', 1, 0, 'old chunk', '[1,0,0]')")
    assert pgvector.keyword_missing() == 1
    assert pgvector.keyword_search("old chunk", 5) == []  # no DDL from a search; the next ingest migrates
    pgvector.ensure(3)
    assert pgvector.keyword_missing() == 1  # the columns exist now, but the old row has no keywords yet
    with pgvector._conn() as c:
        assert c.execute("SELECT to_regclass(%s)", (pgvector.table + "_tsv",)).fetchone()[0] is not None
    pgvector.upsert([chunk("a", 0, "old chunk, re-indexed")])  # what a re-index does
    assert pgvector.keyword_missing() == 0
    assert [h["chunk"] for h in pgvector.keyword_search("old chunk", 5)] == [0]


# ---------- Chroma: the BM25 sidecar is rebuilt from the stored chunks ----------

def test_chroma_sidecar_is_rebuilt_from_stored_chunks(tmp_path):
    path = str(tmp_path / "chroma")
    first = chroma_store.ChromaStore(path, "docs")
    first.ensure(3)
    first.upsert([chunk("a", 0, "the quick brown fox")])
    assert [h["doc_id"] for h in first.keyword_search("fox", 5)] == ["a"]

    # A new chunk is found at once: the upsert invalidated the index.
    first.upsert([chunk("b", 0, "a lazy brown dog")])
    assert [h["doc_id"] for h in first.keyword_search("dog", 5)] == ["b"]

    # Re-indexing a document with the same number of chunks still changes what's found.
    first.upsert([chunk("b", 0, "a sleepy brown cat")])
    assert first.keyword_search("dog", 5) == []

    # After a restart nothing is in memory; the index comes back from what Chroma stored.
    bm25.clear()
    again = chroma_store.ChromaStore(path, "docs")
    hits = again.keyword_search("brown", 5)
    assert {h["doc_id"] for h in hits} == {"a", "b"}
    assert {h["text"] for h in hits} == {"the quick brown fox", "a sleepy brown cat"}


def test_chroma_sidecar_loads_more_than_one_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(chroma_store, "LOAD_BATCH", 2)
    store = chroma_store.ChromaStore(str(tmp_path / "chroma"), "docs")
    store.ensure(3)
    store.upsert([chunk("a", n, f"word{n} shared") for n in range(5)])
    assert len(store.keyword_search("shared", 10)) == 5


def test_bm25_ranks_rare_terms_and_keeps_codes_whole():
    assert bm25.terms("IEEE 802.3at-2009 and Código") == ["ieee", "802.3at-2009", "802", "3at", "2009", "codigo"]
    index = bm25.Index([{"doc_id": "d", "chunk": i, "text": t} for i, t in enumerate(
        ["power power power supply", "power over ethernet 802.3at-2009", "nothing relevant"])])
    assert [h["text"] for h in index.search("802.3at-2009 power", 3)][0] == "power over ethernet 802.3at-2009"
    assert index.search("absent", 3) == []


# ---------- fusing keyword and vector rankings ----------

def h(doc, chunk_no, score=0.5, page=1):
    return {"doc_id": doc, "filename": f"{doc}.pdf", "page": page, "chunk": chunk_no, "text": f"{doc}{chunk_no}",
            "score": score}


def test_rrf_puts_chunks_both_searches_found_first():
    vector = [h("v", 0, 0.9), h("both", 0, 0.8), h("v", 1, 0.7)]
    keyword = [h("k", 0, 12.0), h("both", 0, 9.0)]
    fused = rag.fuse([("vector", vector), ("keyword", keyword)])
    # 2nd in both lists (2/61) beats 1st in only one (1/60); ties keep the order they were first seen in.
    assert [(x["doc_id"], x["chunk"]) for x in fused] == [("both", 0), ("v", 0), ("k", 0), ("v", 1)]
    assert [x["found_by"] for x in fused] == [["keyword", "vector"], ["vector"], ["keyword"], ["vector"]]
    # The score shown is the cosine similarity; a chunk only keyword search found has none.
    assert [x["score"] for x in fused] == [0.8, 0.9, 0.0, 0.7]


def test_rrf_keeps_a_negative_similarity():
    fused = rag.fuse([("keyword", [h("a", 0, 5.0)]), ("vector", [h("a", 0, -0.2)])])
    assert fused[0]["score"] == -0.2


@pytest.fixture
def docs():
    for d in ("v", "k", "both"):
        db.execute("INSERT OR REPLACE INTO documents (id, filename, size, status, created) VALUES (?,?,?,?,?)",
                   (d, f"{d}.pdf", 1, "ready", time.time()))
    yield
    for d in ("v", "k", "both"):
        db.execute("DELETE FROM documents WHERE id=?", (d,))


class Emb:
    def embed(self, texts):
        return [[1.0] for _ in texts]


class HybridStore:
    def __init__(self):
        self.keyword_queries = []

    def search(self, vector, k):
        return [h("v", 0, 0.9), h("both", 0, 0.8, page=2), h("v", 1, 0.7, page=3)]

    def keyword_search(self, query, k):
        self.keyword_queries.append(query)
        return [h("k", 0, 12.0), h("both", 1, 9.0, page=2)]


@pytest.mark.usefixtures("docs")
@pytest.mark.parametrize("mode, pages", [
    ("vector", [("v", 1), ("both", 2), ("v", 3)]),
    ("hybrid", [("v", 1), ("k", 1), ("both", 2), ("v", 3)]),  # k:0 ties v:0 at 1/60 per query
])
def test_retrieve_by_mode(monkeypatch, mode, pages):
    store = HybridStore()
    monkeypatch.setattr(rag.embeddings, "get", lambda cfg: Emb())
    monkeypatch.setattr(rag.stores, "get", lambda cfg: store)
    hits = rag.retrieve("q", 5, ["other query"], mode=mode)
    assert [(x["doc_id"], x["page"]) for x in hits] == pages
    assert store.keyword_queries == (["q", "other query"] if mode == "hybrid" else [])
    found = {(x["doc_id"], x["page"]): x["found_by"] for x in hits}
    if mode == "hybrid":
        # Page 2 of "both" merges a vector chunk and a keyword chunk into one source.
        assert found[("both", 2)] == ["keyword", "vector"] and found[("k", 1)] == ["keyword"]
        assert len(hits) <= 5
    else:
        assert set(map(tuple, found.values())) == {("vector",)}


def test_unknown_search_mode_is_refused(monkeypatch):
    monkeypatch.setattr(rag.llm, "get", lambda cfg: pytest.fail("no LLM call for a bad mode"))
    with pytest.raises(ValueError, match="Unknown search mode 'fuzzy'"):
        rag.search("q", 5, rewrite_question=True, mode="fuzzy")


def test_pgvector_missing_count_ignores_orphans(pgvector):
    pgvector.ensure(3)
    pgvector.upsert([chunk("a", 0, "kept"), chunk("gone", 0, "orphan of a deleted document")])
    with pgvector._conn() as c:
        c.execute(f"UPDATE {pgvector.table} SET tsv = NULL")  # as if stored before hybrid search
    assert pgvector.keyword_missing() == 2
    assert pgvector.keyword_missing(["a"]) == 1  # a re-index can only fix documents that still exist


def test_pgvector_concurrent_migrations_dont_collide(pgvector):
    import threading
    with pgvector._conn() as c:
        c.execute(f"CREATE TABLE {pgvector.table} (id TEXT PRIMARY KEY, doc_id TEXT, filename TEXT, page INT, "
                  "chunk INT, text TEXT, embedding vector(3))")
    errors = []

    def migrate():
        try:
            with pgvector._conn() as c:
                pgvector._migrate(c)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=migrate) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_bm25_updates_in_place_instead_of_reloading():
    key, loads = ("t", "k"), []

    def load():
        loads.append(1)
        return [{"doc_id": "a", "chunk": 0, "text": "old words"}]

    assert bm25.search(key, 1, load, "old", 5)[0]["text"] == "old words"
    bm25.added(key, [{"doc_id": "b", "chunk": 0, "text": "new words"}])  # an ingest batch
    assert bm25.search(key, 2, load, "new", 5)[0]["doc_id"] == "b"
    bm25.added(key, [{"doc_id": "b", "chunk": 0, "text": "replaced"}])  # an upsert of the same chunk
    assert bm25.search(key, 2, load, "new", 5) == [] and bm25.search(key, 2, load, "replaced", 5)
    bm25.removed(key, "a")
    assert bm25.search(key, 1, load, "old", 5) == []
    assert loads == [1]  # built once, then kept current
    # A count the index doesn't have (another process wrote to the collection) rebuilds it.
    assert bm25.search(key, 7, load, "old", 5)[0]["doc_id"] == "a" and loads == [1, 1]


def test_bm25_keeps_only_the_recent_collections(monkeypatch):
    monkeypatch.setattr(bm25, "MAX_INDEXES", 2)
    loads = []
    load = lambda name: (lambda: loads.append(name) or [{"doc_id": name, "chunk": 0, "text": "word"}])
    for name in ("one", "two", "three"):
        bm25.search((name,), 1, load(name), "word", 5)
    bm25.search(("one",), 1, load("one"), "word", 5)  # dropped when "three" came in, so loaded again
    assert loads == ["one", "two", "three", "one"]


def test_bm25_folds_stop_words_like_the_text():
    assert bm25.terms("¿Cuándo, dónde y quién? Más también") == []
    assert bm25.terms("set dvfs=2") == ["set", "dvfs=2", "dvfs", "2"]


@pytest.mark.usefixtures("docs")
def test_page_score_is_the_vector_similarity_even_when_negative(monkeypatch):
    class Store:
        def search(self, vector, k):
            return [h("v", 0, -0.3)]

        def keyword_search(self, query, k):
            return [h("v", 1, 7.0)]  # same page, found by keyword only

    monkeypatch.setattr(rag.embeddings, "get", lambda cfg: Emb())
    monkeypatch.setattr(rag.stores, "get", lambda cfg: Store())
    [page] = rag.retrieve("q", 5, mode="hybrid")
    assert page["score"] == -0.3 and page["found_by"] == ["keyword", "vector"]


def test_pgvector_ranks_the_rare_word_above_common_ones(pgvector):
    pgvector.ensure(3)
    common = [chunk("a", n, "power supply power supply board power") for n in range(8)]
    pgvector.upsert(common + [chunk("b", 0, "the power supply for revision a52082")])
    # Every chunk matches "power supply"; only one has the code, and it must come first.
    assert pgvector.keyword_search("power supply a52082", 3)[0]["doc_id"] == "b"
