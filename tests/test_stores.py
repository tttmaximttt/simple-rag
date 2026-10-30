import os
import uuid

import pytest
from qdrant_client import QdrantClient

from app.stores import chroma_store, pgvector_store, qdrant_store

KEYS = {"doc_id", "filename", "page", "chunk", "text", "score"}


def item(doc, n, vec):
    return (f"{doc}:{n}", vec, {"doc_id": doc, "filename": f"{doc}.pdf", "page": 1, "chunk": n, "text": f"t{doc}{n}"})


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


def test_missing_collection(store):
    assert store.count() == 0
    assert store.search([1.0, 0.0, 0.0], 5) == []
    store.delete_doc("nope")


def test_round_trip_and_delete(store):
    store.ensure(3)
    store.upsert([item("a", 0, [1.0, 0.0, 0.0]), item("a", 1, [0.0, 1.0, 0.0]), item("b", 0, [0.9, 0.1, 0.0])])
    assert store.count() == 3
    res = store.search([1.0, 0.0, 0.0], 3)
    assert [r["doc_id"] for r in res] == ["a", "b", "a"]
    assert set(res[0]) == KEYS
    assert res[0]["score"] > res[1]["score"] > res[2]["score"]
    assert res[0]["text"] == "ta0" and res[0]["filename"] == "a.pdf"
    store.delete_doc("a")
    assert store.count() == 1
    assert [r["doc_id"] for r in store.search([1.0, 0.0, 0.0], 5)] == ["b"]


def test_dimension_change_empty_recreates(store):
    store.ensure(3)
    store.upsert([item("a", 0, [1.0, 0.0, 0.0])])
    store.delete_doc("a")
    store.ensure(4)
    store.upsert([item("a", 0, [1.0, 0.0, 0.0, 0.0])])
    assert store.count() == 1


def test_dimension_change_with_data_raises(store):
    store.ensure(3)
    store.upsert([item("a", 0, [1.0, 0.0, 0.0])])
    with pytest.raises(ValueError, match="holds 1 vectors of dimension 3, but the embedding model makes 4"):
        store.ensure(4)
    store.ensure(3)


def _many(dim):
    items = []
    for n in range(60):
        v = [0.0] * dim
        v[0], v[1] = 1.0, n / 60
        items.append(item("a", n, v))
    return items


def test_pgvector_returns_more_than_ef_search_default(pgvector):
    pgvector.ensure(8)
    pgvector.upsert(_many(8))
    assert len(pgvector.search([1.0] + [0.0] * 7, 55)) == 55


def test_pgvector_large_dim_uses_halfvec(pgvector):
    with pgvector._conn() as c:
        if not c.execute("SELECT 1 FROM pg_available_extension_versions WHERE name = 'vector' "
                         "AND version >= '0.7.0'").fetchone():
            pytest.skip("pgvector < 0.7 has no halfvec")
    pgvector.ensure(3072)
    pgvector.upsert(_many(3072))
    assert len(pgvector.search([1.0] + [0.0] * 3071, 55)) == 55


def test_table_name():
    t = pgvector_store.table_name
    assert t("Docs-1") == "rag_docs_1"
    long_a, long_b = "x" * 80, "x" * 79 + "y"
    for name in (long_a, long_b, "a" * 58, "a" * 59):
        assert len((t(name) + "_hnsw").encode()) <= 63
    assert t("a" * 54) == "rag_" + "a" * 54  # 63 bytes with the suffix: unchanged
    assert t("a" * 55) != "rag_" + "a" * 55
    assert t(long_a) != t(long_b)
    assert t(long_a).startswith("rag_xxx")
    assert t(long_a) == t(long_a)
