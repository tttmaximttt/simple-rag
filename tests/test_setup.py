import time
import uuid

import httpx
import pytest

from app import config, db, main, pdftext, rag
from app.providers import llm

SAVED = {
    "configured": True,
    "vector_store": {"type": "chroma", "collection": "docs"},
    "embeddings": {"provider": "openai", "openai": {"api_key": "sk-saved", "model": "text-embedding-3-small",
                                                    "base_url": "https://api.openai.com/v1"}},
    "llm": {"provider": "openai", "openai": {"api_key": "sk-saved", "model": "gpt-4o-mini",
                                             "base_url": "https://api.openai.com/v1"}},
}


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """Saved settings (in a temp file) with one indexed document; re-index requests are recorded."""
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    config.save(config._merge(config.DEFAULTS, SAVED))
    doc_id = uuid.uuid4().hex
    db.execute("INSERT INTO documents (id, filename, size, status, created) VALUES (?,?,?,?,?)",
               (doc_id, "a.pdf", 1, "ready", time.time()))
    queued = []
    monkeypatch.setattr(rag, "enqueue", queued.append)
    yield queued
    db.execute("DELETE FROM documents WHERE id=?", (doc_id,))


# ---------- saved keys only go where they were saved for ----------

def test_masked_key_is_reused_for_the_same_address(logged_in, saved, monkeypatch):
    seen = []
    monkeypatch.setattr(llm.httpx, "get", lambda url, headers, **kw: seen.append(headers) or httpx.Response(
        200, json={"data": [{"id": "gpt-4o"}]}, request=httpx.Request("GET", url)))
    body = {"llm": {"openai": {"api_key": config.MASK, "model": "gpt-4o"}}}
    assert logged_in.post("/api/models/llm", json=body).json()["ok"]
    assert seen == [{"Authorization": "Bearer sk-saved"}]


def test_masked_key_is_not_sent_to_a_new_address(logged_in, saved, monkeypatch):
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **kw: pytest.fail("the saved key must not be sent"))
    body = {"llm": {"openai": {"api_key": config.MASK, "base_url": "https://attacker.example/v1"}}}
    for path in ("/api/models/llm", "/api/test/llm"):
        r = logged_in.post(path, json=body)
        assert r.status_code == 400 and "Enter the API key again: the base URL changed" in r.json()["detail"]
    r = logged_in.put("/api/config", json=body)
    assert r.status_code == 400 and config.load()["llm"]["openai"]["base_url"] == "https://api.openai.com/v1"


def test_token_hint_ignores_secrets(logged_in, saved):
    body = {"embeddings": {"openai": {"api_key": config.MASK, "base_url": "https://other.example/v1"}}}
    assert logged_in.post("/api/tokenizer/limits", json=body).status_code == 200


# ---------- changing what the index depends on ----------

def test_llm_changes_save_without_asking(logged_in, saved):
    r = logged_in.put("/api/config", json={"llm": {"openai": {"model": "gpt-4.1-mini"}}})
    assert r.status_code == 200 and not saved


def test_new_collection_asks_before_saving(logged_in, saved):
    r = logged_in.put("/api/config", json={"vector_store": {"collection": "other"}})
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "reindex_required" and d["documents"] == 1 and d["collection"] is None
    assert config.load()["vector_store"]["collection"] == "docs"


@pytest.mark.parametrize("reindex, queued", [(True, 1), (False, 0)])
def test_new_collection_saves_with_an_answer(logged_in, saved, reindex, queued):
    r = logged_in.put("/api/config", json={"vector_store": {"collection": "other"}, "reindex": reindex})
    assert r.status_code == 200 and config.load()["vector_store"]["collection"] == "other"
    assert len(saved) == queued and "reindex" not in config.load()


@pytest.mark.parametrize("reindex", [None, True, False])
def test_new_embedding_model_needs_a_new_collection(logged_in, saved, reindex):
    body = {"embeddings": {"openai": {"model": "text-embedding-3-large"}}}
    if reindex is not None:
        body["reindex"] = reindex
    r = logged_in.put("/api/config", json=body)
    assert r.status_code == 409 and r.json()["detail"]["collection"] == "docs_text_embedding_3_large"
    assert config.load()["embeddings"]["openai"]["model"] == "text-embedding-3-small" and not saved


def test_new_embedding_model_into_new_collection(logged_in, saved):
    r = logged_in.put("/api/config", json={"embeddings": {"openai": {"model": "text-embedding-3-large"}},
                                           "vector_store": {"collection": "docs_text_embedding_3_large"},
                                           "reindex": True})
    assert r.status_code == 200 and len(saved) == 1


def test_suggested_collection_replaces_its_own_suffix():
    assert main._collection_for("docs", "a", "BAAI/bge-small-en-v1.5") == "docs_baai_bge_small_en_v1_5"
    assert main._collection_for("docs_baai_bge_small_en_v1_5", "BAAI/bge-small-en-v1.5", "text-embedding-3-small") \
        == "docs_text_embedding_3_small"


# ---------- one hit per page ----------

def test_retrieve_merges_chunks_from_the_same_page(monkeypatch):
    hits = [{"score": s, "doc_id": d, "filename": "f", "page": p, "chunk": c, "text": f"t{c}"}
            for s, d, p, c in [(0.9, "a", 1, 2), (0.8, "a", 1, 1), (0.7, "a", 2, 3), (0.6, "b", 1, 0), (0.5, "a", 1, 0)]]
    asked = []

    class Store:
        def search(self, vector, k):
            asked.append(k)
            return hits[:k]

    class Emb:
        def embed(self, texts):
            return [[0.0]]

    monkeypatch.setattr(rag.stores, "get", lambda cfg: Store())
    monkeypatch.setattr(rag.embeddings, "get", lambda cfg: Emb())
    out = rag.retrieve("q", 2)
    assert asked == [6]
    assert [(h["doc_id"], h["page"], h["score"], h["chunk"], h["text"]) for h in out] == [
        ("a", 1, 0.9, 0, "t0 t1 t2"), ("a", 2, 0.7, 3, "t3")]


def test_retrieve_caps_top_k_and_chunks_per_page(monkeypatch):
    hits = [{"score": 1 - i / 100, "doc_id": "a", "filename": "f", "page": 1, "chunk": i, "text": f"t{i}"}
            for i in range(0, 20, 2)]
    asked = []
    monkeypatch.setattr(rag.stores, "get", lambda cfg: type("S", (), {
        "search": lambda self, v, k: asked.append(k) or hits[:k]})())
    monkeypatch.setattr(rag.embeddings, "get", lambda cfg: type("E", (), {"embed": lambda self, t: [[0.0]]})())
    out = rag.retrieve("q", 10**6)
    assert asked == [rag.MAX_TOP_K * rag.CHUNKS_PER_PAGE]
    assert out[0]["text"] == "t0\n…\nt2\n…\nt4"  # 3 chunks, not adjacent


@pytest.mark.parametrize("a, b, joined", [
    ("One. Two. Three.", "Two. Three. Four.", "One. Two. Three. Four."),
    ("Ends with e", "example starts", "Ends with e example starts"),
    ("Same", "Same", "Same"),
    ("No overlap.", "Next part.", "No overlap. Next part."),
])
def test_join_overlap(a, b, joined):
    assert rag._join_overlap(a, b) == joined


def test_top_k_is_bounded_on_the_api(logged_in):
    key = logged_in.post("/api/admin/keys", json={"name": "k"}).json()["key"]
    r = logged_in.post("/v1/retrieve", json={"query": "q", "top_k": 1000}, headers={"x-api-key": key})
    assert r.status_code == 422


# ---------- text cleanup ----------

def test_clean_pages_removes_running_footers_and_glyphs():
    pages = [f"Body text {chr(97 + i)}\nmanual.book {i} ページ 2025年6月{i}日\n Bullet" for i in range(10)]
    out = pdftext.clean_pages(pages)
    assert out[3] == "Body text d\n  Bullet"


def test_clean_pages_keeps_short_repeated_lines_and_small_documents():
    assert pdftext.clean_pages(["NOTE\nA", "NOTE\nB", "NOTE\nC", "NOTE\nD", "NOTE\nE"])[0] == "NOTE\nA"
    assert pdftext.clean_pages(["same footer line here", "same footer line here"]) == \
        ["same footer line here", "same footer line here"]


@pytest.mark.parametrize("line, expected", [
    ("I f  t h e  S R S  a i r b a g  l i g h t  i s  o n ,  i t", "If the SRS airbag light is on, it"),
    ("Brake pedal: C h e c k  t h e  p e d a l  f o r  s m o o t h", "Brake pedal: Check the pedal for smooth"),
    ("Shift to P R N D as needed", "Shift to P R N D as needed"),
    ("A normal line of text", "A normal line of text"),
])
def test_unspace(line, expected):
    assert pdftext._unspace(line) == expected


# ---------- review follow-ups ----------

@pytest.mark.parametrize("path, body", [
    ("/api/models/llm", {"llm": {"openai": {"base_url": "https://attacker.example/v1"}}}),
    ("/api/test/llm", {"llm": {"openai": {"base_url": "https://attacker.example/v1"}}}),
    ("/api/test/vector_store", {"vector_store": {"type": "qdrant", "qdrant": {"url": "https://attacker.example"}}}),
])
def test_saved_key_is_not_sent_when_the_field_is_left_out(logged_in, saved, monkeypatch, path, body):
    cfg = config.load()
    cfg["vector_store"]["qdrant"]["api_key"] = "qk-saved"
    config.save(cfg)
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **kw: pytest.fail("the saved key must not be sent"))
    r = logged_in.post(path, json=body)
    assert r.status_code == 400 and "again" in r.json()["detail"]


def test_a_typed_key_may_go_to_a_new_address(logged_in, saved, monkeypatch):
    seen = []
    monkeypatch.setattr(llm.httpx, "get", lambda url, headers, **kw: seen.append((url, headers)) or httpx.Response(
        200, json={"data": []}, request=httpx.Request("GET", url)))
    body = {"llm": {"openai": {"api_key": "sk-new", "base_url": "https://proxy.example/v1"}}}
    assert logged_in.post("/api/models/llm", json=body).json()["ok"]
    assert seen == [("https://proxy.example/v1/models", {"Authorization": "Bearer sk-new"})]


def test_mask_only_stands_for_secrets(saved):
    assert config.apply({"llm": {"openai": {"model": config.MASK}}}, config.load())["llm"]["openai"]["model"] == config.MASK


def test_first_setup_never_asks(logged_in, saved):
    cfg = config.load()
    cfg["configured"] = False
    config.save(cfg)
    assert logged_in.put("/api/config", json={"vector_store": {"collection": "other"}}).status_code == 200


def test_only_indexed_documents_count(logged_in, saved):
    db.execute("UPDATE documents SET status='failed' WHERE status='ready'")
    assert logged_in.put("/api/config", json={"vector_store": {"collection": "other"}}).status_code == 200


@pytest.mark.parametrize("reindex", ["false", 0, [], "yes"])
def test_reindex_must_be_a_bool(logged_in, saved, reindex):
    r = logged_in.put("/api/config", json={"vector_store": {"collection": "other"}, "reindex": reindex})
    assert r.status_code == 400 and config.load()["vector_store"]["collection"] == "docs"


def test_credential_or_endpoint_changes_are_not_index_changes(logged_in, saved):
    body = {"embeddings": {"openai": {"api_key": "sk-rotated", "base_url": "https://proxy.example/v1"}}}
    assert logged_in.put("/api/config", json=body).status_code == 200


def test_rotating_the_store_password_does_not_allow_mixing_models(logged_in, saved):
    cfg = config.load()
    cfg["vector_store"].update(type="pgvector", pgvector={"dsn": "postgresql://rag:old@db:5432/rag"})
    config.save(cfg)
    body = {"vector_store": {"pgvector": {"dsn": "postgresql://rag:new@db:5432/rag"}},
            "embeddings": {"openai": {"model": "text-embedding-3-large"}}, "reindex": False}
    r = logged_in.put("/api/config", json=body)
    assert r.status_code == 409 and r.json()["detail"]["collection"]


def test_same_model_name_from_another_provider_gets_a_new_collection(logged_in, saved):
    cfg = config.load()
    cfg["embeddings"]["openai"]["model"] = "nomic-embed-text"
    cfg["vector_store"]["collection"] = "docs_nomic_embed_text"
    config.save(cfg)
    r = logged_in.put("/api/config", json={"embeddings": {"provider": "ollama"}})
    suggested = r.json()["detail"]["collection"]
    assert r.status_code == 409 and suggested == "docs_nomic_embed_text_2"
    r = logged_in.put("/api/config", json={"embeddings": {"provider": "ollama"}, "reindex": True,
                                           "vector_store": {"collection": suggested}})
    assert r.status_code == 200


def test_unknown_provider_is_a_400(logged_in, saved):
    assert logged_in.put("/api/config", json={"embeddings": {"provider": "nope"}}).status_code == 400


def test_reindex_skips_queued_and_cancelled(logged_in, saved):
    ids = {}
    for status in ("queued", "cancelled", "processing", "failed"):
        ids[status] = uuid.uuid4().hex
        db.execute("INSERT INTO documents (id, filename, size, status, created) VALUES (?,?,?,?,?)",
                   (ids[status], "x.pdf", 1, status, time.time()))
    try:
        r = logged_in.put("/api/config", json={"vector_store": {"collection": "other"}, "reindex": True})
        assert r.status_code == 200
        assert ids["processing"] in saved and ids["failed"] in saved
        assert ids["queued"] not in saved and ids["cancelled"] not in saved
    finally:
        for i in ids.values():
            db.execute("DELETE FROM documents WHERE id=?", (i,))


def test_lines_on_only_two_pages_of_a_small_document_are_kept():
    pages = ["See the WARNING below now\nA", "See the WARNING below now\nB", "C", "D", "E"]
    assert pdftext.clean_pages(pages)[0] == "See the WARNING below now\nA"


@pytest.mark.parametrize("line", ["1 2 3 4 5", "Y N N Y", "A B C", "x = a + b", "Rows:  1 2 3  4 5 6",
                                  "Gear  P R N D"])
def test_table_rows_are_not_unspaced(line):
    assert pdftext._unspace(line) == line
