"""The optional reranker: providers, where it sits in retrieval, and its Setup settings."""
import time

import httpx
import pytest

from app import config, db, main, rag
from app.providers import reranker

REQ = httpx.Request("POST", "https://api.cohere.com/v2/rerank")


@pytest.mark.parametrize("reply", [
    {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]},  # Cohere, Jina, vLLM
    [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}],  # Hugging Face TEI
])
def test_api_reranker_reads_both_reply_shapes(monkeypatch, reply):
    sent = []
    monkeypatch.setattr(reranker, "post_with_retry", lambda url, **kw: sent.append((url, kw)) or httpx.Response(
        200, json=reply, request=REQ))
    r = reranker.APIReranker("https://api.cohere.com/v2/", "key", "rerank-v3.5")
    assert r.rerank("q", ["a", "b"]) == [0.1, 0.9]
    url, kw = sent[0]
    assert url == "https://api.cohere.com/v2/rerank" and kw["headers"] == {"Authorization": "Bearer key"}
    assert kw["json"] == {"query": "q", "documents": ["a", "b"], "top_n": 2, "model": "rerank-v3.5"}


def test_api_reranker_refuses_a_partial_reply(monkeypatch):
    monkeypatch.setattr(reranker, "post_with_retry", lambda url, **kw: httpx.Response(
        200, json={"results": [{"index": 0, "relevance_score": 0.5}]}, request=REQ))
    with pytest.raises(ValueError, match="scored 1 of 2 passages"):
        reranker.APIReranker("https://x", "", "m").rerank("q", ["a", "b"])


def test_off_is_the_default():
    assert config.DEFAULTS["reranker"]["provider"] == "off" and config.DEFAULTS["reranker"]["top_n"] == 30
    assert reranker.get(config.DEFAULTS) is None


# ---------- in retrieval ----------

class Emb:
    def embed(self, texts):
        return [[1.0] for _ in texts]


class Store:
    def __init__(self):
        self.asked = []

    def search(self, vector, k):
        self.asked.append(k)
        return [{"score": 0.9 - n / 100, "doc_id": "d", "filename": "d.pdf", "page": n + 1, "chunk": n,
                 "text": f"passage {n}"} for n in range(min(k, 40))]


class ReverseReranker:
    """Scores later passages higher, so the order flips."""
    def __init__(self):
        self.seen = []

    def rerank(self, query, texts):
        self.seen.append((query, list(texts)))
        return [float(i) for i in range(len(texts))]


@pytest.fixture
def doc(monkeypatch):
    db.execute("INSERT OR REPLACE INTO documents (id, filename, size, status, created) VALUES (?,?,?,?,?)",
               ("d", "d.pdf", 1, "ready", time.time()))
    yield
    db.execute("DELETE FROM documents WHERE id='d'")


@pytest.fixture
def reranking(monkeypatch, tmp_path, doc):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    config.save(config._merge(config.DEFAULTS, {"reranker": {"provider": "local", "top_n": 10}}))
    store, rr = Store(), ReverseReranker()
    monkeypatch.setattr(rag.embeddings, "get", lambda cfg: Emb())
    monkeypatch.setattr(rag.stores, "get", lambda cfg: store)
    monkeypatch.setattr(rag.reranker, "get", lambda cfg: rr)
    return store, rr


def test_reranker_reorders_the_top_n_before_top_k(reranking):
    store, rr = reranking
    hits = rag.retrieve("which passage?", 3, ["another query"])
    # It saw the first 10 fused chunks, scored against the question, and put the last of them first.
    [(query, texts)] = rr.seen
    assert query == "which passage?" and texts == [f"passage {n}" for n in range(10)]
    assert [h["page"] for h in hits] == [10, 9, 8]
    assert [h["rerank_score"] for h in hits] == [9.0, 8.0, 7.0]
    # Each search fetched enough candidates for the reranker, not only top_k * 3.
    assert store.asked == [10, 10]


def test_no_reranker_means_no_rerank_score(reranking, monkeypatch):
    monkeypatch.setattr(rag.reranker, "get", lambda cfg: None)
    hits = rag.retrieve("q", 3)
    assert [h["page"] for h in hits] == [1, 2, 3] and all("rerank_score" not in h for h in hits)


# ---------- Setup ----------

@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    config.save(config._merge(config.DEFAULTS, {"configured": True, "vector_store": {"type": "chroma"}}))


@pytest.mark.parametrize("body, error", [
    ({"reranker": {"provider": "magic"}}, "Unknown reranker provider"),
    ({"reranker": {"top_n": 0}}, "top N must be a whole number from 1 to 200"),
    ({"reranker": {"top_n": 500}}, "top N"),
    ({"reranker": {"top_n": "30"}}, "top N"),
])
def test_bad_reranker_settings_are_refused(logged_in, saved, body, error):
    r = logged_in.put("/api/config", json=body)
    assert r.status_code == 400 and error in r.json()["detail"]


def test_reranker_settings_round_trip_with_the_key_masked(logged_in, saved):
    body = {"reranker": {"provider": "api", "top_n": 20, "api": {"base_url": "https://api.jina.ai/v1",
                                                                  "api_key": "jina-secret", "model": "jina-reranker-v2"}}}
    assert logged_in.put("/api/config", json=body).status_code == 200
    got = logged_in.get("/api/config").json()["reranker"]
    assert got["provider"] == "api" and got["top_n"] == 20 and got["api"]["api_key"] == config.MASK
    assert config.load()["reranker"]["api"]["api_key"] == "jina-secret"


def test_test_button(logged_in, saved, monkeypatch):
    r = logged_in.post("/api/test/reranker", json={}).json()
    assert r["ok"] and r["detail"] == "off"

    class Good:
        def rerank(self, q, texts):
            return [0.98, 0.01]

    monkeypatch.setattr(main.reranker, "get", lambda cfg: Good())
    r = logged_in.post("/api/test/reranker", json={"reranker": {"provider": "local"}}).json()
    assert r["ok"] and "ranked the relevant passage first (0.980 vs 0.010)" in r["detail"]

    class Backwards:
        def rerank(self, q, texts):
            return [0.01, 0.98]

    monkeypatch.setattr(main.reranker, "get", lambda cfg: Backwards())
    r = logged_in.post("/api/test/reranker", json={"reranker": {"provider": "local"}}).json()
    assert not r["ok"] and "ranked the unrelated passage higher" in r["error"]


def test_local_reranker_models_are_listed(logged_in, saved):
    r = logged_in.post("/api/models/reranker", json={"reranker": {"provider": "local"}}).json()
    assert r["ok"] and "BAAI/bge-reranker-v2-m3-int8" in r["models"]
    assert logged_in.post("/api/models/reranker", json={}).json() == {"ok": True, "models": []}  # off
