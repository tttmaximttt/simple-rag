import httpx
import pytest

from app import auth, main, providers
from app.providers import llm
from conftest import PASSWORD


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_framework_docs_are_off(client, path):
    assert client.get(path).status_code == 404


def test_public_api_spec_still_served(client):
    r = client.get("/v1/openapi.json")
    assert r.status_code == 200
    assert "/v1/query" in r.json()["paths"]


@pytest.mark.parametrize("headers, status", [
    ({"origin": "http://testserver"}, 200),
    ({"host": "app:8080", "origin": "https://proxy.example", "x-forwarded-host": "proxy.example, other"}, 200),
    ({"host": "app:8080", "origin": "https://rag.example.com"}, 200),
    ({}, 200),
    ({"origin": "https://evil.example"}, 403),
    ({"host": "app:8080", "origin": "https://evil.example", "x-forwarded-host": "proxy.example"}, 403),
    ({"origin": "null"}, 403),
])
def test_origin_check(logged_in, monkeypatch, headers, status):
    monkeypatch.setattr(main, "PUBLIC_URL", "https://rag.example.com")
    assert logged_in.post("/api/admin/keys", json={"name": "t"}, headers=headers).status_code == status


def test_api_keys_are_checked(logged_in):
    key = logged_in.post("/api/admin/keys", json={"name": "k"}).json()["key"]
    assert logged_in.get("/v1/documents", headers={"authorization": f"Bearer {key}"}).status_code == 200
    assert logged_in.get("/v1/documents", headers={"authorization": "Bearer nope"}).status_code == 401


@pytest.mark.parametrize("public_url, secure", [("", False), ("https://rag.example.com", True)])
def test_cookie_secure_follows_public_url(client, monkeypatch, public_url, secure):
    monkeypatch.setattr(main, "PUBLIC_URL", public_url)
    r = client.post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
    assert ("secure" in r.headers["set-cookie"].lower()) == secure
    client.cookies.clear()


def test_lockout_returns_429(client):
    codes = [client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code
             for _ in range(auth.MAX_FAILURES + 2)]
    assert codes == [401] * auth.MAX_FAILURES + [429, 429]


def test_oversized_credentials_rejected(client):
    assert client.post("/api/auth/login", json={"username": "a" * 5000, "password": "x"}).status_code == 422


def test_oversized_password_change_rejected(logged_in):
    r = logged_in.post("/api/auth/password", json={"current_password": PASSWORD, "new_password": "a" * 5000})
    assert r.status_code == 422


@pytest.mark.parametrize("host, expected", [
    ("203.0.113.7", "203.0.113.7"),
    ("::1", "::1"),
    ("1.2.3.4\nfake log line" + "x" * 5000, "invalid"),
    (None, "unknown"),
])
def test_client_ip_is_normalised(host, expected):
    request = main.Request({"type": "http", "client": (host, 1234) if host else None})
    assert main._client_ip(request) == expected


def _fake_get(status, body, seen=None):
    def get(url, **kwargs):
        if seen is not None:
            seen.append((url, kwargs.get("headers")))
        return httpx.Response(status, json=body, request=httpx.Request("GET", url))
    return get


def test_models_need_a_session(client):
    assert client.post("/api/models/llm", json={}).status_code == 401


def test_models_unknown_component(logged_in):
    assert logged_in.post("/api/models/vector_store", json={}).status_code == 404


def test_models_lists_chat_models_for_the_draft_settings(logged_in, monkeypatch):
    seen = []
    monkeypatch.setattr(llm.httpx, "get", _fake_get(200, {"data": [
        {"id": "gpt-4o-mini"}, {"id": "text-embedding-3-small"}, {"id": "whisper-1"}, {"id": "gpt-4o"}]}, seen))
    body = {"llm": {"provider": "openai", "openai": {"api_key": "sk-test", "base_url": "http://llm.test/v1/"}}}
    r = logged_in.post("/api/models/llm", json=body).json()
    assert r == {"ok": True, "models": ["gpt-4o", "gpt-4o-mini"]}
    assert seen == [("http://llm.test/v1/models", {"Authorization": "Bearer sk-test"})]


def test_models_lists_embedding_models(logged_in, monkeypatch):
    monkeypatch.setattr(providers.httpx, "get", _fake_get(200, {"data": [
        {"id": "gpt-4o"}, {"id": "text-embedding-3-large"}, {"id": "text-embedding-3-small"}]}))
    body = {"embeddings": {"provider": "openai", "openai": {"base_url": "http://emb.test/v1"}}}
    r = logged_in.post("/api/models/embeddings", json=body).json()
    assert r == {"ok": True, "models": ["text-embedding-3-large", "text-embedding-3-small"]}


def test_models_lists_local_embedding_models(logged_in):
    r = logged_in.post("/api/models/embeddings", json={"embeddings": {"provider": "local"}}).json()
    assert r["ok"] and "BAAI/bge-small-en-v1.5" in r["models"]


def test_models_reports_provider_errors(logged_in, monkeypatch):
    monkeypatch.setattr(llm.httpx, "get", _fake_get(401, {"error": "bad key"}))
    r = logged_in.post("/api/models/llm", json={"llm": {"provider": "anthropic"}}).json()
    assert not r["ok"] and "401" in r["error"] and "bad key" in r["error"]


def test_connect_info_reports_the_public_url(client, logged_in, monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_URL", "https://rag.example.com")
    assert logged_in.get("/api/connect").json() == {"public_url": "https://rag.example.com"}
    client.cookies.clear()
    assert client.get("/api/connect").status_code == 401
