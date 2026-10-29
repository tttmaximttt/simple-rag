import pytest

from app import auth, main
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
