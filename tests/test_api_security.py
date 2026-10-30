import asyncio
import hashlib
import socket
import threading
import time
import uuid

import httpx
import pytest

from app import db, main, mcp_server, rag


@pytest.fixture
def ready(monkeypatch):
    """Setup counts as finished and nothing is ingested; returns the queued ids. Removes documents it added."""
    queued = []
    monkeypatch.setattr(main, "_require_setup", lambda: None)
    monkeypatch.setattr(rag, "enqueue", queued.append)
    yield queued
    for i in queued:
        db.execute("DELETE FROM documents WHERE id=?", (i,))


def _pdf():
    return b"%PDF-1.4 " + uuid.uuid4().hex.encode()  # unique, so never a duplicate of another test's file


def _key(client, scope=None):
    body = {"name": "k"} | ({"scope": scope} if scope else {})
    return {"authorization": f"Bearer {client.post('/api/admin/keys', json=body).json()['key']}"}


# ---------- URL import: no private addresses, generic errors ----------

@pytest.fixture
def net(monkeypatch):
    """Fake DNS (host -> address) and HTTP (Host header -> handler); records the requests made."""
    dns, sites, seen = {}, {}, []

    def getaddrinfo(host, port, *args, **kwargs):
        if host not in dns:
            raise socket.gaierror("no such host")
        family = socket.AF_INET6 if ":" in dns[host] else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (dns[host], 0))]

    def handle(request):
        seen.append((str(request.url), request.headers["host"], request.extensions.get("sni_hostname")))
        return sites[request.headers["host"]](request)

    real = httpx.Client
    monkeypatch.setattr(main.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(main.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handle), **kw))
    return dns, sites, seen


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "172.17.0.2", "192.168.1.1", "169.254.169.254",
                                "0.0.0.0", "224.0.0.1", "100.64.0.1", "::1", "::ffff:127.0.0.1", "fd00::1",
                                "fe80::1"])
def test_private_addresses_are_blocked(net, ready, ip):
    dns, sites, seen = net
    dns["internal.example"] = ip
    sites["internal.example"] = lambda r: httpx.Response(200, content=_pdf())
    r = main._download("http://internal.example/a.pdf")
    assert r["status"] == "skipped" and r["reason"] == "download failed"
    assert seen == []


def test_ip_literals_are_checked_too(net, ready):
    net[0]["127.0.0.1"] = "127.0.0.1"
    assert main._download("http://127.0.0.1:6379/a.pdf")["reason"] == "download failed"


def test_blocked_and_unknown_hosts_get_the_same_reason(net, ready):
    net[0]["internal.example"] = "10.0.0.5"
    assert main._download("http://internal.example/a.pdf")["reason"] == "download failed"
    assert main._download("http://nowhere.example/a.pdf")["reason"] == "download failed"
    logged = [r["message"] for r in db.rows("SELECT message FROM events WHERE source='import' ORDER BY id DESC "
                                              "LIMIT 2")]
    assert logged == ["download refused: nowhere.example does not resolve",
                      "download refused: internal.example resolves to a non-public address"]


def test_header_drip_is_cut_off_at_the_deadline(ready, monkeypatch):
    """A server that sends its headers one byte at a time never trips a read timeout; the deadline still holds."""
    monkeypatch.setattr(main, "ALLOW_PRIVATE_URLS", True)
    monkeypatch.setattr(main, "DOWNLOAD_TIMEOUT_S", 1)
    server = socket.create_server(("127.0.0.1", 0))
    stop = threading.Event()

    def drip():
        conn, _ = server.accept()
        with conn:
            conn.recv(4096)
            for byte in b"HTTP/1.1 200 OK\r\n" + b"X-Slow: " + b"a" * 1000:
                if stop.wait(0.1):
                    return
                try:
                    conn.sendall(bytes([byte]))
                except OSError:
                    return

    thread = threading.Thread(target=drip, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        r = main._download(f"http://127.0.0.1:{server.getsockname()[1]}/a.pdf")
    finally:
        stop.set()
        server.close()
    assert r["reason"] == "took too long"
    assert time.monotonic() - started < 3
    assert ready == []


def test_private_addresses_can_be_allowed(net, ready, monkeypatch):
    monkeypatch.setattr(main, "ALLOW_PRIVATE_URLS", True)
    dns, sites, _ = net
    dns["nas.lan"] = "192.168.1.10"
    sites["nas.lan"] = lambda r: httpx.Response(200, content=_pdf())
    assert main._download("http://nas.lan/a.pdf")["status"] == "queued"


def test_public_address_is_fetched_from_the_checked_ip(net, ready):
    dns, sites, seen = net
    dns["docs.example"] = "93.184.215.14"
    body = _pdf()
    sites["docs.example:8443"] = lambda r: httpx.Response(200, content=body)
    r = main._download("https://docs.example:8443/reports/q3.pdf")
    assert r["status"] == "queued" and r["file"] == "q3.pdf"
    assert seen == [("https://93.184.215.14:8443/reports/q3.pdf", "docs.example:8443", "docs.example")]
    row = db.rows("SELECT sha256 FROM documents WHERE id=?", (r["id"],))[0]
    assert row["sha256"] == hashlib.sha256(body).hexdigest()


def test_each_redirect_hop_is_checked(net, ready):
    dns, sites, seen = net
    dns["public.example"], dns["internal.example"] = "93.184.215.14", "10.0.0.5"
    sites["public.example"] = lambda r: httpx.Response(302, headers={"location": "http://internal.example/x"})
    sites["internal.example"] = lambda r: httpx.Response(200, content=_pdf())
    r = main._download("http://public.example/a.pdf")
    assert r["reason"] == "download failed"
    assert [s[1] for s in seen] == ["public.example"]


def test_redirects_are_followed_to_public_hosts(net, ready):
    dns, sites, _ = net
    dns["a.example"], dns["b.example"] = "93.184.215.14", "93.184.215.15"
    sites["a.example"] = lambda r: httpx.Response(301, headers={"location": "https://b.example/doc.pdf"})
    sites["b.example"] = lambda r: httpx.Response(200, content=_pdf())
    assert main._download("http://a.example/a.pdf")["status"] == "queued"


def test_redirect_loops_and_other_schemes_stop(net, ready):
    dns, sites, seen = net
    dns["loop.example"], dns["file.example"] = "93.184.215.14", "93.184.215.15"
    sites["loop.example"] = lambda r: httpx.Response(302, headers={"location": "/again"})
    sites["file.example"] = lambda r: httpx.Response(302, headers={"location": "file:///etc/passwd"})
    assert main._download("http://loop.example/a.pdf")["reason"] == "too many redirects"
    assert len(seen) == main.MAX_REDIRECTS + 1
    assert main._download("http://file.example/a.pdf")["reason"] == "only http(s) URLs are supported"
    assert main._download("ftp://file.example/a.pdf")["reason"] == "only http(s) URLs are supported"


def test_download_errors_are_generic(net, ready):
    dns, sites, _ = net
    dns["down.example"], dns["gone.example"], dns["html.example"] = "93.184.215.14", "93.184.215.15", "1.1.1.1"

    def refuse(request):
        raise httpx.ConnectError("[Errno 111] Connection refused to 10.1.2.3:6379")

    sites["down.example"] = refuse
    sites["gone.example"] = lambda r: httpx.Response(404, text="secret internal page")
    sites["html.example"] = lambda r: httpx.Response(200, text="<html>")
    assert main._download("http://down.example/a.pdf")["reason"] == "download failed"
    assert main._download("http://unknown.example/a.pdf")["reason"] == "download failed"
    assert main._download("http://gone.example/a.pdf")["reason"] == "HTTP 404"
    assert main._download("http://html.example/a.pdf")["reason"] == "not a PDF"
    # The detail is kept for the admin.
    assert any("Connection refused" in e["message"] for e in db.rows("SELECT message FROM events"))


def test_too_large_download_is_skipped(net, ready, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0)
    dns, sites, _ = net
    dns["big.example"] = "93.184.215.14"
    sites["big.example"] = lambda r: httpx.Response(200, content=_pdf())
    assert main._download("http://big.example/a.pdf")["reason"] == "larger than 0 MB"


def test_malformed_url_is_skipped(logged_in, ready):
    r = logged_in.post("/api/documents/urls", json={"urls": ["http://[::1", "javascript:alert(1)"]})
    assert r.status_code == 200
    assert [x["reason"] for x in r.json()["results"]] == ["invalid URL", "only http(s) URLs are supported"]


def test_url_list_is_capped(logged_in, ready):
    r = logged_in.post("/api/documents/urls", json={"urls": [f"http://x.example/{i}.pdf" for i in range(101)]})
    assert r.status_code == 400


def test_import_and_status_use_separate_pools():
    assert main._download_pool is not main._check_pool


# ---------- duplicates ----------

def test_concurrent_registers_queue_a_file_once(ready, tmp_path):
    body, n = _pdf(), 8
    tmps = []
    for i in range(n):
        t = main.config.UPLOAD_DIR / f".tmp-test-{uuid.uuid4().hex}"
        t.write_bytes(body)
        tmps.append(t)
    barrier, results = threading.Barrier(n), []

    def register(t):
        barrier.wait()
        results.append(main._register("same.pdf", t, len(body)))

    threads = [threading.Thread(target=register, args=(t,)) for t in tmps]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r["status"] for r in results) == ["queued"] + ["skipped"] * (n - 1)
    assert len(db.rows("SELECT id FROM documents WHERE sha256=?", (hashlib.sha256(body).hexdigest(),))) == 1
    assert not any(t.exists() for t in tmps)


# ---------- API key scopes ----------

def test_keys_default_to_read_and_list_their_scope(logged_in):
    _key(logged_in)
    _key(logged_in, "write")
    assert [k["scope"] for k in logged_in.get("/api/admin/keys").json()] == ["write", "read"]


def test_unknown_scope_is_rejected(logged_in):
    assert logged_in.post("/api/admin/keys", json={"name": "k", "scope": "admin"}).status_code == 422


def test_read_keys_cannot_write(logged_in, ready):
    read = _key(logged_in)
    assert logged_in.get("/v1/documents", headers=read).status_code == 200
    for r in (logged_in.post("/v1/documents", files=[("files", ("a.pdf", _pdf()))], headers=read),
              logged_in.post("/v1/documents/urls", json={"urls": []}, headers=read)):
        assert r.status_code == 403 and r.json() == {"error": "This API key is read-only"}
    assert ready == []


def test_write_keys_can_upload_and_import(logged_in, ready):
    write = _key(logged_in, "write")
    r = logged_in.post("/v1/documents", files=[("files", ("a.pdf", _pdf()))], headers=write)
    assert r.status_code == 200 and r.json()["queued"] == 1
    assert logged_in.post("/v1/documents/urls", json={"urls": []}, headers=write).status_code == 200


def test_keys_from_before_scopes_become_read_only():
    db.execute("ALTER TABLE api_keys DROP COLUMN scope")
    db.execute("INSERT INTO api_keys (name, prefix, hash, created) VALUES ('old', 'rag_x', 'h', 0)")
    db.init()
    db.init()
    assert db.rows("SELECT scope FROM api_keys") == [{"scope": "read"}]


# ---------- limits ----------

@pytest.mark.parametrize("path, field", [("/v1/query", "question"), ("/v1/retrieve", "query")])
def test_long_api_questions_are_rejected(logged_in, path, field):
    r = logged_in.post(path, json={field: "x" * 4001}, headers=_key(logged_in))
    assert r.status_code == 422


def test_long_chat_questions_are_rejected(logged_in):
    assert logged_in.post("/api/chat", json={"question": "x" * 4001}).status_code == 422


@pytest.mark.parametrize("tool, arg", [(mcp_server.search, "query"), (mcp_server.ask, "question")])
def test_long_mcp_questions_are_rejected(tool, arg, monkeypatch):
    monkeypatch.setattr(rag, "search", lambda *a: pytest.fail("searched"))
    monkeypatch.setattr(rag, "answer", lambda *a: pytest.fail("answered"))
    with pytest.raises(ValueError, match="at most 4000 characters"):
        asyncio.run(tool(**{arg: "x" * 4001}))


def test_query_log_keeps_the_newest_rows(monkeypatch):
    monkeypatch.setattr(db, "KEEP_QUERIES", 5)
    db.init()
    for i in range(12):
        db.execute("INSERT INTO queries (question, ts, ok) VALUES (?, ?, 1)", (f"q{i}", i))
    assert [r["question"] for r in db.rows("SELECT question FROM queries ORDER BY id")] == \
        [f"q{i}" for i in range(7, 12)]
    assert "queries_ts" in [r["name"] for r in db.rows("PRAGMA index_list(queries)")]


def test_oversized_upload_is_refused_before_reading(logged_in, ready, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_MB", 1)
    r = logged_in.post("/api/documents", files=[("files", ("a.pdf", b"%PDF-" + b"x" * (2 << 20)))])
    assert r.status_code == 413
    write = _key(logged_in, "write")
    r = logged_in.post("/v1/documents", files=[("files", ("a.pdf", b"%PDF-" + b"x" * (2 << 20)))], headers=write)
    assert r.status_code == 413
    assert ready == []


def _asgi_upload(headers: dict, chunks) -> tuple[int, int]:
    """POST chunks to /v1/documents straight through ASGI with no Content-Length, as a chunked request arrives.
    Returns (status, number of chunks the app read)."""
    chunks, read, sent = iter(chunks), [0], []

    async def receive():
        chunk = next(chunks, None)
        if chunk is None:
            return {"type": "http.request", "body": b""}
        read[0] += 1
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/documents", "raw_path": b"/v1/documents", "query_string": b"", "root_path": "",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
             "client": ("127.0.0.1", 1), "server": ("testserver", 80)}
    asyncio.run(main.app(scope, receive, send))
    return next(m["status"] for m in sent if m["type"] == "http.response.start"), read[0]


def _multipart(n_files: int, file_size: int):
    for i in range(n_files):
        yield (b'--B\r\nContent-Disposition: form-data; name="files"; filename="%d.pdf"\r\n'
               b'Content-Type: application/pdf\r\n\r\n' % i) + _pdf()
        for _ in range(file_size >> 16):
            yield b"x" * (1 << 16)
        yield b"\r\n"
    yield b"--B--\r\n"


def test_chunked_upload_is_cut_off_at_the_request_cap(logged_in, ready, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_MB", 1)
    headers = _key(logged_in, "write") | {"content-type": "multipart/form-data; boundary=B",
                                          "transfer-encoding": "chunked"}
    status, read = _asgi_upload(headers, _multipart(1, 8 << 20))  # 128 chunks of 64 KB
    assert status == 413
    assert read <= 20  # stopped just past 1 MB, not after the whole 8 MB
    assert ready == []


def test_chunked_upload_under_the_cap_still_works(logged_in, ready, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_MB", 1)
    headers = _key(logged_in, "write") | {"content-type": "multipart/form-data; boundary=B"}
    assert _asgi_upload(headers, _multipart(2, 1 << 16)) == (200, 2 * 3 + 1)
    assert len(ready) == 2


def test_too_many_files_are_refused(logged_in, ready, monkeypatch):
    monkeypatch.setattr(main, "MAX_FILES_PER_UPLOAD", 2)
    r = logged_in.post("/api/documents", files=[("files", (f"{i}.pdf", _pdf())) for i in range(3)])
    assert r.status_code == 400 and ready == []
    write = _key(logged_in, "write")
    r = logged_in.post("/v1/documents", files=[("files", (f"{i}.pdf", _pdf())) for i in range(3)], headers=write)
    assert r.status_code == 400 and ready == []


def test_upload_hashes_while_copying(logged_in, ready):
    body = _pdf()
    r = logged_in.post("/api/documents", files=[("files", ("x/../a.pdf", body)), ("files", ("b.txt", b"x"))])
    assert [x["status"] for x in r.json()["results"]] == ["queued", "skipped"]
    row = db.rows("SELECT filename, sha256 FROM documents WHERE id=?", (ready[0],))[0]
    assert row == {"filename": "x/a.pdf", "sha256": hashlib.sha256(body).hexdigest()}
    assert (main.config.UPLOAD_DIR / f"{ready[0]}.pdf").read_bytes() == body


def test_oversized_file_in_upload_is_skipped(logged_in, ready, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0)
    r = logged_in.post("/api/documents", files=[("files", ("a.pdf", _pdf()))])
    assert r.json()["results"][0]["reason"] == "larger than 0 MB"


# ---------- delete, chat, errors, headers ----------

def test_delete_removes_the_row_before_the_vectors(logged_in, ready, monkeypatch):
    doc_id = logged_in.post("/api/documents", files=[("files", ("a.pdf", _pdf()))]).json()["results"][0]["id"]
    seen = []

    class Store:
        def delete_doc(self, i):
            seen.append(db.rows("SELECT id FROM documents WHERE id=?", (i,)))

    monkeypatch.setattr(main.stores, "get", lambda cfg: Store())
    assert logged_in.delete(f"/api/documents/{doc_id}").status_code == 200
    assert seen == [[]]


def test_chat_passes_k(logged_in, monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "answer", lambda q, k=None: calls.append((q, k)) or {"answer": "a", "sources": []})
    logged_in.post("/api/chat", json={"question": "q", "k": 3})
    assert calls == [("q", 3)]


def _fail(*args, **kwargs):
    raise RuntimeError("cannot reach db:5432")


@pytest.mark.parametrize("path, body, message", [("/v1/query", {"question": "q"}, "Answer failed"),
                                                 ("/v1/retrieve", {"query": "q"}, "Search failed")])
def test_api_errors_are_generic(logged_in, monkeypatch, path, body, message):
    monkeypatch.setattr(rag, "answer", _fail)
    monkeypatch.setattr(rag, "search", _fail)
    r = logged_in.post(path, json=body, headers=_key(logged_in))
    assert r.status_code == 502 and r.json() == {"error": message}
    assert any("db:5432" in e["message"] for e in db.rows("SELECT message FROM events WHERE source='api'"))


def test_chat_errors_stay_detailed_for_the_admin(logged_in, monkeypatch):
    monkeypatch.setattr(rag, "answer", _fail)
    assert "db:5432" in logged_in.post("/api/chat", json={"question": "q"}).json()["error"]


@pytest.mark.parametrize("method, path", [("get", "/"), ("get", "/app.js"), ("get", "/api/health"),
                                          ("get", "/api/documents"), ("get", "/v1/documents"),
                                          ("get", "/v1/openapi.json"), ("post", "/mcp")])
def test_security_headers_on_every_response(client, method, path):
    h = getattr(client, method)(path).headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "no-referrer"
    assert h["x-frame-options"] == "DENY"
    assert h["content-security-policy"] == \
        "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
