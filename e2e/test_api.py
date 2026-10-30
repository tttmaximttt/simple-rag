"""The public /v1 API and the document pipeline behind it, over real HTTP."""
import httpx
import pytest

from conftest import ANSWER, BASE_URL, FAKE_URL, LIGHTHOUSE, QUESTION, RUN, upload, wait_ready
from pdfs import make_pdf


def test_openapi_is_public_and_lists_v1_only():
    spec = httpx.get(BASE_URL + "/v1/openapi.json").json()
    assert {"/v1/query", "/v1/retrieve", "/v1/documents"} <= set(spec["paths"])
    assert not [p for p in spec["paths"] if not p.startswith("/v1/")]
    assert spec["servers"][0]["url"]


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/api/documents", "/api/config", "/api/admin/status"])
def test_internal_routes_are_closed(path):
    assert httpx.get(BASE_URL + path).status_code in (401, 404)


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer rag_wrong"}, {"X-API-Key": "rag_wrong"}])
def test_v1_needs_a_valid_key(headers):
    assert httpx.get(BASE_URL + "/v1/documents", headers=headers).status_code == 401


def test_every_key_style_works(api_key):
    for kw in ({"headers": {"Authorization": f"Bearer {api_key}"}}, {"headers": {"X-API-Key": api_key}},
               {"params": {"key": api_key}}):
        assert httpx.get(BASE_URL + "/v1/documents", **kw).status_code == 200


def test_revoked_key_stops_working(admin):
    key = admin.post("/api/admin/keys", json={"name": "to-revoke"}).json()["key"]
    auth = {"Authorization": f"Bearer {key}"}
    assert httpx.get(BASE_URL + "/v1/documents", headers=auth).status_code == 200
    key_id = next(k["id"] for k in admin.get("/api/admin/keys").json() if k["name"] == "to-revoke")
    assert admin.get("/api/admin/keys").json()[0]["last_used"]
    admin.delete(f"/api/admin/keys/{key_id}")
    assert httpx.get(BASE_URL + "/v1/documents", headers=auth).status_code == 401


def test_ingested_document(lighthouse):
    assert lighthouse["filename"] == "lighthouse.pdf"
    assert lighthouse["pages"] == 2
    assert lighthouse["chunks"] >= 2


def test_retrieve_finds_the_right_page(v1, lighthouse):
    r = v1.post("/v1/retrieve", json={"query": QUESTION, "top_k": 3})
    assert r.status_code == 200, r.text
    top = r.json()["results"][0]
    assert (top["doc_id"], top["page"]) == (lighthouse["id"], 1)
    assert ANSWER in top["text"]
    assert len(r.json()["results"]) <= 3


def test_query_answers_with_sources(v1, lighthouse):
    r = v1.post("/v1/query", json={"question": QUESTION})
    assert r.status_code == 200, r.text
    body = r.json()
    assert ANSWER in body["answer"]
    assert body["sources"][0]["filename"] == "lighthouse.pdf"
    assert set(body["timing_ms"]) == {"retrieve", "generate"}


def test_duplicates_and_non_pdfs_are_skipped(v1, lighthouse):
    r = v1.post("/v1/documents", files=[
        ("files", ("again.pdf", make_pdf(LIGHTHOUSE), "application/pdf")),
        ("files", ("notes.txt", b"plain text", "text/plain")),
        ("files", ("fake.pdf", b"not really a pdf", "application/pdf")),
    ]).json()
    reasons = {x["file"]: x["reason"] for x in r["results"]}
    assert r["queued"] == 0
    assert reasons == {"again.pdf": "duplicate of lighthouse.pdf", "notes.txt": "not a .pdf file",
                       "fake.pdf": "not a PDF"}


def test_folder_upload_keeps_paths(v1):
    res = upload(v1, "reports/2024/q1.pdf", [f"Quarterly report {RUN}: revenue of the walrus division grew."])
    assert res["file"] == "reports/2024/q1.pdf"
    assert wait_ready(v1, res["id"])["filename"] == "reports/2024/q1.pdf"
    evil = upload(v1, "../../etc/evil.pdf", [f"Path traversal attempt {RUN}."])
    assert evil["file"] == "etc/evil.pdf"


def test_url_import(v1):
    r = v1.post("/v1/documents/urls", json={"urls": [
        f"{FAKE_URL}/files/kestrel-{RUN}.pdf", f"{FAKE_URL}/files/missing.pdf", "ftp://example.com/x.pdf"]})
    assert r.status_code == 200, r.text
    results = {x["url"].rsplit("/", 1)[1]: x for x in r.json()["results"]}
    assert r.json()["queued"] == 1
    assert results["missing.pdf"]["reason"] == "download failed: HTTP 404"
    assert "only http(s)" in results["x.pdf"]["reason"]
    wait_ready(v1, results[f"kestrel-{RUN}.pdf"]["id"])
    hits = v1.post("/v1/retrieve", json={"query": f"kestrel bridges codename kestrel-{RUN}"}).json()["results"]
    assert hits[0]["filename"] == f"kestrel-{RUN}.pdf"


def test_server_folder_import(admin, import_dir):
    sub = import_dir / f"batch-{RUN}"
    sub.mkdir(parents=True)
    (sub / "a.pdf").write_bytes(make_pdf([f"Folder import A {RUN}: the heron count was 41."]))
    (sub / "b.PDF").write_bytes(make_pdf([f"Folder import B {RUN}: the otter count was 12."]))
    (sub / "readme.txt").write_text("ignored")

    first = admin.post("/api/documents/import-folder").json()
    mine = [x for x in first["results"] if x["file"].startswith(f"batch-{RUN}/")]
    assert sorted(x["file"] for x in mine) == [f"batch-{RUN}/a.pdf", f"batch-{RUN}/b.PDF"]
    assert all(x["status"] == "queued" for x in mine)
    for x in mine:
        wait_ready(admin, x["id"], path="/api/documents")

    again = admin.post("/api/documents/import-folder").json()
    assert again["queued"] == 0
    assert all(x["reason"].startswith("duplicate") for x in again["results"] if x["file"].startswith(f"batch-{RUN}/"))


def test_reindex_and_delete(admin, v1):
    res = upload(v1, "temporary.pdf", [f"A short-lived document {RUN} about narwhal migration."])
    doc = wait_ready(v1, res["id"])
    assert admin.post(f"/api/documents/{doc['id']}/reindex").status_code == 200
    assert wait_ready(v1, doc["id"])["chunks"] == doc["chunks"]

    hits = v1.post("/v1/retrieve", json={"query": f"narwhal migration {RUN}", "top_k": 10}).json()["results"]
    assert doc["id"] in {h["doc_id"] for h in hits}
    assert admin.delete(f"/api/documents/{doc['id']}").status_code == 200
    assert doc["id"] not in {d["id"] for d in v1.get("/v1/documents").json()}
    hits = v1.post("/v1/retrieve", json={"query": f"narwhal migration {RUN}", "top_k": 10}).json()["results"]
    assert doc["id"] not in {h["doc_id"] for h in hits}


def test_admin_status_and_logs(admin, v1, lighthouse):
    v1.post("/v1/query", json={"question": QUESTION})
    s = admin.get("/api/admin/status").json()
    assert s["configured"] is True
    assert all(c["ok"] for c in s["components"].values()), s["components"]
    assert s["documents"]["ready"]["n"] >= 1
    assert s["vectors"] >= lighthouse["chunks"]
    assert s["queries_24h"]["n"] >= 1
    assert admin.post("/api/admin/check-llm").json()["ok"] is True

    events = admin.get("/api/admin/events").json()
    assert any(e["source"] == "ingest" and "lighthouse.pdf" in e["message"] for e in events)
    assert any(e["source"] == "auth" for e in events)
    assert any(q["question"] == QUESTION and q["ok"] for q in admin.get("/api/admin/queries").json())

    hits = admin.post("/api/admin/retrieve", json={"question": QUESTION, "k": 2}).json()["hits"]
    assert len(hits) == 2 and hits[0]["doc_id"] == lighthouse["id"]


def test_cross_origin_writes_are_blocked(admin):
    r = admin.post("/api/admin/keys", json={"name": "csrf"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
