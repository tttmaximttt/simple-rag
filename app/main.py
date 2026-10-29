import hashlib
import os
import platform
import secrets
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, config, db, rag, stores, tokenizer
from .mcp_server import http_app as mcp_http_app
from .mcp_server import mcp
from .providers import embeddings, llm

STARTED = time.time()
# Public base URL (e.g. https://rag.example.com), used in the OpenAPI spec for GPT Actions.
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
IMPORT_DIR = Path(os.environ.get("IMPORT_DIR", "/import"))
CHECK_TIMEOUT_S = 15
_pool = ThreadPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(_app):
    db.init()
    auth.bootstrap()
    rag.start_worker()
    db.log("info", "app", "started")
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="RAG Studio", lifespan=lifespan)


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@app.middleware("http")
async def authenticate(request: Request, call_next):
    path = request.url.path
    # Public API and MCP: always require an API key created on the Admin page.
    if path == "/v1/openapi.json":
        return await call_next(request)
    if path.startswith("/v1/") or path == "/mcp":
        header = request.headers.get("authorization", "")
        key = (header[7:] if header.startswith("Bearer ") else request.headers.get("x-api-key", "")
               # For clients that only accept a URL (e.g. claude.ai custom connectors).
               or request.query_params.get("key", ""))
        found = key and db.rows("SELECT id FROM api_keys WHERE hash=?", (_key_hash(key),))
        if not found:
            return JSONResponse({"error": "Missing or invalid API key"}, status_code=401)
        db.execute("UPDATE api_keys SET last_used=? WHERE id=?", (time.time(), found[0]["id"]))
        return await call_next(request)
    # Internal API used by the web UI: requires an admin session.
    if path.startswith("/api/") and path not in PUBLIC_API_PATHS:
        user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE))
        if not user:
            return JSONResponse({"error": "Login required"}, status_code=401)
        if user["must_change"] and path not in ("/api/auth/password", "/api/auth/logout"):
            return JSONResponse({"error": "Change the temporary password first"}, status_code=403)
        # CSRF defence in depth (the cookie is also SameSite=Strict): state changes must come from our origin.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD") and origin and origin.split("://", 1)[-1] != request.headers.get("host"):
            return JSONResponse({"error": "Cross-origin request blocked"}, status_code=403)
        request.state.user = user
    # Static UI files are public; the page shows a login form when the API answers 401.
    return await call_next(request)


PUBLIC_API_PATHS = {"/api/health", "/api/auth/status", "/api/auth/login"}


# ---------- auth ----------

def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@app.get("/api/auth/status")
def auth_status(request: Request):
    user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE))
    return {"user": user["username"] if user else None, "must_change_password": bool(user and user["must_change"])}


class Credentials(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def auth_login(body: Credentials, request: Request, response: Response):
    ip = _client_ip(request)
    if auth.locked_out(ip):
        raise HTTPException(429, f"Too many failed attempts. Try again in {auth.LOCKOUT_S // 60} minutes.")
    token = auth.login(body.username, body.password, ip)
    if not token:
        raise HTTPException(401, "Wrong username or password")
    response.set_cookie(auth.SESSION_COOKIE, token, max_age=auth.SESSION_TTL_S, httponly=True,
                        samesite="strict", secure=request.url.scheme == "https")
    return {"ok": True}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response):
    auth.logout(request.cookies.get(auth.SESSION_COOKIE, ""))
    db.log("info", "auth", f"{request.state.user['username']} logged out")
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"ok": True}


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


@app.post("/api/auth/password")
def auth_password(body: PasswordChange, request: Request):
    try:
        auth.change_password(request.state.user["id"], body.current_password, body.new_password,
                             request.cookies.get(auth.SESSION_COOKIE, ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


# ---------- setup ----------

@app.get("/api/health")
def health():
    return {"ok": True, "configured": config.load()["configured"]}


@app.post("/api/tokenizer/limits")
def token_limits(body: dict | None = None):
    """Embedding model limit and effective chunk size for (possibly unsaved) settings."""
    old = config.load()
    cfg = config._merge(old, config.unredact(body or {}, old))
    return {"tokenizer": tokenizer.tokenizer_name(), "model_limit": tokenizer.model_limit(cfg),
            "effective_chunk_tokens": tokenizer.effective_chunk_tokens(cfg)}


@app.get("/api/config")
def get_config():
    return config.redact(config.load())


@app.put("/api/config")
def put_config(body: dict):
    old = config.load()
    new = config._merge(old, config.unredact(body, old))
    new["configured"] = True
    config.save(new)
    db.log("info", "setup", f"config saved: store={new['vector_store']['type']} "
           f"embeddings={new['embeddings']['provider']} llm={new['llm']['provider']}")
    return config.redact(new)


def _timed(fn):
    t0 = time.time()
    try:
        detail = fn()
        return {"ok": True, "ms": round((time.time() - t0) * 1000), "detail": detail}
    except Exception as e:
        return {"ok": False, "ms": round((time.time() - t0) * 1000), "error": f"{type(e).__name__}: {e}"}


def _check(component, cfg):
    if component == "vector_store":
        return _timed(lambda: (stores.get(cfg).ping(), "reachable")[1])
    if component == "embeddings":
        return _timed(lambda: f"dimension {len(embeddings.get(cfg).embed(['ping'])[0])}")
    if component == "llm":
        return _timed(lambda: llm.get(cfg).complete("Reply with the single word: pong", "ping")[:80])
    raise HTTPException(404, "unknown component")


@app.post("/api/test/{component}")
def test_component(component: str, body: dict | None = None):
    """Test a component with the (possibly unsaved) settings from the wizard."""
    old = config.load()
    cfg = config._merge(old, config.unredact(body or {}, old))
    return _check(component, cfg)


# ---------- documents ----------

@app.get("/api/documents")
def list_documents():
    return db.rows("SELECT * FROM documents ORDER BY created DESC")


class TooLarge(Exception):
    pass


def _register(filename: str, tmp: Path, size: int) -> dict:
    """Move a downloaded/uploaded temp file into storage and queue it, skipping duplicates."""
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    dup = db.rows("SELECT id, filename FROM documents WHERE sha256=?", (digest,))
    if dup:
        tmp.unlink(missing_ok=True)
        return {"file": filename, "status": "skipped", "reason": f"duplicate of {dup[0]['filename']}"}
    if tmp.read_bytes()[:5] != b"%PDF-":
        tmp.unlink(missing_ok=True)
        return {"file": filename, "status": "skipped", "reason": "not a PDF"}
    doc_id = uuid.uuid4().hex
    tmp.replace(config.UPLOAD_DIR / f"{doc_id}.pdf")
    now = time.time()
    db.execute("INSERT INTO documents (id, filename, size, status, created, updated, sha256) VALUES (?,?,?,?,?,?,?)",
               (doc_id, filename, size, "queued", now, now, digest))
    rag.enqueue(doc_id)
    return {"file": filename, "status": "queued", "id": doc_id}


def _require_setup():
    if not config.load()["configured"]:
        raise HTTPException(400, "Finish setup first")


def _clean_name(name: str) -> str:
    # Keep folder structure from folder uploads (e.g. "contracts/2024/a.pdf"), drop anything unsafe.
    parts = [p for p in Path(name or "upload.pdf").parts if p not in ("", ".", "..", "/", "\\")]
    return "/".join(parts)[-255:] or "upload.pdf"


@app.post("/api/documents")
async def upload(files: list[UploadFile]):
    """Upload one or many PDFs. Non-PDFs, oversized files, and duplicates are skipped and reported."""
    _require_setup()
    results = []
    for f in files:
        name = _clean_name(f.filename)
        if not name.lower().endswith(".pdf"):
            results.append({"file": name, "status": "skipped", "reason": "not a .pdf file"})
            continue
        tmp = config.UPLOAD_DIR / f".tmp-{uuid.uuid4().hex}"
        size = 0
        with tmp.open("wb") as out:
            while chunk := await f.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB << 20:
                    break
                out.write(chunk)
        if size > MAX_UPLOAD_MB << 20:
            tmp.unlink(missing_ok=True)
            results.append({"file": name, "status": "skipped", "reason": f"larger than {MAX_UPLOAD_MB} MB"})
            continue
        results.append(_register(name, tmp, size))
    return {"results": results, "queued": sum(r["status"] == "queued" for r in results)}


class UrlList(BaseModel):
    urls: list[str]


def _download(url: str) -> dict:
    name = _clean_name(Path(httpx.URL(url).path).name or "download.pdf")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    tmp = config.UPLOAD_DIR / f".tmp-{uuid.uuid4().hex}"
    try:
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("only http(s) URLs are supported")
        size = 0
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r, tmp.open("wb") as out:
            r.raise_for_status()
            for chunk in r.iter_bytes(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB << 20:
                    raise TooLarge()
                out.write(chunk)
        res = _register(name, tmp, size)
    except TooLarge:
        res = {"file": name, "status": "skipped", "reason": f"larger than {MAX_UPLOAD_MB} MB"}
    except httpx.HTTPStatusError as e:
        res = {"file": name, "status": "skipped", "reason": f"download failed: HTTP {e.response.status_code}"}
    except Exception as e:
        res = {"file": name, "status": "skipped", "reason": f"download failed: {e}"}
    tmp.unlink(missing_ok=True)
    res["url"] = url
    return res


@app.post("/api/documents/urls")
def import_urls(body: UrlList):
    """Download and ingest a list of PDF URLs."""
    _require_setup()
    urls = [u.strip() for u in body.urls if u.strip()]
    results = list(_pool.map(_download, urls))
    db.log("info", "import", f"URL import: {sum(r['status'] == 'queued' for r in results)}/{len(urls)} queued")
    return {"results": results, "queued": sum(r["status"] == "queued" for r in results)}


@app.post("/api/documents/import-folder")
def import_folder():
    """Ingest every PDF under IMPORT_DIR (a mounted folder), skipping ones already imported."""
    _require_setup()
    if not IMPORT_DIR.is_dir():
        raise HTTPException(404, f"Import folder {IMPORT_DIR} does not exist; mount one in docker-compose.yml")
    results = []
    for path in sorted(IMPORT_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        name = str(path.relative_to(IMPORT_DIR))
        size = path.stat().st_size
        if size > MAX_UPLOAD_MB << 20:
            results.append({"file": name, "status": "skipped", "reason": f"larger than {MAX_UPLOAD_MB} MB"})
            continue
        tmp = config.UPLOAD_DIR / f".tmp-{uuid.uuid4().hex}"
        shutil.copyfile(path, tmp)
        results.append(_register(name, tmp, size))
    db.log("info", "import", f"folder import: {sum(r['status'] == 'queued' for r in results)}/{len(results)} queued")
    return {"folder": str(IMPORT_DIR), "results": results, "queued": sum(r["status"] == "queued" for r in results)}


@app.post("/api/documents/{doc_id}/reindex")
def reindex(doc_id: str):
    if not db.rows("SELECT id FROM documents WHERE id=?", (doc_id,)):
        raise HTTPException(404)
    rag.enqueue(doc_id)
    return {"ok": True}


@app.post("/api/documents/reindex-all")
def reindex_all():
    ids = [d["id"] for d in db.rows("SELECT id FROM documents")]
    for i in ids:
        rag.enqueue(i)
    return {"queued": len(ids)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        stores.get(config.load()).delete_doc(doc_id)
    except Exception as e:
        db.log("warn", "delete", f"could not remove vectors for {doc_id}: {e}")
    (config.UPLOAD_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)
    db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    return {"ok": True}


# ---------- chat ----------

class Question(BaseModel):
    question: str
    k: int | None = None


@app.post("/api/chat")
def chat(q: Question):
    try:
        return rag.answer(q.question)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


# ---------- admin / diagnostics ----------

@app.post("/api/admin/retrieve")
def debug_retrieve(q: Question):
    t0 = time.time()
    try:
        hits = rag.retrieve(q.question, q.k)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)
    return {"ms": round((time.time() - t0) * 1000), "hits": hits}


@app.get("/api/admin/status")
def status():
    cfg = config.load()
    # Run checks concurrently with a cap so one slow component can't stall the page.
    names = ("vector_store", "embeddings")
    futures = {c: _pool.submit(_check, c, cfg) for c in names}
    checks = {}
    for c, f in futures.items():
        try:
            checks[c] = f.result(timeout=CHECK_TIMEOUT_S)
        except TimeoutError:
            checks[c] = {"ok": False, "ms": CHECK_TIMEOUT_S * 1000,
                         "error": f"no response within {CHECK_TIMEOUT_S}s (a local model may still be downloading)"}
    vector_count = None
    try:
        vector_count = stores.get(cfg).count()
    except Exception:
        pass
    docs = db.rows("SELECT status, COUNT(*) n, COALESCE(SUM(chunks),0) chunks, COALESCE(SUM(size),0) bytes "
                   "FROM documents GROUP BY status")
    since = time.time() - 86400
    q = db.rows("SELECT COUNT(*) n, SUM(1-ok) errors, AVG(total_ms) avg_ms, AVG(retrieve_ms) avg_retrieve_ms, "
                "AVG(generate_ms) avg_generate_ms, MAX(total_ms) max_ms FROM queries WHERE ts > ?", (since,))[0]
    disk = shutil.disk_usage(config.DATA_DIR)
    return {
        "configured": cfg["configured"],
        "components": checks,
        "settings": {"vector_store": cfg["vector_store"]["type"], "embeddings": cfg["embeddings"]["provider"],
                     "llm": f"{cfg['llm']['provider']} / {cfg['llm'][cfg['llm']['provider']]['model']}",
                     **cfg["rag"], "tokenizer": tokenizer.tokenizer_name(),
                     "embedding_token_limit": tokenizer.model_limit(cfg),
                     "effective_chunk_tokens": tokenizer.effective_chunk_tokens(cfg)},
        "documents": {d["status"]: d for d in docs},
        "vectors": vector_count,
        "queue": rag.queue_size(),
        "queries_24h": q,
        "system": {"uptime_s": round(time.time() - STARTED), "python": platform.python_version(),
                   "disk_free_gb": round(disk.free / 1e9, 1), "disk_total_gb": round(disk.total / 1e9, 1)},
    }


@app.get("/api/admin/keys")
def list_keys():
    return db.rows("SELECT id, name, prefix, created, last_used FROM api_keys ORDER BY id DESC")


class KeyName(BaseModel):
    name: str = "default"


@app.post("/api/admin/keys")
def create_key(body: KeyName):
    key = "rag_" + secrets.token_urlsafe(32)
    db.execute("INSERT INTO api_keys (name, prefix, hash, created) VALUES (?,?,?,?)",
               (body.name, key[:10], _key_hash(key), time.time()))
    db.log("info", "api", f"API key created: {body.name}")
    return {"key": key}  # shown once; only the hash is stored


@app.delete("/api/admin/keys/{key_id}")
def delete_key(key_id: int):
    db.execute("DELETE FROM api_keys WHERE id=?", (key_id,))
    db.log("info", "api", f"API key {key_id} revoked")
    return {"ok": True}


@app.post("/api/admin/check-llm")
def check_llm():
    return _check("llm", config.load())


@app.get("/api/admin/events")
def events(limit: int = 200, level: str | None = None):
    if level:
        return db.rows("SELECT * FROM events WHERE level=? ORDER BY id DESC LIMIT ?", (level, limit))
    return db.rows("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/api/admin/queries")
def queries(limit: int = 100):
    return db.rows("SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,))


# ---------- public API (/v1, API-key auth) ----------

class QueryIn(BaseModel):
    question: str
    top_k: int | None = None


class RetrieveIn(BaseModel):
    query: str
    top_k: int | None = None


@app.post("/v1/query", operation_id="askDocuments")
def v1_query(q: QueryIn):
    """Retrieve relevant chunks and generate an answer with citations."""
    try:
        return rag.answer(q.question, q.top_k)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.post("/v1/retrieve", operation_id="searchDocuments")
def v1_retrieve(q: RetrieveIn):
    """Retrieval only: the top matching chunks, for use with your own LLM."""
    try:
        return {"results": rag.retrieve(q.query, q.top_k)}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.get("/v1/documents", operation_id="listDocuments")
def v1_documents():
    return list_documents()


@app.post("/v1/documents", operation_id="uploadDocuments", include_in_schema=False)
async def v1_upload(files: list[UploadFile]):
    return await upload(files)


@app.post("/v1/documents/urls", operation_id="importDocumentUrls", include_in_schema=False)
def v1_import_urls(body: UrlList):
    return import_urls(body)


@app.get("/v1/openapi.json", include_in_schema=False)
def v1_openapi(request: Request):
    """OpenAPI spec of the /v1 API only, with a server URL and bearer auth, for ChatGPT GPT Actions."""
    routes = [r for r in app.routes if getattr(r, "path", "").startswith("/v1/")]
    spec = get_openapi(title="RAG Studio API", version="1.0.0", routes=routes,
                       description="Search and ask questions about the user's uploaded PDF documents.")
    spec["servers"] = [{"url": PUBLIC_URL or str(request.base_url).rstrip("/")}]
    spec.setdefault("components", {})["securitySchemes"] = {"bearerAuth": {"type": "http", "scheme": "bearer"}}
    spec["security"] = [{"bearerAuth": []}]
    return spec


# MCP endpoint at /mcp; must come before the static catch-all mount.
app.router.routes.extend(mcp_http_app.routes)

app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
