import hashlib
import ipaddress
import os
import platform
import re
import secrets
import shutil
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, config, db, rag, stores, tokenizer
from .mcp_server import MAX_QUERY_CHARS, mcp
from .mcp_server import http_app as mcp_http_app
from .providers import embeddings, llm

STARTED = time.time()
# Public base URL (e.g. https://rag.example.com), used in the OpenAPI spec for GPT Actions.
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
MAX_FILES_PER_UPLOAD = 100
# Cap on one upload request's whole body, counted as it arrives (a chunked request declares no Content-Length).
MAX_REQUEST_MB = int(os.environ.get("MAX_REQUEST_MB", "1024"))
UPLOAD_PATHS = ("/api/documents", "/v1/documents")
IMPORT_DIR = Path(os.environ.get("IMPORT_DIR", "/import"))
# URL imports refuse hosts that resolve to loopback, private, link-local and other non-public addresses, so an API
# key can't reach the Docker network or a cloud metadata endpoint. ALLOW_PRIVATE_URLS=1 lifts that for LAN setups.
ALLOW_PRIVATE_URLS = os.environ.get("ALLOW_PRIVATE_URLS") == "1"
# DOWNLOAD_TIMEOUT_S bounds a whole URL download: every redirect hop, connecting, headers and body.
MAX_URLS, MAX_REDIRECTS, DOWNLOAD_TIMEOUT_S = 100, 5, 120
CHECK_TIMEOUT_S = 15
# Separate pools, so a big URL import can't starve the status checks.
_download_pool = ThreadPoolExecutor(max_workers=4)
_check_pool = ThreadPoolExecutor(max_workers=4)
_register_lock = threading.Lock()
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
    # No inline scripts, event handlers, style= attributes or <style> blocks in the UI: they would be blocked.
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
}

@asynccontextmanager
async def lifespan(_app):
    db.init()
    auth.bootstrap()
    rag.start_worker()
    db.log("info", "app", "started")
    async with mcp.session_manager.run():
        yield


# No public /docs or /openapi.json: they would list every internal route. /v1/openapi.json covers the public API.
app = FastAPI(title="RAG Studio", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def _same_origin(request: Request, origin: str) -> bool:
    # Behind a proxy that rewrites Host, fall back to X-Forwarded-Host or PUBLIC_URL.
    allowed = {request.headers.get("host", ""), request.headers.get("x-forwarded-host", "").split(",")[0].strip(),
               urlsplit(PUBLIC_URL).netloc}
    return urlsplit(origin).netloc.lower() in {h.lower() for h in allowed if h}


class _BodyTooLarge(Exception):
    pass


class LimitUploadBody:
    """Counts an upload's body bytes as they arrive and answers 413 once MAX_REQUEST_MB is passed, whatever the
    headers say, instead of letting Starlette spool the whole body to disk first."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in UPLOAD_PATHS:
            return await self.app(scope, receive, send)
        received, started = 0, False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_MB << 20:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message):
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if started:
                raise
            await _request_too_large()(scope, receive, send)


# Added before the http middlewares below, so it runs inside them: authenticate refuses the request before any
# body is read, and the 413 still gets the security headers.
app.add_middleware(LimitUploadBody)


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
        found = key and db.rows("SELECT id, scope FROM api_keys WHERE hash=?", (auth.secret_hash(key),))
        if not found:
            return JSONResponse({"error": "Missing or invalid API key"}, status_code=401)
        db.execute("UPDATE api_keys SET last_used=? WHERE id=?", (time.time(), found[0]["id"]))
        request.state.api_scope = found[0]["scope"]
        if request.method == "POST" and path in WRITE_API_PATHS and request.state.api_scope != "write":
            return JSONResponse({"error": "This API key is read-only"}, status_code=403)
        return _upload_too_large(request, path) or await call_next(request)
    # Internal API used by the web UI: requires an admin session.
    if path.startswith("/api/") and path not in PUBLIC_API_PATHS:
        user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE))
        if not user:
            return JSONResponse({"error": "Login required"}, status_code=401)
        if user["must_change"] and path not in ("/api/auth/password", "/api/auth/logout"):
            return JSONResponse({"error": "Change the temporary password first"}, status_code=403)
        # CSRF defence in depth (the cookie is also SameSite=Strict): state changes must come from our origin.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD") and origin and not _same_origin(request, origin):
            return JSONResponse({"error": "Cross-origin request blocked"}, status_code=403)
        request.state.user = user
        if too_large := _upload_too_large(request, path):
            return too_large
    # Static UI files are public; the page shows a login form when the API answers 401.
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Added after authenticate, so it wraps it and also covers its early 401/403 answers.
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    return response


def _request_too_large() -> JSONResponse:
    return JSONResponse({"error": f"Upload too large: at most {MAX_REQUEST_MB} MB per request"}, status_code=413)


def _upload_too_large(request: Request, path: str) -> JSONResponse | None:
    """Refuse an upload whose declared size can't fit before its body is read (per-file sizes are checked later)."""
    if request.method != "POST" or path not in UPLOAD_PATHS:
        return None
    try:
        length = int(request.headers.get("content-length", 0))
    except ValueError:
        return JSONResponse({"error": "Invalid Content-Length"}, status_code=400)
    return _request_too_large() if length > MAX_REQUEST_MB << 20 else None


PUBLIC_API_PATHS = {"/api/health", "/api/auth/status", "/api/auth/login"}
# Need an API key with the "write" scope; any key can use the other /v1 routes and /mcp.
WRITE_API_PATHS = {"/v1/documents", "/v1/documents/urls"}


# ---------- auth ----------

def _client_ip(request: Request) -> str:
    if not request.client:
        return "unknown"
    # uvicorn copies X-Forwarded-For from trusted proxies unchecked; keep junk out of lockout keys and logs.
    try:
        return str(ipaddress.ip_address(request.client.host))
    except ValueError:
        return "invalid"


@app.get("/api/auth/status")
def auth_status(request: Request):
    user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE))
    return {"user": user["username"] if user else None, "must_change_password": bool(user and user["must_change"])}


class Credentials(BaseModel):
    username: str = Field(max_length=128)
    password: str = Field(max_length=1024)


@app.post("/api/auth/login")
def auth_login(body: Credentials, request: Request, response: Response):
    ip = _client_ip(request)
    try:
        token = auth.login(body.username, body.password, ip)
    except auth.LockedOut:
        raise HTTPException(429, f"Too many failed attempts. Try again in {auth.LOCKOUT_S // 60} minutes.")
    if not token:
        raise HTTPException(401, "Wrong username or password")
    response.set_cookie(auth.SESSION_COOKIE, token, max_age=auth.SESSION_TTL_S, httponly=True, samesite="strict",
                        secure=PUBLIC_URL.startswith("https://") or request.url.scheme == "https")
    return {"ok": True}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response):
    auth.logout(request.cookies.get(auth.SESSION_COOKIE, ""))
    db.log("info", "auth", f"{request.state.user['username']} logged out")
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"ok": True}


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=1024)
    new_password: str = Field(max_length=1024)


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


def _draft_config(body: dict | None) -> dict:
    try:
        cfg = config.apply(body or {}, config.load())
    except config.SecretNeeded as e:
        raise HTTPException(400, str(e))
    for section, key in (("vector_store", "type"), ("embeddings", "provider"), ("llm", "provider")):
        if not isinstance(config.DEFAULTS[section].get(cfg[section][key]), dict):
            raise HTTPException(400, f"Unknown {section.replace('_', ' ')} {key}: {cfg[section][key]!r}")
    return cfg


@app.post("/api/tokenizer/limits")
def token_limits(body: dict | None = None):
    """Embedding model limit and effective chunk size for (possibly unsaved) settings."""
    cfg = config._merge(config.load(), body or {})  # needs no secrets, so a changed URL beside a mask is fine
    return {"tokenizer": tokenizer.tokenizer_name(), "model_limit": tokenizer.model_limit(cfg),
            "effective_chunk_tokens": tokenizer.effective_chunk_tokens(cfg)}


@app.get("/api/config")
def get_config():
    return config.redact(config.load())


def _index_settings(cfg):
    """What the stored vectors depend on: (where they are, which embedding model made them). Credentials and
    provider endpoints are left out: rotating a password or moving to a proxy doesn't change the vectors."""
    v, e = cfg["vector_store"], cfg["embeddings"]
    t = v["type"]
    if t == "pgvector":
        u = urlsplit(v["pgvector"]["dsn"])
        place = (u.hostname, u.port, u.path)
    else:
        place = v[t].get("url") or v[t].get("path")
    return (t, place, v["collection"]), (e["provider"], e[e["provider"]].get("model"))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def _collection_for(collection: str, old_model: str, new_model: str) -> str:
    """A new collection name for vectors from new_model, replacing a suffix this function added for old_model."""
    stem = f"{collection.removesuffix('_' + _slug(old_model)) or collection}_{_slug(new_model)}"
    name, n = stem[:63].strip("_"), 2
    while name == collection:  # e.g. the same model name from another provider
        name, n = f"{stem[:60].strip('_')}_{n}", n + 1
    return name


@app.put("/api/config")
def put_config(body: dict):
    """Save the wizard settings. If they change where vectors live or which model makes them while documents are
    indexed, answer 409 unless the body says what to do: "reindex": true to re-index everything after saving, or
    false to save anyway. A different embedding model always needs a new collection, as vectors can't be mixed."""
    reindex = body.pop("reindex", None)
    if reindex is not None and not isinstance(reindex, bool):
        raise HTTPException(400, '"reindex" must be true or false')
    old = config.load()
    new = _draft_config(body)
    indexed = db.rows("SELECT COUNT(*) n FROM documents WHERE status='ready'")[0]["n"]
    (old_where, old_model), (new_where, new_model) = _index_settings(old), _index_settings(new)
    if old["configured"] and indexed and (old_where, old_model) != (new_where, new_model):
        mixing = old_model != new_model and old_where == new_where
        if mixing or reindex is None:
            model = lambda m: m[1] or m[0]
            raise HTTPException(409, {
                "code": "reindex_required", "documents": indexed,
                "collection": _collection_for(new_where[2], model(old_model), model(new_model)) if mixing else None,
                "message": (f"Vectors from {model(old_model)} can't share a collection with {model(new_model)}. "
                            f"Save into a new collection and re-index the {indexed} documents."
                            if mixing else f"The {indexed} indexed documents were stored with the old settings, "
                            "so search won't find them until they're re-indexed.")})
    new["configured"] = True
    config.save(new)
    db.log("info", "setup", f"config saved: store={new['vector_store']['type']} "
           f"embeddings={new['embeddings']['provider']} llm={new['llm']['provider']}")
    if reindex:
        # Queued ones will use the new settings anyway; cancelled ones stay cancelled. Running ones restart.
        for d in db.rows("SELECT id FROM documents WHERE status IN ('ready','failed','processing')"):
            rag.enqueue(d["id"])
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
    return _check(component, _draft_config(body))


@app.post("/api/models/{component}")
def list_models(component: str, body: dict | None = None):
    """Models the (possibly unsaved) embeddings or LLM provider offers, for the wizard's model picker."""
    factory = {"embeddings": embeddings.get, "llm": llm.get}.get(component)
    if not factory:
        raise HTTPException(404, "unknown component")
    cfg = _draft_config(body)
    try:
        return {"ok": True, "models": sorted(set(factory(cfg).models()))}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------- documents ----------

@app.get("/api/documents")
def list_documents():
    docs = db.rows("SELECT * FROM documents ORDER BY created DESC")
    queued = {d["id"] for d in docs if d["status"] == "queued"}
    position = {i: n for n, i in enumerate(dict.fromkeys(i for i in rag.pending() if i in queued), start=1)}
    prog = rag.progress()
    for d in docs:
        if d["id"] in position:
            d["queue_position"] = position[d["id"]]
        if d["status"] == "processing" and prog.get("doc_id") == d["id"]:
            d["progress"] = {k: prog[k] for k in ("stage", "done", "total")}
    return docs


class Skip(Exception):
    """A URL that isn't imported; the message is the reason shown to the caller, so it must not leak details."""


def _register(filename: str, tmp: Path, size: int, digest: str | None = None) -> dict:
    """Move a downloaded/uploaded temp file into storage and queue it, skipping duplicates. Pass the sha256 if
    it was computed while writing the file."""
    if digest is None:
        with tmp.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
    with tmp.open("rb") as f:
        is_pdf = f.read(5) == b"%PDF-"
    # Check and insert together: parallel URL downloads and uploads of the same file must queue it once.
    with _register_lock:
        dup = db.rows("SELECT id, filename FROM documents WHERE sha256=?", (digest,))
        if dup or not is_pdf:
            tmp.unlink(missing_ok=True)
            return {"file": filename, "status": "skipped",
                    "reason": f"duplicate of {dup[0]['filename']}" if dup else "not a PDF"}
        doc_id = uuid.uuid4().hex
        tmp.replace(config.UPLOAD_DIR / f"{doc_id}.pdf")
        now = time.time()
        db.execute("INSERT INTO documents (id, filename, size, status, created, updated, sha256) "
                   "VALUES (?,?,?,?,?,?,?)", (doc_id, filename, size, "queued", now, now, digest))
    rag.enqueue(doc_id)
    return {"file": filename, "status": "queued", "id": doc_id}


def _require_setup():
    if not config.load()["configured"]:
        raise HTTPException(400, "Finish setup first")


def _clean_name(name: str) -> str:
    # Keep folder structure from folder uploads (e.g. "contracts/2024/a.pdf"), drop anything unsafe.
    parts = [p for p in Path(name or "upload.pdf").parts if p not in ("", ".", "..", "/", "\\")]
    return "/".join(parts)[-255:] or "upload.pdf"


def _save_upload(name: str, src) -> dict:
    """Copy one parsed upload into storage, hashing as it goes. Blocking: run it off the event loop."""
    tmp = config.UPLOAD_DIR / f".tmp-{uuid.uuid4().hex}"
    size, sha = 0, hashlib.sha256()
    with tmp.open("wb") as out:
        while chunk := src.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                break
            sha.update(chunk)
            out.write(chunk)
    if size > MAX_UPLOAD_MB << 20:
        tmp.unlink(missing_ok=True)
        return {"file": name, "status": "skipped", "reason": f"larger than {MAX_UPLOAD_MB} MB"}
    return _register(name, tmp, size, sha.hexdigest())


@app.post("/api/documents")
async def upload(request: Request):
    """Upload one or many PDFs (multipart "files" fields). Non-PDFs, oversized files, and duplicates are skipped
    and reported."""
    _require_setup()
    results = []
    # Parsed here rather than by FastAPI so the file count is capped while parsing (400 past it), before more parts
    # are spooled.
    async with request.form(max_files=MAX_FILES_PER_UPLOAD, max_fields=10) as form:
        for f in form.getlist("files"):
            if isinstance(f, str):
                continue
            name = _clean_name(f.filename)
            if not name.lower().endswith(".pdf"):
                results.append({"file": name, "status": "skipped", "reason": "not a .pdf file"})
                continue
            # The body is already parsed into f.file, so copy it synchronously in a worker thread.
            results.append(await run_in_threadpool(_save_upload, name, f.file))
    return {"results": results, "queued": sum(r["status"] == "queued" for r in results)}


class UrlList(BaseModel):
    urls: list[str]


def _blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    ip = getattr(ip, "ipv4_mapped", None) or ip  # ::ffff:127.0.0.1 is loopback too
    return (ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast
            or ip.is_unspecified or not ip.is_global)


def _resolve(host: str) -> str:
    """The address to connect to for host, or Skip if any of its addresses is non-public (unless allowed)."""
    try:
        addrs = [info[4][0].split("%")[0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)]
    except (OSError, UnicodeError):
        addrs = []
    # One reason for both, so a caller can't tell an internal host name that resolves from one that doesn't.
    if not addrs:
        db.log("warn", "import", f"download refused: {host[:255]} does not resolve")
        raise Skip("download failed")
    if not ALLOW_PRIVATE_URLS and any(_blocked(ipaddress.ip_address(a)) for a in addrs):
        db.log("warn", "import", f"download refused: {host[:255]} resolves to a non-public address")
        raise Skip("download failed")
    return addrs[0]


def _fetch(url: httpx.URL, tmp: Path) -> tuple[int, str]:
    """Download url into tmp, following redirects by hand so every hop is checked. Returns (size, sha256)."""
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_S
    sockets, expired = [], threading.Event()

    def on_connect(event, info):
        if event.endswith("connect_tcp.complete"):
            sockets.append(info["return_value"].get_extra_info("socket"))
            if expired.is_set():  # connected just as the deadline passed
                expire()

    def expire():
        # Read timeouts are per read, so a server dripping bytes would never trip them: cut the socket instead.
        expired.set()
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    timer = threading.Timer(DOWNLOAD_TIMEOUT_S, expire)
    timer.daemon = True
    timer.start()
    try:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            for _ in range(MAX_REDIRECTS + 1):
                if url.scheme not in ("http", "https"):
                    raise Skip("only http(s) URLs are supported")
                # Connect to the address that was checked (no second lookup a DNS rebind could change), still
                # sending the real host name for virtual hosting and TLS certificate checks.
                target = url.copy_with(host=_resolve(url.host))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Skip("took too long")
                with client.stream("GET", target, headers={"Host": url.netloc.decode("ascii")},
                                   timeout=min(30, remaining),
                                   extensions={"sni_hostname": url.host, "trace": on_connect}) as r:
                    if r.is_redirect:
                        url = url.join(r.headers["location"])
                        continue
                    if not r.is_success:
                        raise Skip(f"HTTP {r.status_code}")
                    size, sha = 0, hashlib.sha256()
                    with tmp.open("wb") as out:
                        for chunk in r.iter_bytes():
                            size += len(chunk)
                            if size > MAX_UPLOAD_MB << 20:
                                raise Skip(f"larger than {MAX_UPLOAD_MB} MB")
                            if expired.is_set():
                                raise Skip("took too long")
                            sha.update(chunk)
                            out.write(chunk)
                    return size, sha.hexdigest()
        raise Skip("too many redirects")
    except httpx.HTTPError:
        if expired.is_set():
            raise Skip("took too long")
        raise
    finally:
        timer.cancel()


def _download(url: str) -> dict:
    name = "download.pdf"
    tmp = config.UPLOAD_DIR / f".tmp-{uuid.uuid4().hex}"
    try:
        try:
            parsed = httpx.URL(url)
        except httpx.InvalidURL:
            raise Skip("invalid URL")
        name = _clean_name(Path(parsed.path).name or name)
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        size, digest = _fetch(parsed, tmp)
        res = _register(name, tmp, size, digest)
    except Skip as e:
        res = {"file": name, "status": "skipped", "reason": str(e)}
    except Exception as e:
        # Details stay in the event log: they would tell an API caller which hosts and ports answer.
        db.log("warn", "import", f"download of {url[:500]} failed: {type(e).__name__}: {e}")
        res = {"file": name, "status": "skipped", "reason": "download failed"}
    tmp.unlink(missing_ok=True)
    res["url"] = url
    return res


@app.post("/api/documents/urls")
def import_urls(body: UrlList):
    """Download and ingest a list of PDF URLs."""
    _require_setup()
    urls = [u.strip() for u in body.urls if u.strip()]
    if len(urls) > MAX_URLS:
        raise HTTPException(400, f"At most {MAX_URLS} URLs per import")
    results = list(_download_pool.map(_download, urls))
    queued = sum(r["status"] == "queued" for r in results)
    db.log("info", "import", f"URL import: {queued}/{len(urls)} queued")
    return {"results": results, "queued": queued}


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
    queued = sum(r["status"] == "queued" for r in results)
    db.log("info", "import", f"folder import: {queued}/{len(results)} queued")
    return {"folder": str(IMPORT_DIR), "results": results, "queued": queued}


@app.post("/api/documents/{doc_id}/reindex")
def reindex(doc_id: str):
    if not db.rows("SELECT id FROM documents WHERE id=?", (doc_id,)):
        raise HTTPException(404)
    rag.enqueue(doc_id)
    return {"ok": True}


@app.post("/api/documents/{doc_id}/cancel")
def cancel_ingest(doc_id: str):
    if not db.rows("SELECT id FROM documents WHERE id=?", (doc_id,)):
        raise HTTPException(404)
    if not rag.cancel(doc_id):
        raise HTTPException(409, "Not queued or processing")
    return {"ok": True}


@app.post("/api/documents/reindex-all")
def reindex_all():
    ids = [d["id"] for d in db.rows("SELECT id FROM documents")]
    for i in ids:
        rag.enqueue(i)
    return {"queued": len(ids)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    # Row first, like cancel(): a running ingest then sees it gone instead of finishing as "ready" after the
    # vectors were dropped.
    db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    try:
        stores.get(config.load()).delete_doc(doc_id)
    except Exception as e:
        db.log("warn", "delete", f"could not remove vectors for {doc_id}: {e}")
    (config.UPLOAD_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/connect")
def connect_info():
    """The address MCP clients and apps should use, for the Connect page (it falls back to the browser's)."""
    return {"public_url": PUBLIC_URL}


# ---------- chat ----------

class Question(BaseModel):
    question: str = Field(max_length=MAX_QUERY_CHARS)
    k: int | None = Field(None, ge=1, le=rag.MAX_TOP_K)


@app.post("/api/chat")
def chat(q: Question):
    try:
        return rag.answer(q.question, q.k)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


# ---------- admin / diagnostics ----------

@app.post("/api/admin/retrieve")
def debug_retrieve(q: Question):
    """Retrieval as Chat and the API do it, including the rewritten queries when that's on."""
    t0 = time.time()
    try:
        _, queries, hits = rag.search(q.question, q.k)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)
    return {"ms": round((time.time() - t0) * 1000), "queries": queries, "hits": hits}


@app.get("/api/admin/status")
def status():
    cfg = config.load()
    # Run checks concurrently with a cap so one slow component can't stall the page.
    names = ("vector_store", "embeddings")
    futures = {c: _check_pool.submit(_check, c, cfg) for c in names}
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
    return db.rows("SELECT id, name, prefix, scope, created, last_used FROM api_keys ORDER BY id DESC")


class KeyName(BaseModel):
    name: str = "default"
    scope: Literal["read", "write"] = "read"  # "write" may also upload and import documents


@app.post("/api/admin/keys")
def create_key(body: KeyName):
    key = "rag_" + secrets.token_urlsafe(32)
    db.execute("INSERT INTO api_keys (name, prefix, hash, created, scope) VALUES (?,?,?,?,?)",
               (body.name, key[:10], auth.secret_hash(key), time.time(), body.scope))
    db.log("info", "api", f"API key created: {body.name} ({body.scope})")
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
    question: str = Field(max_length=MAX_QUERY_CHARS)
    top_k: int | None = Field(None, ge=1, le=rag.MAX_TOP_K)


class RetrieveIn(BaseModel):
    query: str = Field(max_length=MAX_QUERY_CHARS)
    top_k: int | None = Field(None, ge=1, le=rag.MAX_TOP_K)
    rewrite: bool | None = Field(None, description="Also search with the query reworded and translated into the "
                                 "documents' languages by the server's LLM. Defaults to the server's setting.")


@app.post("/v1/query", operation_id="askDocuments")
def v1_query(q: QueryIn):
    """Retrieve relevant chunks and generate an answer with citations."""
    try:
        return rag.answer(q.question, q.top_k)
    except Exception as e:
        # API callers get no internals (hosts, provider errors); the admin sees them in the Admin event log.
        db.log("error", "api", f"query failed: {type(e).__name__}: {e}")
        return JSONResponse({"error": "Answer failed"}, status_code=502)


@app.post("/v1/retrieve", operation_id="searchDocuments")
def v1_retrieve(q: RetrieveIn):
    """Retrieval only: the top matching pages, for use with your own LLM, and the extra queries searched."""
    try:
        _, queries, hits = rag.search(q.query, q.top_k, q.rewrite)
        return {"results": hits, "queries": queries}
    except Exception as e:
        db.log("error", "api", f"retrieve failed: {type(e).__name__}: {e}")
        return JSONResponse({"error": "Search failed"}, status_code=502)


@app.get("/v1/documents", operation_id="listDocuments")
def v1_documents():
    return list_documents()


@app.post("/v1/documents", operation_id="uploadDocuments", include_in_schema=False)
async def v1_upload(request: Request):
    return await upload(request)


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

class _RevalidatedStatic(StaticFiles):
    """Make browsers revalidate (cheap 304s via ETag) so an upgrade never runs a stale app.js."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", _RevalidatedStatic(directory=Path(__file__).parent / "static", html=True), name="static")
