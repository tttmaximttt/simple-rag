import base64
import hashlib
import os
import platform
import secrets
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, rag, stores, tokenizer
from .providers import embeddings, llm

STARTED = time.time()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
CHECK_TIMEOUT_S = 15
_pool = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="RAG Studio")


@app.on_event("startup")
def _startup():
    db.init()
    rag.start_worker()
    db.log("info", "app", "started")


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@app.middleware("http")
async def auth(request: Request, call_next):
    path = request.url.path
    # Public API: always requires an API key created on the Admin page.
    if path.startswith("/v1/"):
        header = request.headers.get("authorization", "")
        key = header[7:] if header.startswith("Bearer ") else request.headers.get("x-api-key", "")
        found = key and db.rows("SELECT id FROM api_keys WHERE hash=?", (_key_hash(key),))
        if not found:
            return JSONResponse({"error": "Missing or invalid API key"}, status_code=401)
        db.execute("UPDATE api_keys SET last_used=? WHERE id=?", (time.time(), found[0]["id"]))
        return await call_next(request)
    # UI and internal API: optional basic auth.
    if ADMIN_PASSWORD and path != "/api/health":
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                ok = secrets.compare_digest(pw, ADMIN_PASSWORD)
            except Exception:
                pass
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="RAG Studio"'})
    return await call_next(request)


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


@app.post("/api/documents")
async def upload(files: list[UploadFile]):
    if not config.load()["configured"]:
        raise HTTPException(400, "Finish setup first")
    created = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename}: only PDF files are supported")
        doc_id = uuid.uuid4().hex
        dest = config.UPLOAD_DIR / f"{doc_id}.pdf"
        size = 0
        with dest.open("wb") as out:
            while chunk := await f.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB << 20:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"{f.filename}: larger than {MAX_UPLOAD_MB} MB")
                out.write(chunk)
        now = time.time()
        db.execute("INSERT INTO documents (id, filename, size, status, created, updated) VALUES (?,?,?,?,?,?)",
                   (doc_id, Path(f.filename).name, size, "queued", now, now))
        rag.enqueue(doc_id)
        created.append(doc_id)
    return {"ids": created}


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


@app.post("/v1/query")
def v1_query(q: QueryIn):
    """Retrieve relevant chunks and generate an answer with citations."""
    try:
        return rag.answer(q.question, q.top_k)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.post("/v1/retrieve")
def v1_retrieve(q: RetrieveIn):
    """Retrieval only: the top matching chunks, for use with your own LLM."""
    try:
        return {"results": rag.retrieve(q.query, q.top_k)}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.get("/v1/documents")
def v1_documents():
    return list_documents()


@app.post("/v1/documents")
async def v1_upload(files: list[UploadFile]):
    return await upload(files)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
