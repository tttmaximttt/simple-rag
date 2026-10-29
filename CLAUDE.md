# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

RAG Studio: a self-hosted FastAPI server that ingests PDFs into a vector store and answers questions over them, with a no-build web UI, a key-authenticated HTTP API (`/v1`) and an MCP server (`/mcp`). README.md is the user-facing reference for setup, deployment behind a proxy, and client integrations.

## Commands

Python 3.12 is required (the Dockerfile uses `python:3.12-slim`); the pinned `fastapi` does not install on 3.9.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt          # includes requirements.txt + pytest
DATA_DIR=./data uvicorn app.main:app --reload --port 8080

python -m pytest -q                                            # all tests
python -m pytest tests/test_auth.py::test_name -q              # one test
docker compose up -d --build                                   # full stack (app + pgvector)
docker compose exec app python -m app.auth reset-admin         # new temporary admin password
```

There is no linter or formatter configured.

## Architecture

**Request gating lives in one middleware** (`authenticate` in `app/main.py`), keyed on path prefix:
- `/v1/*` and `/mcp`: API key (Bearer, `X-API-Key`, or `?key=`), checked against hashed keys in `api_keys`. Only `/v1/openapi.json` is public.
- `/api/*`: admin session cookie, except `PUBLIC_API_PATHS`. A user with `must_change` set can only reach the password and logout routes. Non-GET requests with an `Origin` header must pass `_same_origin` (CSRF defence on top of SameSite=Strict).
- Everything else is the static UI (`app/static`, mounted last as a catch-all). The MCP routes are appended to `app.router.routes` before that mount; order matters.

The `/v1` document routes delegate to their `/api` handlers; `/v1/query` and `/v1/retrieve` call `rag` directly. `/v1/openapi.json` is generated from the `/v1` routes only (minus those with `include_in_schema=False`), with `PUBLIC_URL` or the request's base URL as the server. FastAPI's default `/docs` and `/openapi.json` are disabled so internal routes aren't listed.

**Startup** (`lifespan`): `db.init()` (schema plus idempotent migrations and indexes), `auth.bootstrap()` (creates the first-run admin), `rag.start_worker()`, then the MCP session manager.

**Config** (`app/config.py`): wizard settings are a JSON file in `DATA_DIR`, deep-merged over `DEFAULTS`. `config.load()` is called per operation, so providers and stores are rebuilt from the current config each time rather than cached. Keys named `api_key`/`dsn` are masked by `redact()` before going to the browser, and `unredact()` restores them when the masked value comes back. `DATA_DIR` is read at import time.

**Pluggable backends**, chosen by config via a `get(cfg)` factory in each package:
- `app/stores/`: pgvector, Qdrant, Chroma. The interface contract (`ping`, `ensure`, `upsert`, `search`, `delete_doc`, `count`, and payload keys) is documented in `app/stores/__init__.py`.
- `app/providers/embeddings.py` (local fastembed, OpenAI-compatible, Ollama) and `app/providers/llm.py` (Anthropic, OpenAI-compatible, Ollama, each with `complete(system, user)`), called with raw `httpx` rather than vendor SDKs.

**Ingestion** (`app/rag.py`): uploads, URL imports and the import-folder scan all register a `documents` row (deduplicated by sha256) and enqueue it. A single daemon thread processes the queue one document at a time. Text is chunked per page by `app/tokenizer.py` (token-based, sentence-packed, never crossing pages; tiktoken `cl100k_base` with an approximate fallback) and upserted with ids `"{doc_id}:{chunk}"`. Documents left `queued`/`processing` are re-queued on restart. Changing chunking or the embedding model requires re-indexing.

**Metadata** (`app/db.py`): one SQLite file (`DATA_DIR/meta.sqlite`), with a new connection per call serialised by a global lock. Use `db.rows` / `db.execute` / `db.log`. `db.log` caps the `events` table separately for `auth` and all other sources (`KEEP_EVENTS` each). New columns or indexes go in `init()` as idempotent migrations.

**Auth** (`app/auth.py`): scrypt password hashes, session tokens stored hashed, a per-IP failed-login lockout kept in module-level dicts under a lock, and a capped failed-login audit log. The client IP comes from `_client_ip` in `main.py`, which relies on uvicorn `--proxy-headers` and `FORWARDED_ALLOW_IPS`.

## Tests

`tests/conftest.py` sets `DATA_DIR`, `IMPORT_DIR` and `ADMIN_PASSWORD` to temp values before importing the app, so tests never touch real data. The autouse `fresh_auth` fixture wipes the auth tables and resets the auth module's in-memory state before every test. If you add module-level state to `app/auth.py`, reset it there too. Use the `client` fixture for anonymous requests and `logged_in` for an admin session.
