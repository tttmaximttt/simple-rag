# RAG Studio

A self-hosted RAG (retrieval-augmented generation) server you can run with one command. A web UI walks you through setup, then you upload PDFs and ask questions about them. An admin page shows health and diagnostics.

## Quick start

```bash
git clone <this repo> rag-studio && cd rag-studio
docker compose up -d --build
```

Open http://localhost:8080. The **Setup** page opens first.

1. **Vector database.** The default is PostgreSQL + pgvector, which is bundled. Its connection string is pre-filled (`postgresql://rag:rag@postgres:5432/rag`). You can also pick Qdrant (`docker compose --profile qdrant up -d`, URL `http://qdrant:6333`) or embedded Chroma, which needs no server.
2. **Embeddings.** Choose Local (runs on CPU in the container, no key needed, downloads the model on first use), any OpenAI-compatible API, or Ollama.
3. **LLM.** Choose Anthropic Claude, any OpenAI-compatible API, or Ollama.
4. **Retrieval.** Set the chunk size, the overlap, and top-K.

Press **Test** on each step, then **Save**. Then go to **Documents** and drop PDFs in. Once a file shows `ready`, use **Chat**.

### Fully local, with no API keys

```bash
docker compose --profile ollama up -d --build
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull nomic-embed-text
```

Then pick Ollama for both embeddings and the LLM (URL `http://ollama:11434`).

## Running on a server

Set a password before you expose the app. This turns on HTTP basic auth for the whole UI and API (user `admin`):

```bash
ADMIN_PASSWORD='choose-something-long' PORT=8080 docker compose up -d --build
```

Put a TLS reverse proxy (Caddy, nginx, Traefik) in front of it. API keys are stored in `/data/config.json` on the `rag-data` volume with mode 600, and the UI never shows them back.

## Admin & diagnostics

The **Admin** page shows:

- Health and latency for the vector store, embeddings, and LLM (the LLM check is on demand because each check is one API call).
- Document counts by status, the ingestion queue, the vector count, and disk space.
- Query stats for the last 24 hours: count, average latency, and errors. A log of recent queries splits each one into retrieval and generation time.
- A **retrieval debugger** that runs a query against the vector store and shows the chunks and scores, without calling the LLM.
- An event log of ingestion results, errors, and config changes.
- **Re-index all**, for after you change the chunking or the embedding model.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | Host port |
| `ADMIN_PASSWORD` | empty | Turns on basic auth when set |
| `MAX_UPLOAD_MB` | `100` | Max size per PDF |
| `DATA_DIR` | `/data` | Config, uploads, metadata DB, embedded Chroma |

## Layout

```
app/main.py          FastAPI routes (setup, documents, chat, admin)
app/rag.py           PDF parsing, chunking, ingestion worker, answering
app/config.py        Settings persisted from the wizard
app/db.py            SQLite metadata: documents, queries, events
app/providers/       Embedding and LLM clients
app/stores/          pgvector, Qdrant, and Chroma backends
app/static/          Web UI (no build step)
```

## Limitations

- Only PDFs with a text layer are supported. Scanned PDFs need OCR, which is not built in yet.
- One ingestion worker processes files one at a time.
- If you change the embedding model, you need to re-index. Use a new collection name if the vector dimension changes.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
DATA_DIR=./data uvicorn app.main:app --reload --port 8080
```
