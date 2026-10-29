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
4. **Retrieval.** Set the chunk size and overlap in tokens, and top-K. Chunks are capped at the embedding model's input limit, and the wizard shows the limit.

### Tokenization

Text is chunked by tokens, not characters. Each page is split into sentences, and whole sentences are packed into chunks up to the size limit. A sentence that is longer than the limit is split on token boundaries. The last sentences of each chunk are carried into the next one as overlap. Chunks never cross pages, so each citation points to one page.

Token counts come from tiktoken `cl100k_base`, which is baked into the image. These counts are exact for OpenAI embedding models. Most open models (bge, nomic, and others) use their own tokenizers, which produce more tokens for the same text, so for those models the chunk size is capped at 75% of the model's limit. If tiktoken can't load, the app falls back to an approximate count. The Admin page shows which tokenizer is active.

Press **Test** on each step, then **Save**. Then go to **Documents** to add PDFs. Once a file shows `ready`, use **Chat**.

### Adding many PDFs at once

- **Files or folders:** drag many files or whole folders onto the Documents page, or use **Choose files** or **Choose folder**. Subfolders are included, and anything that isn't a PDF is ignored. Each file gets its own progress bar and result.
- **A list of URLs:** under **More ways to add documents**, paste PDF links, one per line. You can also call `POST /v1/documents/urls` with `{"urls": [...]}`.
- **A server folder:** copy PDFs into `./import` next to `docker-compose.yml` (mounted read-only at `/import`), then click **Import from server folder**. You can run it again any time, and only new files are added.
- **The API:** send several `-F files=@...` fields in one `POST /v1/documents` request.

Duplicates are detected by content hash and skipped. So are files that aren't real PDFs and files over `MAX_UPLOAD_MB`. Each one is reported with the reason.

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

## Using it from your apps (HTTP API)

Once PDFs are ingested, any app or script can query the RAG at `http://<host>:8080/v1`. Create an API key under **Admin → API keys** and send it as `Authorization: Bearer <key>` (or as an `X-API-Key` header). The `/v1` endpoints always require a key, whether or not `ADMIN_PASSWORD` is set.

**Ask a question** and get an answer with sources:

```bash
curl -s http://localhost:8080/v1/query \
  -H "Authorization: Bearer $RAG_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What does the contract say about termination?", "top_k": 5}'
```

```json
{
  "answer": "The contract can be terminated with 30 days notice [1] ...",
  "sources": [{"filename": "contract.pdf", "page": 4, "score": 0.83, "text": "...", "doc_id": "...", "chunk": 12}],
  "timing_ms": {"retrieve": 42, "generate": 1310}
}
```

**Retrieval only**, if you want to use the chunks with your own LLM or agent:

```bash
curl -s http://localhost:8080/v1/retrieve -H "Authorization: Bearer $RAG_KEY" \
  -H "Content-Type: application/json" -d '{"query": "termination clause", "top_k": 8}'
```

**Upload PDFs** and **list documents**:

```bash
curl -s http://localhost:8080/v1/documents -H "Authorization: Bearer $RAG_KEY" -F files=@report.pdf -F files=@manual.pdf
curl -s http://localhost:8080/v1/documents -H "Authorization: Bearer $RAG_KEY"
```

FastAPI serves interactive API docs at `/docs`.

## Connecting Claude and ChatGPT

RAG Studio exposes an **MCP server** at `/mcp` (streamable HTTP) with four tools:

- `search(query, top_k)` returns matching passages, each with an id.
- `fetch(id)` returns the full text of one page.
- `ask(question)` returns an answer generated by the LLM configured here, with sources.
- `list_documents()` lists the uploaded documents.

The `search` and `fetch` names and result shapes follow the convention that ChatGPT connectors and deep research expect.

Authentication uses an API key from **Admin → API keys**. Send it as `Authorization: Bearer <key>`. Clients that only take a URL can append `?key=<key>` instead. The key then shows up in the URLs of proxy logs, so use a dedicated key that you can revoke.

> Hosted clients (claude.ai, ChatGPT, the Claude API, and the OpenAI API) connect from the provider's cloud. They need a **public HTTPS URL**, for example `https://rag.example.com`. On a laptop, use a tunnel such as `cloudflared tunnel --url http://localhost:8080` or ngrok. Claude Code and the `mcp-remote` bridge for Claude Desktop also work with `localhost`.

### Claude Code

```bash
claude mcp add --transport http rag-studio http://localhost:8080/mcp \
  --header "Authorization: Bearer $RAG_KEY"
```

### Claude Desktop

**Option 1:** open Settings → Connectors → Add custom connector, and enter `https://rag.example.com/mcp?key=<key>`.

**Option 2:** for a local server, add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-studio": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8080/mcp", "--header", "Authorization: Bearer ${RAG_KEY}"],
      "env": { "RAG_KEY": "rag_..." }
    }
  }
}
```

### claude.ai (web and mobile)

Open Settings → Connectors → Add custom connector, and enter `https://rag.example.com/mcp?key=<key>`. Then turn the connector on in a chat from the tools menu.

### Claude API (MCP connector)

```bash
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: mcp-client-2025-11-20" -H "content-type: application/json" \
  -d '{
    "model": "claude-opus-5-5", "max_tokens": 4096,
    "mcp_servers": [{"type": "url", "url": "https://rag.example.com/mcp", "name": "rag-studio",
                     "authorization_token": "'"$RAG_KEY"'"}],
    "tools": [{"type": "mcp_toolset", "mcp_server_name": "rag-studio"}],
    "messages": [{"role": "user", "content": "Search my documents for the termination clause and summarize it."}]
  }'
```

### ChatGPT: custom GPT with Actions

1. Open Explore GPTs → Create → Configure → Create new action.
2. Choose **Import from URL** and enter `https://rag.example.com/v1/openapi.json`. That spec covers only the `/v1` API and includes the server URL. Set `PUBLIC_URL` so the URL in the spec is your HTTPS address.
3. Under **Authentication**, choose API Key with auth type **Bearer**, and paste a RAG Studio key.
4. In the instructions, write something like: "Use searchDocuments or askDocuments to answer questions from my documents, and cite the filename and page."

### ChatGPT: connector (MCP)

In ChatGPT, turn on developer mode under Settings → Connectors → Advanced. Then create a connector with the URL `https://rag.example.com/mcp?key=<key>` and no authentication.

### OpenAI API (Responses API, remote MCP)

```bash
curl https://api.openai.com/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "tools": [{"type": "mcp", "server_label": "rag-studio", "server_url": "https://rag.example.com/mcp",
               "headers": {"Authorization": "Bearer '"$RAG_KEY"'"}, "require_approval": "never"}],
    "input": "Search my documents for the termination clause and summarize it."
  }'
```

From any other app or agent framework, you can call the plain HTTP API above (`/v1/retrieve` and `/v1/query`) directly.

## Admin & diagnostics

The **Admin** page shows:

- Health and latency for the vector store, embeddings, and LLM (the LLM check is on demand because each check is one API call).
- Document counts by status, the ingestion queue, the vector count, and disk space.
- Query stats for the last 24 hours: count, average latency, and errors. A log of recent queries splits each one into retrieval and generation time.
- A **retrieval debugger** that runs a query against the vector store and shows the chunks and scores, without calling the LLM.
- An event log of ingestion results, errors, and config changes.
- API keys: create them, revoke them, and see when each was last used.
- **Re-index all**, for after you change the chunking or the embedding model.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | Host port |
| `ADMIN_PASSWORD` | empty | Turns on basic auth for the UI when set |
| `PUBLIC_URL` | empty | Public HTTPS base URL, used as the server in `/v1/openapi.json` |
| `MAX_UPLOAD_MB` | `100` | Max size per PDF |
| `IMPORT_DIR` | `/import` | Folder scanned by "Import from server folder" |
| `DATA_DIR` | `/data` | Config, uploads, metadata DB, embedded Chroma |

## Layout

```
app/main.py          FastAPI routes (setup, documents, chat, admin)
app/mcp_server.py    MCP server (/mcp) for Claude, ChatGPT, and other MCP clients
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
