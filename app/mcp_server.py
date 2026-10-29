"""MCP server (streamable HTTP at /mcp) so Claude, ChatGPT, and other MCP clients can
search the documents. Tool names and result shapes follow the search/fetch convention
ChatGPT connectors and deep research expect."""
from anyio import to_thread
from mcp.server.mcpserver import MCPServer
from pypdf import PdfReader

from . import config, db, rag

mcp = MCPServer(
    "RAG Studio",
    instructions="Search and cite the user's uploaded PDF documents. Use `search` to find relevant "
                 "passages, `fetch` to read a full page, and `ask` for a generated answer with sources.",
)


def _hit_id(h):
    return f"{h['doc_id']}:{h['page']}"


@mcp.tool()
async def search(query: str, top_k: int = 8) -> dict:
    """Semantic search over the uploaded documents. Returns matching passages with ids
    that can be passed to `fetch` to read the whole page."""
    hits = await to_thread.run_sync(rag.retrieve, query, top_k)
    return {"results": [{"id": _hit_id(h), "title": f"{h['filename']} (page {h['page']})",
                         "text": h["text"], "score": round(float(h["score"]), 4),
                         "url": f"rag://{_hit_id(h)}"} for h in hits]}


def _page_text(id_):
    doc_id, _, page = id_.rpartition(":")
    rows = db.rows("SELECT filename FROM documents WHERE id=?", (doc_id,))
    if not rows or not page.isdigit():
        raise ValueError(f"Unknown id: {id_}")
    reader = PdfReader(str(config.UPLOAD_DIR / f"{doc_id}.pdf"))
    p = int(page)
    if not 1 <= p <= len(reader.pages):
        raise ValueError(f"Page {p} out of range")
    return {"id": id_, "title": f"{rows[0]['filename']} (page {p})",
            "text": reader.pages[p - 1].extract_text() or "", "url": f"rag://{id_}",
            "metadata": {"filename": rows[0]["filename"], "page": p, "pages": len(reader.pages)}}


@mcp.tool()
async def fetch(id: str) -> dict:
    """Return the full text of one document page, by an id from `search` results."""
    return await to_thread.run_sync(_page_text, id)


@mcp.tool()
async def ask(question: str, top_k: int = 5) -> dict:
    """Answer a question from the documents using the server's configured LLM, with sources."""
    r = await to_thread.run_sync(rag.answer, question, top_k)
    return {"answer": r["answer"], "sources": [
        {"id": _hit_id(h), "filename": h["filename"], "page": h["page"]} for h in r["sources"]]}


@mcp.tool()
async def list_documents() -> dict:
    """List uploaded documents with their ingestion status."""
    docs = await to_thread.run_sync(lambda: db.rows(
        "SELECT id, filename, pages, chunks, status FROM documents ORDER BY created DESC"))
    return {"documents": docs}


# Not localhost-only: DNS rebinding protection is off because access is gated by API keys.
http_app = mcp.streamable_http_app(stateless_http=True, json_response=True, host="0.0.0.0")
