"""PDF ingestion pipeline and retrieval-augmented answering."""
import queue
import threading
import time

from pypdf import PdfReader

from . import config, db, stores
from .providers import embeddings, llm

SYSTEM_PROMPT = (
    "You answer questions using only the provided context from the user's documents. "
    "Cite sources inline as [n] using the context numbers. If the context does not contain "
    "the answer, say you could not find it in the documents."
)

_jobs: "queue.Queue[str]" = queue.Queue()


def chunk_text(text, size, overlap):
    text = " ".join(text.split())
    if not text:
        return []
    out, i = [], 0
    while i < len(text):
        end = min(len(text), i + size)
        # Prefer to break at a sentence or word boundary.
        if end < len(text):
            cut = max(text.rfind(". ", i, end), text.rfind(" ", i, end))
            if cut > i + size // 2:
                end = cut + 1
        out.append(text[i:end].strip())
        if end >= len(text):
            break
        nxt = end - overlap if end - overlap > i else end
        # Start the next chunk on a word boundary.
        space = text.find(" ", nxt, end)
        i = space + 1 if space != -1 else nxt
    return [c for c in out if c]


def ingest(doc_id):
    cfg = config.load()
    doc = db.rows("SELECT * FROM documents WHERE id = ?", (doc_id,))[0]
    path = config.UPLOAD_DIR / f"{doc_id}.pdf"
    t0 = time.time()
    db.execute("UPDATE documents SET status='processing', error=NULL, updated=? WHERE id=?", (t0, doc_id))
    try:
        reader = PdfReader(str(path))
        r = cfg["rag"]
        pieces = []  # (page, chunk_index, text)
        for pno, page in enumerate(reader.pages, start=1):
            for c in chunk_text(page.extract_text() or "", r["chunk_size"], r["chunk_overlap"]):
                pieces.append((pno, len(pieces), c))
        if not pieces:
            raise ValueError("No extractable text (scanned PDF? OCR is not supported yet)")

        emb = embeddings.get(cfg)
        store = stores.get(cfg)
        store.delete_doc(doc_id)
        for b in range(0, len(pieces), 64):
            batch = pieces[b:b + 64]
            vectors = emb.embed([t for _, _, t in batch])
            if b == 0:
                store.ensure(len(vectors[0]))
            store.upsert([(f"{doc_id}:{ci}", v, {"doc_id": doc_id, "filename": doc["filename"],
                                                 "page": p, "chunk": ci, "text": t})
                          for (p, ci, t), v in zip(batch, vectors)])
        dur = time.time() - t0
        db.execute("UPDATE documents SET status='ready', pages=?, chunks=?, updated=?, duration=? WHERE id=?",
                   (len(reader.pages), len(pieces), time.time(), dur, doc_id))
        db.log("info", "ingest", f"{doc['filename']}: {len(pieces)} chunks from {len(reader.pages)} pages in {dur:.1f}s")
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        db.execute("UPDATE documents SET status='failed', error=?, updated=? WHERE id=?", (str(e), time.time(), doc_id))
        db.log("error", "ingest", f"{doc['filename']}: {e}")


def enqueue(doc_id):
    db.execute("UPDATE documents SET status='queued', updated=? WHERE id=?", (time.time(), doc_id))
    _jobs.put(doc_id)


def queue_size():
    return _jobs.qsize()


def _worker():
    while True:
        doc_id = _jobs.get()
        try:
            ingest(doc_id)
        finally:
            _jobs.task_done()


def start_worker():
    threading.Thread(target=_worker, daemon=True, name="ingest-worker").start()
    # Re-queue anything interrupted by a restart.
    for d in db.rows("SELECT id FROM documents WHERE status IN ('queued','processing')"):
        _jobs.put(d["id"])


def retrieve(question, k=None):
    cfg = config.load()
    vec = embeddings.get(cfg).embed([question])[0]
    return stores.get(cfg).search(vec, k or cfg["rag"]["top_k"])


def answer(question):
    t0 = time.time()
    try:
        hits = retrieve(question)
        t1 = time.time()
        context = "\n\n".join(f"[{n}] ({h['filename']}, page {h['page']})\n{h['text']}"
                              for n, h in enumerate(hits, start=1))
        reply = llm.get(config.load()).complete(
            SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
        t2 = time.time()
        db.execute("INSERT INTO queries (question, ts, retrieve_ms, generate_ms, total_ms, ok) VALUES (?,?,?,?,?,1)",
                   (question, t0, (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000))
        return {"answer": reply, "sources": hits, "timing_ms": {
            "retrieve": round((t1 - t0) * 1000), "generate": round((t2 - t1) * 1000)}}
    except Exception as e:
        db.execute("INSERT INTO queries (question, ts, total_ms, ok, error) VALUES (?,?,?,0,?)",
                   (question, t0, (time.time() - t0) * 1000, str(e)))
        db.log("error", "query", str(e))
        raise
