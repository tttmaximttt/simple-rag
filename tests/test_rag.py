import queue
import threading
import time
import uuid

import pytest

from app import db, main, pdftext, rag


class FakeStore:
    def __init__(self):
        self.vectors, self.deleted = {}, []

    def ensure(self, dim):
        pass

    def upsert(self, items):
        self.vectors.update({i: payload for i, _, payload in items})

    def delete_doc(self, doc_id):
        self.deleted.append(doc_id)
        self.vectors = {k: v for k, v in self.vectors.items() if v["doc_id"] != doc_id}


class FakeEmbeddings:
    concurrency = 1

    def __init__(self, on_embed=None):
        self.on_embed, self.calls, self._lock = on_embed, 0, threading.Lock()

    def embed(self, texts):
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.on_embed:
            self.on_embed(n)
        return [[0.1, 0.2] for _ in texts]


@pytest.fixture
def doc(monkeypatch):
    """A queued document whose 'PDF' has 40 pages of text, with fake embeddings and store."""
    doc_id = uuid.uuid4().hex
    now = time.time()
    db.execute("INSERT INTO documents (id, filename, size, status, created, updated) VALUES (?,?,?,?,?,?)",
               (doc_id, "manual.pdf", 1, "queued", now, now))
    monkeypatch.setattr(pdftext, "page_count", lambda path: 40)
    monkeypatch.setattr(pdftext, "extract", lambda path, a, b: [f"Page {_word(i)} text. " * 200 for i in range(a, b)])
    monkeypatch.setattr(rag, "EXTRACT_PROCS", 1)  # no worker processes in tests
    store, emb = FakeStore(), FakeEmbeddings()
    monkeypatch.setattr(rag.stores, "get", lambda cfg: store)
    monkeypatch.setattr(rag.embeddings, "get", lambda cfg: emb)
    yield doc_id, store, emb
    db.execute("DELETE FROM documents WHERE id=?", (doc_id,))


def _word(i):
    # Letters, not digits: page text that differs only in numbers counts as a running header.
    return chr(97 + i // 26) + chr(97 + i % 26)


def _status(doc_id):
    return db.rows("SELECT status, chunks FROM documents WHERE id=?", (doc_id,))[0]


def test_ingest_reports_progress_and_finishes(doc, monkeypatch):
    doc_id, store, emb = doc
    seen = []
    real_report = rag._report
    monkeypatch.setattr(rag, "_report", lambda *a: seen.append(a[1:]) or real_report(*a))
    rag.ingest(doc_id)
    row = _status(doc_id)
    assert row["status"] == "ready" and row["chunks"] == len(store.vectors) > rag.EMBED_BATCH
    assert seen[0] == ("reading", 0, 40) and ("reading", 40, 40) in seen
    assert seen[-1] == ("embedding", row["chunks"], row["chunks"])
    assert rag.progress() == {}


@pytest.mark.parametrize("concurrency", [1, 3])
def test_cancel_while_embedding_stops_and_removes_partial_vectors(doc, monkeypatch, concurrency):
    doc_id, store, emb = doc
    monkeypatch.setattr(rag, "EMBED_BATCH", 8)  # 15 batches
    emb.concurrency = concurrency
    emb.on_embed = lambda n: n == 2 and rag.cancel(doc_id)
    rag.ingest(doc_id)
    assert _status(doc_id)["status"] == "cancelled"
    # Only batches already in flight when it was cancelled may still run.
    assert emb.calls <= 2 + concurrency and store.vectors == {}
    assert rag.progress() == {}


def test_cancel_queued_reindex_removes_the_old_index(doc):
    doc_id, store, emb = doc
    rag.ingest(doc_id)
    assert store.vectors and _status(doc_id)["status"] == "ready"
    db.execute("UPDATE documents SET status='queued' WHERE id=?", (doc_id,))  # Re-index, not yet picked up
    assert rag.cancel(doc_id)
    assert store.vectors == {} and _status(doc_id) == {"status": "cancelled", "chunks": None}


def test_failure_removes_partial_vectors(doc, monkeypatch):
    doc_id, store, emb = doc
    monkeypatch.setattr(rag, "EMBED_BATCH", 8)

    def fail(n):
        if n == 3:
            raise RuntimeError("HTTP 429")
    emb.on_embed = fail
    rag.ingest(doc_id)
    assert _status(doc_id) == {"status": "failed", "chunks": None} and store.vectors == {}


def test_failure_does_not_override_a_reindex_requested_meanwhile(doc, monkeypatch):
    doc_id, store, emb = doc
    monkeypatch.setattr(rag, "_jobs", queue.Queue())

    def reindex_then_fail(n):
        rag.enqueue(doc_id)
        raise RuntimeError("HTTP 429")
    emb.on_embed = reindex_then_fail
    rag.ingest(doc_id)
    assert _status(doc_id)["status"] == "queued" and list(rag._jobs.queue) == [doc_id]


def test_requeue_while_running_restarts_instead_of_finishing(doc, monkeypatch):
    doc_id, store, emb = doc
    monkeypatch.setattr(rag, "_jobs", queue.Queue())  # keep the app's worker thread from picking it up
    emb.on_embed = lambda n: n == 1 and rag.enqueue(doc_id)
    rag.ingest(doc_id)
    assert _status(doc_id)["status"] == "queued" and store.vectors == {}
    assert list(rag._jobs.queue) == [doc_id]
    emb.on_embed = None
    rag.ingest(rag._jobs.get())
    assert _status(doc_id)["status"] == "ready" and store.vectors


def test_cancelled_queue_entry_is_skipped(doc):
    doc_id, store, emb = doc
    assert rag.cancel(doc_id)
    rag.ingest(doc_id)
    assert _status(doc_id)["status"] == "cancelled" and emb.calls == 0


def test_cancel_endpoint(logged_in, doc):
    doc_id, _, _ = doc
    assert logged_in.post("/api/documents/nope/cancel").status_code == 404
    assert logged_in.post(f"/api/documents/{doc_id}/cancel").status_code == 200
    assert logged_in.post(f"/api/documents/{doc_id}/cancel").status_code == 409


def test_documents_list_shows_queue_position_and_progress(logged_in, doc, monkeypatch):
    doc_id, _, _ = doc
    other = uuid.uuid4().hex
    db.execute("INSERT INTO documents (id, filename, size, status, created, updated) VALUES (?,?,?,?,?,?)",
               (other, "b.pdf", 1, "processing", time.time(), time.time()))
    try:
        monkeypatch.setattr(rag, "pending", lambda: ["gone", doc_id, doc_id])
        monkeypatch.setattr(rag, "_progress", {"doc_id": other, "stage": "embedding", "done": 64, "total": 900})
        docs = {d["id"]: d for d in logged_in.get("/api/documents").json()}
        assert docs[doc_id]["queue_position"] == 1 and "progress" not in docs[doc_id]
        assert docs[other]["progress"] == {"stage": "embedding", "done": 64, "total": 900}
    finally:
        db.execute("DELETE FROM documents WHERE id=?", (other,))


def test_pages_are_read_in_order_by_range(monkeypatch):
    calls = []
    monkeypatch.setattr(pdftext, "page_count", lambda path: 50)
    monkeypatch.setattr(pdftext, "extract", lambda path, a, b: calls.append((a, b)) or [str(i) for i in range(a, b)])
    monkeypatch.setattr(rag, "_report", lambda *a: None)
    monkeypatch.setattr(rag, "EXTRACT_PROCS", 1)
    assert rag._read_pages("d", "x.pdf") == [str(i) for i in range(50)]
    assert calls == [(0, 16), (16, 32), (32, 48), (48, 50)]


def _text_pdf(path, pages):
    """A minimal PDF with one line of text per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(pages))}] /Count {pages} >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i in range(pages):
        stream = f"BT /F1 12 Tf 72 720 Td (Page {i} text) Tj ET"
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out, offsets = b"%PDF-1.4\n", []
    for n, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def test_parallel_extraction_with_worker_processes(tmp_path, monkeypatch):
    pdf = tmp_path / "t.pdf"
    _text_pdf(pdf, 7)
    monkeypatch.setattr(rag, "_report", lambda *a: None)
    monkeypatch.setattr(rag, "EXTRACT_PROCS", 2)
    monkeypatch.setattr(rag, "PAGES_PER_TASK", 2)
    texts = rag._read_pages("d", str(pdf))
    assert [t.strip() for t in texts] == [f"Page {i} text" for i in range(7)]


def test_cancel_while_reading_removes_the_previous_index(doc, monkeypatch):
    doc_id, store, emb = doc
    store.vectors = {f"{doc_id}:0": {"doc_id": doc_id}}  # from an earlier ingest
    fake_extract = pdftext.extract
    monkeypatch.setattr(pdftext, "extract", lambda path, a, b: rag.cancel(doc_id) and fake_extract(path, a, b))
    rag.ingest(doc_id)
    assert _status(doc_id)["status"] == "cancelled" and store.vectors == {} and emb.calls == 0


def test_delete_during_ingest_leaves_no_orphan_vectors(logged_in, doc):
    doc_id, store, emb = doc
    emb.on_embed = lambda n: n == 1 and logged_in.delete(f"/api/documents/{doc_id}")
    rag.ingest(doc_id)
    assert not db.rows("SELECT id FROM documents WHERE id=?", (doc_id,))
    assert store.vectors == {}


def test_worker_survives_an_ingest_that_raises(monkeypatch):
    jobs, done = queue.Queue(), threading.Event()

    def ingest(doc_id):
        if doc_id == "bad":
            raise RuntimeError("boom")
        done.set()

    monkeypatch.setattr(rag, "ingest", ingest)
    t = threading.Thread(target=rag._worker, args=(jobs,), daemon=True)
    t.start()
    jobs.put("bad")
    jobs.put("good")
    assert done.wait(5) and t.is_alive()


def test_ingest_of_a_document_deleted_after_the_claim_just_returns(doc, monkeypatch):
    doc_id, store, emb = doc
    real = db.rows
    monkeypatch.setattr(db, "rows", lambda sql, *a: [] if "SELECT *" in sql else real(sql, *a))
    rag.ingest(doc_id)
    assert emb.calls == 0


def test_ingest_failing_before_the_document_is_read_is_marked_failed(doc, monkeypatch):
    doc_id, _, _ = doc
    monkeypatch.setattr(rag.config, "load", lambda: (_ for _ in ()).throw(OSError("disk")))
    rag.ingest(doc_id)
    assert db.rows("SELECT status, error FROM documents WHERE id=?", (doc_id,))[0]["status"] == "failed"


def test_a_page_that_cannot_be_read_is_empty(monkeypatch):
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            if self.text is None:
                raise ValueError("bad stream")
            return self.text

    class Reader:
        pages = [Page("one"), Page(None), Page("three")]

    monkeypatch.setattr(pdftext, "PdfReader", lambda path: Reader())
    assert pdftext.extract("x.pdf", 0, 3) == ["one", "", "three"]
