import chromadb

from . import bm25, dimension_mismatch

LOAD_BATCH = 5000


class ChromaStore:
    """Embedded Chroma, persisted on the app's data volume. No extra container needed."""

    def __init__(self, path, collection):
        self.client = chromadb.PersistentClient(path=path)
        self.name = collection
        self._key = ("chroma", path, collection)

    def _coll(self):
        return self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def ping(self):
        self.client.heartbeat()

    def ensure(self, dim):
        coll = self._coll()
        n = coll.count()
        if not n:
            # an emptied collection may stay locked to its old dimension; start fresh
            self.client.delete_collection(self.name)
            self._coll()
            return
        have = len(coll.get(limit=1, include=["embeddings"])["embeddings"][0])
        if have != dim:
            raise dimension_mismatch(self.name, n, have, dim)

    def upsert(self, items):
        metadatas = [{k: p[k] for k in ("doc_id", "filename", "page", "chunk")} for _, _, p in items]
        self._coll().upsert(
            ids=[i for i, _, _ in items], embeddings=[v for _, v, _ in items],
            documents=[p["text"] for _, _, p in items], metadatas=metadatas)
        bm25.added(self._key, [{"text": p["text"], **m} for (_, _, p), m in zip(items, metadatas)])

    def search(self, vector, k):
        r = self._coll().query(query_embeddings=[vector], n_results=k)
        return [{"score": 1 - d, "text": t, **m}
                for d, t, m in zip(r["distances"][0], r["documents"][0], r["metadatas"][0])]

    def keyword_search(self, query, k):
        coll = self._coll()
        return bm25.search(self._key, coll.count(), lambda: self._all_chunks(coll), query, k)

    def _all_chunks(self, coll):
        out = []
        while True:
            batch = coll.get(include=["documents", "metadatas"], limit=LOAD_BATCH, offset=len(out))
            out += [{"text": t, **m} for t, m in zip(batch["documents"], batch["metadatas"])]
            if len(batch["ids"]) < LOAD_BATCH:
                return out

    def keyword_missing(self, doc_ids=None):
        return 0  # the BM25 sidecar is built from the stored chunks

    def delete_doc(self, doc_id):
        self._coll().delete(where={"doc_id": doc_id})
        bm25.removed(self._key, doc_id)

    def count(self):
        return self._coll().count()
