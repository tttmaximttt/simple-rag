import chromadb


class ChromaStore:
    """Embedded Chroma, persisted on the app's data volume. No extra container needed."""

    def __init__(self, path, collection):
        self.client = chromadb.PersistentClient(path=path)
        self.name = collection

    def _coll(self):
        return self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def ping(self):
        self.client.heartbeat()

    def ensure(self, dim):
        self._coll()

    def upsert(self, items):
        self._coll().upsert(
            ids=[i for i, _, _ in items], embeddings=[v for _, v, _ in items],
            documents=[p["text"] for _, _, p in items],
            metadatas=[{k: p[k] for k in ("doc_id", "filename", "page", "chunk")} for _, _, p in items])

    def search(self, vector, k):
        r = self._coll().query(query_embeddings=[vector], n_results=k)
        return [{"score": 1 - d, "text": t, **m}
                for d, t, m in zip(r["distances"][0], r["documents"][0], r["metadatas"][0])]

    def delete_doc(self, doc_id):
        self._coll().delete(where={"doc_id": doc_id})

    def count(self):
        return self._coll().count()
