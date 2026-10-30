"""Vector store backends.

Each store implements:
  ping() -> None               raise if unreachable
  ensure(dim) -> None          create the collection/table if missing
  upsert(items) -> None        items: list of (id, vector, payload)
  search(vector, k) -> list    [{"score": float, **payload}]
  delete_doc(doc_id) -> None
  count() -> int
Payload keys: doc_id, filename, page, chunk, text.
"""


def dimension_mismatch(collection, n, have, dim):
    """The error ensure(dim) raises when a collection that holds vectors was made with another dimension."""
    return ValueError(f"Collection {collection} holds {n} vectors of dimension {have}, "
                      f"but the embedding model makes {dim}. Use a new collection name and re-index.")


def get(cfg):
    v = cfg["vector_store"]
    t, coll = v["type"], v["collection"]
    if t == "pgvector":
        from .pgvector_store import PgVectorStore
        return PgVectorStore(v["pgvector"]["dsn"], coll)
    if t == "qdrant":
        from .qdrant_store import QdrantStore
        return QdrantStore(v["qdrant"]["url"], v["qdrant"].get("api_key") or None, coll)
    if t == "chroma":
        from .chroma_store import ChromaStore
        return ChromaStore(v["chroma"]["path"], coll)
    raise ValueError(f"Unknown vector store: {t}")
