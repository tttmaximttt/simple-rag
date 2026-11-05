import uuid

from qdrant_client import QdrantClient, models

from . import bm25, dimension_mismatch

LOAD_BATCH = 1000


class QdrantStore:
    def __init__(self, url, api_key, collection):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self.collection = collection
        self._key = ("qdrant", url, collection)

    def ping(self):
        self.client.get_collections()

    def ensure(self, dim):
        if self.client.collection_exists(self.collection):
            have = self.client.get_collection(self.collection).config.params.vectors.size
            if have == dim:
                return
            n = self.client.count(self.collection).count
            if n:
                raise dimension_mismatch(self.collection, n, have, dim)
            self.client.delete_collection(self.collection)
        self.client.create_collection(self.collection, vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE))
        self.client.create_payload_index(self.collection, "doc_id", models.PayloadSchemaType.KEYWORD)

    def upsert(self, items):
        self.client.upsert(self.collection, points=[
            models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, i)), vector=v, payload=p)
            for i, v, p in items])
        bm25.changed(self._key)

    def search(self, vector, k):
        if not self.client.collection_exists(self.collection):
            return []
        res = self.client.query_points(self.collection, query=vector, limit=k, with_payload=True).points
        return [{"score": r.score, **r.payload} for r in res]

    def delete_doc(self, doc_id):
        if self.client.collection_exists(self.collection):
            self.client.delete(self.collection, points_selector=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]))
            bm25.changed(self._key)

    def keyword_search(self, query, k):
        return bm25.search(self._key, self.count(), self._all_chunks, query, k)

    def _all_chunks(self):
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(self.collection, limit=LOAD_BATCH, offset=offset,
                                                with_payload=True, with_vectors=False)
            out += [p.payload for p in points]
            if offset is None:
                return out

    def keyword_missing(self, doc_ids=None):
        return 0  # the BM25 sidecar is built from the stored chunks

    def count(self):
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count
