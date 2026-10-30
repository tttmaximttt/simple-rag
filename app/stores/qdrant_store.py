import uuid

from qdrant_client import QdrantClient, models

from . import dimension_mismatch


class QdrantStore:
    def __init__(self, url, api_key, collection):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self.collection = collection

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

    def search(self, vector, k):
        if not self.client.collection_exists(self.collection):
            return []
        res = self.client.query_points(self.collection, query=vector, limit=k, with_payload=True).points
        return [{"score": r.score, **r.payload} for r in res]

    def delete_doc(self, doc_id):
        if self.client.collection_exists(self.collection):
            self.client.delete(self.collection, points_selector=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]))

    def count(self):
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count
