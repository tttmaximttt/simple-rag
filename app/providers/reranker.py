"""Rerankers: a cross-encoder that scores (query, passage) pairs, used on the top candidates before the final
top_k. Each exposes rerank(query, texts) -> list[float], one relevance score per text (higher is better)."""
from functools import lru_cache

from . import check, post_with_retry


@lru_cache(maxsize=1)
def _cross_encoder(model):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=model)


class LocalReranker:
    """Runs on CPU via fastembed (ONNX), like the local embeddings. Downloads the model on first use."""

    def __init__(self, model):
        self.model = model

    def rerank(self, query, texts):
        return [float(s) for s in _cross_encoder(self.model).rerank(query, list(texts))]

    def models(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        return [m["model"] for m in TextCrossEncoder.list_supported_models()]


class APIReranker:
    """A rerank endpoint in the shape Cohere made common: POST {base_url}/rerank with {model, query, documents},
    answering {"results": [{"index", "relevance_score"}]}. Jina, Voyage, vLLM, Infinity and Cohere itself speak
    it; Hugging Face TEI's bare list of {"index", "score"} is accepted too."""

    def __init__(self, base_url, api_key, model):
        self.url, self.model = base_url.rstrip("/") + "/rerank", model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def rerank(self, query, texts):
        texts = list(texts)
        if not texts:
            return []
        body = {"query": query, "documents": texts, "top_n": len(texts)}
        if self.model:
            body["model"] = self.model
        data = check(post_with_retry(self.url, headers=self.headers, json=body, timeout=60)).json()
        results = data.get("results", data.get("data")) if isinstance(data, dict) else data
        scores = [float("-inf")] * len(texts)
        for r in results or []:
            scores[int(r["index"])] = float(r.get("relevance_score", r.get("score")))
        if float("-inf") in scores:
            raise ValueError(f"{self.url} scored {sum(s != float('-inf') for s in scores)} of {len(texts)} passages")
        return scores


def get(cfg):
    """The configured reranker, or None when it's off."""
    r = cfg["reranker"]
    p = r["provider"]
    if p == "off":
        return None
    if p == "local":
        return LocalReranker(**r["local"])
    if p == "api":
        return APIReranker(**r["api"])
    raise ValueError(f"Unknown reranker provider: {p}")
