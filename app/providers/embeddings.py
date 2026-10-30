"""Embedding providers. Each exposes embed(texts) -> list[list[float]]."""
from functools import lru_cache

import httpx

from . import check, ollama_models, openai_model_ids, post_with_retry


class OpenAIEmbeddings:
    concurrency = 4  # embed() calls ingestion may run at once

    def __init__(self, api_key, model, base_url):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 96):
            r = post_with_retry(f"{self.base_url}/embeddings", headers=self.headers,
                                json={"model": self.model, "input": texts[i:i + 96]}, timeout=120)
            check(r)
            out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        return out

    def models(self):
        ids = openai_model_ids(self.base_url, self.headers)
        # OpenAI lists every model; other compatible servers may not name embedding models this way.
        return [i for i in ids if "embed" in i] or ids


class OllamaEmbeddings:
    def __init__(self, url, model):
        self.url, self.model = url.rstrip("/"), model

    def embed(self, texts):
        r = httpx.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts}, timeout=300)
        check(r)
        return r.json()["embeddings"]

    def models(self):
        return ollama_models(self.url)


@lru_cache(maxsize=2)
def _fastembed(model):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model)


class LocalEmbeddings:
    """Runs on CPU inside the container via fastembed (ONNX). Downloads the model on first use."""

    def __init__(self, model):
        self.model = model

    def embed(self, texts):
        return [v.tolist() for v in _fastembed(self.model).embed(texts)]

    def models(self):
        from fastembed import TextEmbedding
        return [m["model"] for m in TextEmbedding.list_supported_models()]


def get(cfg):
    e = cfg["embeddings"]
    p = e["provider"]
    if p == "openai":
        return OpenAIEmbeddings(**e["openai"])
    if p == "ollama":
        return OllamaEmbeddings(**e["ollama"])
    if p == "local":
        return LocalEmbeddings(**e["local"])
    raise ValueError(f"Unknown embeddings provider: {p}")
