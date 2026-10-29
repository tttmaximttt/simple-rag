"""Persistent settings chosen in the setup wizard, stored as JSON in DATA_DIR."""
import json
import os
import threading
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = DATA_DIR / "config.json"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULTS = {
    "configured": False,
    "vector_store": {
        "type": "pgvector",  # pgvector | qdrant | chroma
        "pgvector": {"dsn": "postgresql://rag:rag@postgres:5432/rag"},
        "qdrant": {"url": "http://qdrant:6333", "api_key": ""},
        "chroma": {"path": str(DATA_DIR / "chroma")},
        "collection": "documents",
    },
    "embeddings": {
        "provider": "local",  # local | openai | ollama
        "local": {"model": "BAAI/bge-small-en-v1.5"},
        "openai": {"api_key": "", "model": "text-embedding-3-small", "base_url": "https://api.openai.com/v1"},
        "ollama": {"url": "http://ollama:11434", "model": "nomic-embed-text"},
    },
    "llm": {
        "provider": "anthropic",  # anthropic | openai | ollama
        "anthropic": {"api_key": "", "model": "claude-sonnet-5-5"},
        "openai": {"api_key": "", "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
        "ollama": {"url": "http://ollama:11434", "model": "llama3.1"},
        "temperature": 0.2,
    },
    "rag": {"chunk_tokens": 400, "chunk_overlap_tokens": 60, "top_k": 5},
}

SECRET_KEYS = {"api_key", "dsn"}
_lock = threading.Lock()


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load() -> dict:
    if CONFIG_PATH.exists():
        return _merge(DEFAULTS, json.loads(CONFIG_PATH.read_text()))
    return _merge(DEFAULTS, {})


def save(cfg: dict) -> None:
    with _lock:
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(CONFIG_PATH)


def redact(cfg):
    """Copy of the config safe to send to the browser: secrets are masked."""
    if isinstance(cfg, dict):
        return {k: ("********" if k in SECRET_KEYS and v else redact(v)) for k, v in cfg.items()}
    return cfg


def unredact(new, old):
    """Keep stored secrets when the browser sends back the mask."""
    if isinstance(new, dict) and isinstance(old, dict):
        return {k: (old.get(k) if v == "********" else unredact(v, old.get(k))) for k, v in new.items()}
    return new
