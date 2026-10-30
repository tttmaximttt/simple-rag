import httpx
import pytest

from app import providers
from app.providers import check, llm

REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]}, request=REQ)


@pytest.mark.parametrize("body, expected", [
    ({"error": {"message": "Invalid API key"}}, "HTTP 401 from api.openai.com: Invalid API key"),
    ({"error": "model 'x' not found"}, "HTTP 401 from api.openai.com: model 'x' not found"),
    ({"detail": "nope"}, "HTTP 401 from api.openai.com: nope"),
    (["not", "an", "object"], r"HTTP 401 from api.openai.com: \["),
    ("Unauthorized", "HTTP 401 from api.openai.com: \"Unauthorized\""),
])
def test_check_keeps_the_provider_message(body, expected):
    with pytest.raises(httpx.HTTPStatusError, match=expected):
        check(httpx.Response(401, json=body, request=REQ))


def test_retries_without_temperature_once_per_model(monkeypatch):
    monkeypatch.setattr(llm, "_DEFAULT_TEMPERATURE_ONLY", set())
    sent = []

    def post(url, json, **kwargs):
        sent.append("temperature" in json)
        if "temperature" in json:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature'"}}, request=REQ)
        return _reply("ok")

    monkeypatch.setattr(llm.httpx, "post", post)
    model = llm.OpenAILLM("k", "gpt-5-mini", "https://api.openai.com/v1", 0.2)
    assert model.complete("s", "u") == "ok"
    assert model.complete("s", "u") == "ok"
    assert sent == [True, False, False]


def test_other_400s_are_not_retried(monkeypatch):
    monkeypatch.setattr(llm, "_DEFAULT_TEMPERATURE_ONLY", set())
    calls = []
    monkeypatch.setattr(llm.httpx, "post", lambda url, json, **kw: calls.append(1) or httpx.Response(
        400, json={"error": {"message": "context too long"}}, request=REQ))
    with pytest.raises(httpx.HTTPStatusError, match="context too long"):
        llm.OpenAILLM("k", "gpt-4o", "https://api.openai.com/v1", 0.2).complete("s", "u")
    assert calls == [1]


def test_post_with_retry_waits_out_rate_limits(monkeypatch):
    replies = [httpx.Response(429, headers={"retry-after": "0"}, request=REQ),
               httpx.Response(503, request=REQ), _reply("ok")]
    waits = []
    monkeypatch.setattr(providers.httpx, "post", lambda url, **kw: replies.pop(0))
    monkeypatch.setattr(providers.time, "sleep", waits.append)
    assert providers.post_with_retry("https://x/embeddings").status_code == 200
    assert waits == [0.0, 2]


def test_post_with_retry_gives_up(monkeypatch):
    monkeypatch.setattr(providers.httpx, "post", lambda url, **kw: httpx.Response(429, request=REQ))
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    assert providers.post_with_retry("https://x/embeddings", attempts=3).status_code == 429


@pytest.mark.parametrize("module, cls, args", [
    ("embeddings", "OllamaEmbeddings", ("http://ollama:11434/", "nomic-embed-text")),
    ("llm", "OllamaLLM", ("http://ollama:11434/", "llama3.1", 0.2)),
])
def test_ollama_models_come_from_tags(monkeypatch, module, cls, args):
    from app import providers as p
    seen = []
    monkeypatch.setattr(p.httpx, "get", lambda url, **kw: seen.append(url) or httpx.Response(
        200, json={"models": [{"name": "llama3.1"}, {"name": "nomic-embed-text"}]}, request=httpx.Request("GET", url)))
    provider = getattr(__import__(f"app.providers.{module}", fromlist=[cls]), cls)(*args)
    assert provider.models() == ["llama3.1", "nomic-embed-text"]
    assert seen == ["http://ollama:11434/api/tags"]
