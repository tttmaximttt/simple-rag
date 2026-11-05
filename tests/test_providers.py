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


def test_anthropic_retries_without_temperature_once_per_model(monkeypatch):
    monkeypatch.setattr(llm, "_DEFAULT_TEMPERATURE_ONLY", set())
    sent = []
    areq = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def post(url, json, **kwargs):
        sent.append("temperature" in json)
        if "temperature" in json:
            return httpx.Response(400, json={"type": "error", "error": {
                "message": "`temperature` is deprecated for this model."}}, request=areq)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]}, request=areq)

    monkeypatch.setattr(llm.httpx, "post", post)
    model = llm.AnthropicLLM("k", "claude-sonnet-5-5", 0.2)
    assert model.complete("s", "u") == "ok"
    assert model.complete("s", "u") == "ok"
    assert sent == [True, False, False]


def test_anthropic_other_400s_are_not_retried(monkeypatch):
    monkeypatch.setattr(llm, "_DEFAULT_TEMPERATURE_ONLY", set())
    calls = []
    areq = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    monkeypatch.setattr(llm.httpx, "post", lambda url, json, **kw: calls.append(1) or httpx.Response(
        400, json={"error": {"message": "prompt is too long"}}, request=areq))
    with pytest.raises(httpx.HTTPStatusError, match="prompt is too long"):
        llm.AnthropicLLM("k", "claude-sonnet-5-5", 0.2).complete("s", "u")
    assert calls == [1]


@pytest.mark.parametrize("make, url, message", [
    (lambda: llm.AnthropicLLM("k", "claude-sonnet-5-5", 1.5), "https://api.anthropic.com/v1/messages",
     "temperature: range: 0..1"),
    (lambda: llm.OpenAILLM("k", "gpt-4o", "https://api.openai.com/v1", 3.0), "https://api.openai.com/v1",
     "Invalid 'temperature': decimal above maximum value. Expected a value <= 2, but got 3 instead."),
])
def test_out_of_range_temperature_is_reported_not_dropped(monkeypatch, make, url, message):
    monkeypatch.setattr(llm, "_DEFAULT_TEMPERATURE_ONLY", set())
    calls = []
    monkeypatch.setattr(llm.httpx, "post", lambda u, json, **kw: calls.append(1) or httpx.Response(
        400, json={"error": {"message": message}}, request=httpx.Request("POST", url)))
    with pytest.raises(httpx.HTTPStatusError, match="temperature"):
        make().complete("s", "u")
    assert calls == [1] and llm._DEFAULT_TEMPERATURE_ONLY == set()


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


def test_529_is_retried():
    assert 529 in providers.RETRY_STATUSES


def test_llm_providers_retry_transient_errors(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    anthropic_ok = httpx.Response(200, json={"content": [{"text": "ok"}]}, request=REQ)
    ollama_ok = httpx.Response(200, json={"message": {"content": "ok"}}, request=REQ)
    for model, ok in [(llm.AnthropicLLM("k", "m", 0.2), anthropic_ok), (llm.OllamaLLM("http://o", "m", 0.2), ollama_ok),
                      (llm.OpenAILLM("k", "gpt-4o", "https://x/v1", 0.2), _reply("ok"))]:
        replies = [httpx.Response(529, request=REQ), ok]
        monkeypatch.setattr(providers.httpx, "post", lambda url, **kw: replies.pop(0))
        monkeypatch.setattr(llm, "_DEFAULT_TEMPERATURE_ONLY", set())
        assert model.complete("s", "u") == "ok" and not replies


def test_ollama_embeddings_retry_transient_errors(monkeypatch):
    from app.providers import embeddings
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    replies = [httpx.Response(503, request=REQ), httpx.Response(200, json={"embeddings": [[1.0]]}, request=REQ)]
    monkeypatch.setattr(providers.httpx, "post", lambda url, **kw: replies.pop(0))
    assert embeddings.OllamaEmbeddings("http://o", "m").embed(["a"]) == [[1.0]]


def _anthropic_replies(monkeypatch, replies):
    monkeypatch.setattr(llm, "_DEFAULT_TEMPERATURE_ONLY", set())
    areq = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    sent = []
    monkeypatch.setattr(llm.httpx, "post", lambda url, json, **kw: sent.append(1) or httpx.Response(
        200, json=replies.pop(0), request=areq))
    return sent


def test_anthropic_asks_again_after_a_thinking_only_reply(monkeypatch):
    thinking_only = {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": ""}]}
    answer = {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": ""},
                                                     {"type": "text", "text": "ok"}]}
    sent = _anthropic_replies(monkeypatch, [thinking_only, answer])
    assert llm.AnthropicLLM("k", "claude-sonnet-5-5", 0.2).complete("s", "u") == "ok"
    assert len(sent) == 2


def test_anthropic_gives_up_on_replies_that_stay_empty(monkeypatch):
    thinking_only = {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": ""}]}
    sent = _anthropic_replies(monkeypatch, [thinking_only] * llm.EMPTY_REPLY_ATTEMPTS)
    with pytest.raises(RuntimeError, match="returned no text in 3 attempts"):
        llm.AnthropicLLM("k", "claude-sonnet-5-5", 0.2).complete("s", "u")
    assert len(sent) == llm.EMPTY_REPLY_ATTEMPTS


def test_anthropic_refusal_is_an_error(monkeypatch):
    refusal = {"stop_reason": "refusal", "stop_details": {"type": "refusal", "category": "cyber"}, "content": []}
    sent = _anthropic_replies(monkeypatch, [refusal])
    with pytest.raises(RuntimeError, match=r"declined the request \(cyber\)"):
        llm.AnthropicLLM("k", "claude-sonnet-5-5", 0.2).complete("s", "u")
    assert len(sent) == 1


def test_anthropic_thinking_that_uses_up_max_tokens_is_an_error(monkeypatch):
    out_of_tokens = {"stop_reason": "max_tokens", "content": [{"type": "thinking", "thinking": ""}]}
    sent = _anthropic_replies(monkeypatch, [out_of_tokens])
    with pytest.raises(RuntimeError, match="used all 2048 output tokens without answering"):
        llm.AnthropicLLM("k", "claude-sonnet-5-5", 0.2).complete("s", "u")
    assert len(sent) == 1  # asking again would only do the same
