"""Chat model providers. Each exposes complete(system, user) -> str."""
import httpx

from . import check, ollama_models, openai_model_ids, post_with_retry


class AnthropicLLM:
    def __init__(self, api_key, model, temperature):
        self.model, self.temperature = model, temperature
        self.headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

    def complete(self, system, user):
        body = {"model": self.model, "max_tokens": 2048, "system": system,
                "messages": [{"role": "user", "content": user}]}
        r = _post_with_temperature(("anthropic", "", self.model), body, self.temperature, self._post)
        return "".join(b.get("text", "") for b in r.json()["content"])

    def _post(self, body):
        return post_with_retry("https://api.anthropic.com/v1/messages", timeout=180, headers=self.headers, json=body)

    def models(self):
        r = check(httpx.get("https://api.anthropic.com/v1/models", params={"limit": 1000}, timeout=30,
                            headers=self.headers))
        return [m["id"] for m in r.json()["data"]]


class OpenAILLM:
    def __init__(self, api_key, model, base_url, temperature):
        self.model, self.base_url, self.temperature = model, base_url.rstrip("/"), temperature
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def complete(self, system, user):
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        r = _post_with_temperature(("openai", self.base_url, self.model), body, self.temperature, self._post)
        return r.json()["choices"][0]["message"]["content"]

    def _post(self, body):
        return post_with_retry(f"{self.base_url}/chat/completions", headers=self.headers, timeout=180, json=body)

    def models(self):
        # OpenAI's list includes models that /chat/completions can't serve: embedding, audio, image, video,
        # moderation, legacy completions, and the Responses-API-only pro and codex models.
        return [i for i in openai_model_ids(self.base_url, self.headers) if not any(x in i for x in _NOT_CHAT)]


# (provider, base_url, model) seen rejecting the temperature parameter: newer Claude models ("`temperature` is
# deprecated for this model") and OpenAI's reasoning models ("Unsupported value: 'temperature' ...").
_DEFAULT_TEMPERATURE_ONLY: set[tuple[str, str, str]] = set()
_REJECTS_TEMPERATURE = ("deprecated", "unsupported", "not supported", "does not support")


def _rejects_temperature(r: httpx.Response) -> bool:
    """A 400 saying the model takes no temperature at all, as opposed to one out of range, which is the user's
    to fix and must not be dropped silently."""
    text = r.text.lower()
    return r.status_code == 400 and "temperature" in text and any(w in text for w in _REJECTS_TEMPERATURE)


def _post_with_temperature(key, body, temperature, post) -> httpx.Response:
    """POST with the configured temperature; when the model rejects the parameter, remember that and resend
    without it."""
    if key not in _DEFAULT_TEMPERATURE_ONLY:
        r = post({**body, "temperature": temperature})
        if not _rejects_temperature(r):
            return check(r)
        _DEFAULT_TEMPERATURE_ONLY.add(key)
    return check(post(body))


_NOT_CHAT = ("embed", "whisper", "tts", "dall-e", "moderation", "davinci", "babbage", "transcribe",
             "image", "audio", "realtime", "search", "sora", "instruct", "codex", "-pro")


class OllamaLLM:
    def __init__(self, url, model, temperature):
        self.url, self.model, self.temperature = url.rstrip("/"), model, temperature

    def complete(self, system, user):
        r = post_with_retry(f"{self.url}/api/chat", timeout=600, json={
            "model": self.model, "stream": False, "options": {"temperature": self.temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        check(r)
        return r.json()["message"]["content"]

    def models(self):
        return ollama_models(self.url)


def get(cfg):
    l = cfg["llm"]
    p, t = l["provider"], l["temperature"]
    if p == "anthropic":
        return AnthropicLLM(**l["anthropic"], temperature=t)
    if p == "openai":
        return OpenAILLM(**l["openai"], temperature=t)
    if p == "ollama":
        return OllamaLLM(**l["ollama"], temperature=t)
    raise ValueError(f"Unknown LLM provider: {p}")
