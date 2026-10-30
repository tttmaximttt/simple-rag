"""Chat model providers. Each exposes complete(system, user) -> str."""
import httpx

from . import check


class AnthropicLLM:
    def __init__(self, api_key, model, temperature):
        self.api_key, self.model, self.temperature = api_key, model, temperature

    def complete(self, system, user):
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=180, headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, json={
            "model": self.model, "max_tokens": 2048, "temperature": self.temperature,
            "system": system, "messages": [{"role": "user", "content": user}]})
        check(r)
        return "".join(b.get("text", "") for b in r.json()["content"])

    def models(self):
        r = httpx.get("https://api.anthropic.com/v1/models", params={"limit": 1000}, timeout=30, headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        check(r)
        return [m["id"] for m in r.json()["data"]]


class OpenAILLM:
    def __init__(self, api_key, model, base_url, temperature):
        self.model, self.base_url, self.temperature = model, base_url.rstrip("/"), temperature
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def complete(self, system, user):
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        key = (self.base_url, self.model)
        if key not in _DEFAULT_TEMPERATURE_ONLY:
            r = self._post({**body, "temperature": self.temperature})
            # Reasoning models (OpenAI's gpt-5 and o-series) reject any temperature but the default.
            if not (r.status_code == 400 and "temperature" in r.text):
                return check(r).json()["choices"][0]["message"]["content"]
            _DEFAULT_TEMPERATURE_ONLY.add(key)
        return check(self._post(body)).json()["choices"][0]["message"]["content"]

    def _post(self, body):
        return httpx.post(f"{self.base_url}/chat/completions", headers=self.headers, timeout=180, json=body)

    def models(self):
        r = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=30)
        check(r)
        # OpenAI's list includes models that /chat/completions can't serve: embedding, audio, image, video,
        # moderation, legacy completions, and the Responses-API-only pro and codex models.
        return [m["id"] for m in r.json()["data"] if not any(x in m["id"] for x in _NOT_CHAT)]


_DEFAULT_TEMPERATURE_ONLY: set[tuple[str, str]] = set()  # (base_url, model) seen rejecting a temperature
_NOT_CHAT = ("embed", "whisper", "tts", "dall-e", "moderation", "davinci", "babbage", "transcribe",
             "image", "audio", "realtime", "search", "sora", "instruct", "codex", "-pro")


class OllamaLLM:
    def __init__(self, url, model, temperature):
        self.url, self.model, self.temperature = url.rstrip("/"), model, temperature

    def complete(self, system, user):
        r = httpx.post(f"{self.url}/api/chat", timeout=600, json={
            "model": self.model, "stream": False, "options": {"temperature": self.temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        check(r)
        return r.json()["message"]["content"]

    def models(self):
        r = httpx.get(f"{self.url}/api/tags", timeout=30)
        check(r)
        return [m["name"] for m in r.json()["models"]]


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
