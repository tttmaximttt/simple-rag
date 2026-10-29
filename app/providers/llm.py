"""Chat model providers. Each exposes complete(system, user) -> str."""
import httpx


class AnthropicLLM:
    def __init__(self, api_key, model, temperature):
        self.api_key, self.model, self.temperature = api_key, model, temperature

    def complete(self, system, user):
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=180, headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, json={
            "model": self.model, "max_tokens": 2048, "temperature": self.temperature,
            "system": system, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


class OpenAILLM:
    def __init__(self, api_key, model, base_url, temperature):
        self.model, self.base_url, self.temperature = model, base_url.rstrip("/"), temperature
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def complete(self, system, user):
        r = httpx.post(f"{self.base_url}/chat/completions", headers=self.headers, timeout=180, json={
            "model": self.model, "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class OllamaLLM:
    def __init__(self, url, model, temperature):
        self.url, self.model, self.temperature = url.rstrip("/"), model, temperature

    def complete(self, system, user):
        r = httpx.post(f"{self.url}/api/chat", timeout=600, json={
            "model": self.model, "stream": False, "options": {"temperature": self.temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["message"]["content"]


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
