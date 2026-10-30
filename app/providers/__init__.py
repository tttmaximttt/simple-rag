import time

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504, 529}  # 529: Anthropic overloaded


def check(r: httpx.Response) -> httpx.Response:
    """Like raise_for_status, but keeps the provider's own error message, which is what says how to fix it.

    OpenAI and Anthropic send {"error": {"message": ...}}, Ollama {"error": "..."}."""
    if r.is_error:
        msg = None
        try:
            body = r.json()
            if isinstance(body, dict):
                err = body.get("error")
                msg = err.get("message") if isinstance(err, dict) else err or body.get("detail")
        except ValueError:
            pass
        msg = str(msg or r.text or r.reason_phrase)[:500]
        raise httpx.HTTPStatusError(f"HTTP {r.status_code} from {r.request.url.host}: {msg}",
                                    request=r.request, response=r)
    return r


def openai_model_ids(base_url: str, headers: dict) -> list[str]:
    r = check(httpx.get(f"{base_url}/models", headers=headers, timeout=30))
    return [m["id"] for m in r.json()["data"]]


def ollama_models(url: str) -> list[str]:
    r = check(httpx.get(f"{url}/api/tags", timeout=30))
    return [m["name"] for m in r.json()["models"]]


def post_with_retry(url, attempts=5, **kwargs) -> httpx.Response:
    """httpx.post that waits out rate limits and transient server errors, honouring Retry-After."""
    for attempt in range(attempts):
        r = httpx.post(url, **kwargs)
        if r.status_code not in RETRY_STATUSES or attempt == attempts - 1:
            return r
        try:
            wait = min(float(r.headers.get("retry-after", "")), 60)
        except ValueError:
            wait = 2 ** attempt
        time.sleep(wait)
