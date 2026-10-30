"""Fake OpenAI-compatible model server for the e2e tests. Standard library only.

  POST /v1/embeddings        hashed bag-of-words vectors, so passages that share words with the
                             query rank first and retrieval is deterministic
  POST /v1/chat/completions  "pong" for the health check; the question itself for a search-query
                             rewrite; otherwise an answer that quotes the first context passage, so
                             tests can check that retrieval fed the LLM
  GET  /files/<name>.pdf     a small PDF, for the URL import tests (404 for names starting "missing")
  GET  /health

Run: python e2e/fake_models.py [--port 9000]
"""
import argparse
import hashlib
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pdfs import make_pdf

DIM = 256


def embed(text: str) -> list[float]:
    v = [0.0] * DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
        v[h % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def chat(messages: list[dict]) -> str:
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if any(m["content"].startswith("Reply with the single word: pong") for m in messages):
        return "pong"
    if any(m["role"] == "system" and m["content"].startswith("You turn a question into search queries")
           for m in messages):
        return f"Question language: English\n{user}"  # rag.rewrite: search with the question as asked
    question = user.rpartition("Question:")[2].strip()
    # rag.answer formats context as "[1] (file, page N)\n<text>\n\n[2] ..."
    first = re.search(r"\[1\] \((.+?), page (\d+)\)\n(.*?)(?:\n\n\[2\]|\n\nQuestion:)", user, re.S)
    if not first:
        return f"I could not find it in the documents. (question: {question})"
    return f"According to {first[1]} page {first[2]} [1]: {first[3].strip()}"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        m = re.fullmatch(r"/files/([\w-]+)\.pdf", self.path)
        if m and not m[1].startswith("missing"):
            text = f"This document was imported from a URL. Its codename is {m[1]}. Kestrel bridges span the river."
            return self._send(200, make_pdf([text]), "application/pdf")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/v1/embeddings":
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self._send(200, {"object": "list", "model": body.get("model"), "data": [
                {"object": "embedding", "index": i, "embedding": embed(t)} for i, t in enumerate(texts)]})
        if self.path == "/v1/chat/completions":
            return self._send(200, {"object": "chat.completion", "model": body.get("model"), "choices": [
                {"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": chat(body["messages"])}}]})
        self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    a = p.parse_args()
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
