"""The MCP server at /mcp, spoken to as a client would: JSON-RPC over streamable HTTP."""
import json

import httpx
import pytest

from conftest import ANSWER, BASE_URL, QUESTION

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class MCP:
    def __init__(self, **auth):
        self.client = httpx.Client(base_url=BASE_URL, timeout=120, **auth)
        self.n = 0

    def rpc(self, method, params=None) -> httpx.Response:
        self.n += 1
        return self.client.post("/mcp", headers=HEADERS,
                                json={"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}})

    def initialize(self) -> dict:
        r = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "e2e", "version": "1"}})
        assert r.status_code == 200, r.text
        self.client.post("/mcp", headers=HEADERS, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        return r.json()["result"]

    def call(self, tool, **arguments) -> dict:
        r = self.rpc("tools/call", {"name": tool, "arguments": arguments})
        assert r.status_code == 200, r.text
        result = r.json()["result"]
        assert not result.get("isError"), result
        return result.get("structuredContent") or json.loads(result["content"][0]["text"])


@pytest.fixture(scope="module")
def mcp(api_key):
    m = MCP(headers={"Authorization": f"Bearer {api_key}"})
    m.initialize()
    yield m
    m.client.close()


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer rag_wrong"}])
def test_mcp_needs_a_key(headers):
    assert httpx.post(BASE_URL + "/mcp", headers={**HEADERS, **headers}, json={}).status_code == 401


def test_key_in_url_works(api_key):
    # claude.ai custom connectors can only pass the key in the URL.
    info = MCP(params={"key": api_key}).initialize()
    assert info["serverInfo"]["name"] == "RAG Studio"
    assert "search" in info["instructions"]


def test_tools_are_listed(mcp):
    tools = {t["name"] for t in mcp.rpc("tools/list").json()["result"]["tools"]}
    assert tools == {"search", "fetch", "ask", "list_documents"}


def test_list_documents(mcp, lighthouse):
    docs = mcp.call("list_documents")["documents"]
    assert any(d["id"] == lighthouse["id"] and d["status"] == "ready" for d in docs)


def test_search_then_fetch(mcp, lighthouse):
    results = mcp.call("search", query=QUESTION, top_k=3)["results"]
    top = results[0]
    assert top["id"] == f"{lighthouse['id']}:1"
    assert top["title"] == "lighthouse.pdf (page 1)"
    assert ANSWER in top["text"]

    page = mcp.call("fetch", id=top["id"])
    assert ANSWER in page["text"]
    assert page["metadata"] == {"filename": "lighthouse.pdf", "page": 1, "pages": 2}


def test_fetch_unknown_id_is_an_error(mcp):
    r = mcp.rpc("tools/call", {"name": "fetch", "arguments": {"id": "nope:1"}}).json()
    assert r.get("error") or r["result"]["isError"]


def test_ask(mcp, lighthouse):
    r = mcp.call("ask", question=QUESTION)
    assert ANSWER in r["answer"]
    assert r["sources"][0] == {"id": f"{lighthouse['id']}:1", "filename": "lighthouse.pdf", "page": 1}
