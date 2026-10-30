"""End-to-end tests against a running RAG Studio (see e2e/run.sh, which starts a fresh stack).

Settings come from env vars, so the same tests run against the docker stack, a local uvicorn,
fake models or real providers.
"""
import os
import secrets
import time
from pathlib import Path

import httpx
import pytest

from pdfs import make_pdf

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:18080").rstrip("/")
# The first-run password the stack was started with (ADMIN_PASSWORD), and the one the tests change it to.
TEMP_PASSWORD = os.environ.get("E2E_ADMIN_PASSWORD", "e2e-temporary-pw")
PASSWORD = os.environ.get("E2E_PASSWORD", "e2e-admin-password-1")
# Fake model server, as the app sees it.
FAKE_URL = os.environ.get("E2E_FAKE_URL", "http://fake-models:9000").rstrip("/")
VECTOR_STORE = os.environ.get("E2E_VECTOR_STORE", "pgvector")  # pgvector | chroma | qdrant
EMBEDDINGS = os.environ.get("E2E_EMBEDDINGS", "fake")  # fake | openai | local
LLM = os.environ.get("E2E_LLM", "fake")  # fake | openai | anthropic
# Host path mounted as the app's import folder, for the server-folder import test.
IMPORT_DIR = os.environ.get("E2E_IMPORT_DIR")
INGEST_TIMEOUT_S = float(os.environ.get("E2E_INGEST_TIMEOUT", "180"))
# Makes every PDF unique per run, so re-running against a kept stack doesn't hit duplicate detection.
RUN = secrets.token_hex(4)

LIGHTHOUSE = [
    f"The Marrowstone lighthouse keeper is Orla Featherstone. She has kept the light since 1987. Run {RUN}.",
    f"Orchard notes: the quince trees on the north slope flower in late April. Run {RUN}.",
]
QUESTION = "Who is the Marrowstone lighthouse keeper?"
ANSWER = "Featherstone"


def _key(name):
    if not os.environ.get(name):
        pytest.exit(f"{name} must be set to test with the real {name.split('_')[0].title()} API", 2)
    return os.environ[name]


def wizard_settings() -> dict:
    """The setup-wizard values for the chosen backends (only the fields of the active providers)."""
    fake = {"api_key": "fake", "base_url": FAKE_URL + "/v1"}
    emb = {
        "fake": {"provider": "openai", "openai": {**fake, "model": "fake-embed"}},
        "openai": {"provider": "openai", "openai": {"api_key": _key("OPENAI_API_KEY") if EMBEDDINGS == "openai" else "",
                                                    "base_url": "https://api.openai.com/v1"}},
        "local": {"provider": "local"},
    }[EMBEDDINGS]
    llm = {
        "fake": {"provider": "openai", "openai": {**fake, "model": "fake-chat"}},
        "openai": {"provider": "openai", "openai": {"api_key": _key("OPENAI_API_KEY") if LLM == "openai" else "",
                                                    "base_url": "https://api.openai.com/v1"}},
        "anthropic": {"provider": "anthropic",
                      "anthropic": {"api_key": _key("ANTHROPIC_API_KEY") if LLM == "anthropic" else ""}},
    }[LLM]
    store = {"type": VECTOR_STORE, "collection": f"e2e_{EMBEDDINGS}"}
    if VECTOR_STORE == "qdrant":
        store["qdrant"] = {"url": os.environ.get("E2E_QDRANT_URL", "http://qdrant:6333")}
    return {"vector_store": store, "embeddings": emb, "llm": llm}


def pytest_collection_modifyitems(items):
    # The first-run flow needs the untouched stack, so it goes before anything that logs in.
    items.sort(key=lambda i: "test_first_run" not in i.nodeid)


@pytest.fixture(scope="session", autouse=True)
def stack_is_up():
    wizard_settings()  # fails fast when a real provider is chosen without its key
    try:
        httpx.get(BASE_URL + "/api/health", timeout=5).raise_for_status()
    except httpx.HTTPError as e:
        pytest.exit(f"RAG Studio is not reachable at {BASE_URL} ({e}). Start it with e2e/run.sh.", 2)


def login(client: httpx.Client, password: str) -> httpx.Response:
    return client.post("/api/auth/login", json={"username": "admin", "password": password})


@pytest.fixture(scope="session")
def admin() -> httpx.Client:
    """An admin session. Finishes the first-run steps over the API if the UI test didn't."""
    c = httpx.Client(base_url=BASE_URL, timeout=60)
    if login(c, PASSWORD).status_code != 200:
        assert login(c, TEMP_PASSWORD).status_code == 200, "neither E2E_PASSWORD nor E2E_ADMIN_PASSWORD works"
        r = c.post("/api/auth/password", json={"current_password": TEMP_PASSWORD, "new_password": PASSWORD})
        assert r.status_code == 200, r.text
        assert login(c, PASSWORD).status_code == 200
    if not c.get("/api/health").json()["configured"]:
        c.put("/api/config", json=wizard_settings()).raise_for_status()
    yield c
    c.close()


@pytest.fixture(scope="session")
def api_key(admin) -> str:
    # Write scope: the suite uploads and imports documents with it.
    return admin.post("/api/admin/keys", json={"name": f"e2e-{RUN}", "scope": "write"}).json()["key"]


@pytest.fixture(scope="session")
def v1(api_key) -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=120, headers={"Authorization": f"Bearer {api_key}"}) as c:
        yield c


def wait_ready(client: httpx.Client, doc_id: str, path="/v1/documents") -> dict:
    deadline = time.time() + INGEST_TIMEOUT_S
    while time.time() < deadline:
        doc = next((d for d in client.get(path).json() if d["id"] == doc_id), None)
        assert doc, f"document {doc_id} disappeared"
        if doc["status"] in ("ready", "failed"):
            assert doc["status"] == "ready", doc["error"]
            return doc
        time.sleep(0.5)
    raise AssertionError(f"document {doc_id} not ingested within {INGEST_TIMEOUT_S}s")


def upload(client: httpx.Client, name: str, pages: list[str], path="/v1/documents") -> dict:
    r = client.post(path, files=[("files", (name, make_pdf(pages), "application/pdf"))])
    assert r.status_code == 200, r.text
    return r.json()["results"][0]


@pytest.fixture(scope="session")
def lighthouse(v1) -> dict:
    """A two-page PDF uploaded over /v1 and fully ingested."""
    res = upload(v1, "lighthouse.pdf", LIGHTHOUSE)
    assert res["status"] == "queued", res
    return wait_ready(v1, res["id"])


@pytest.fixture
def import_dir() -> Path:
    if not IMPORT_DIR:
        pytest.skip("E2E_IMPORT_DIR is not set (e2e/run.sh sets it)")
    return Path(IMPORT_DIR)


# ---------- Playwright ----------

@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    # Use an already installed Chromium instead of `playwright install` (e.g. in a sandbox).
    path = os.environ.get("E2E_CHROMIUM")
    return {**browser_type_launch_args, **({"executable_path": path} if path else {})}


@pytest.fixture
def ui(page, admin):
    """A browser page logged in as the admin, on the Documents view."""
    page.goto("/")
    page.fill("#login-form [name=password]", PASSWORD)
    page.click("#login-form button")
    page.wait_for_selector("#nav", state="visible")
    return page
