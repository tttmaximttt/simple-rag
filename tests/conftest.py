import os
import tempfile

# app.config reads DATA_DIR at import, so set it before anything imports the app.
os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["IMPORT_DIR"] = tempfile.mkdtemp()
os.environ["ADMIN_PASSWORD"] = "initial-pass-123"

import pytest
from fastapi.testclient import TestClient

from app import auth, db, main

PASSWORD = "initial-pass-123"


@pytest.fixture(autouse=True)
def fresh_auth(monkeypatch):
    """A single admin with PASSWORD, no sessions, events or failure history."""
    db.init()
    for table in ("events", "sessions", "users", "api_keys"):
        db.execute(f"DELETE FROM {table}")
    auth._failures.clear()
    auth._failure_logs[:] = [0.0, 0]
    auth.bootstrap()
    db.execute("UPDATE users SET must_change=0")
    monkeypatch.setattr(main, "PUBLIC_URL", "")


@pytest.fixture(scope="session")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def logged_in(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
    assert r.status_code == 200
    yield client
    client.cookies.clear()
