import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import auth, db
from conftest import PASSWORD


def test_lockout_after_max_failures():
    for _ in range(auth.MAX_FAILURES):
        assert auth.login("admin", "wrong", "1.1.1.1") is None
    with pytest.raises(auth.LockedOut):
        auth.login("admin", PASSWORD, "1.1.1.1")


def test_unknown_user_counts_toward_lockout():
    for _ in range(auth.MAX_FAILURES):
        assert auth.login("nobody", "x", "2.2.2.2") is None
    with pytest.raises(auth.LockedOut):
        auth.login("nobody", "x", "2.2.2.2")


def test_unknown_user_costs_a_password_check():
    t = time.perf_counter()
    auth.login("nobody", "x", "2.2.2.2")
    unknown = time.perf_counter() - t
    t = time.perf_counter()
    auth.login("admin", "x", "3.3.3.3")
    known = time.perf_counter() - t
    assert unknown > known / 2


def test_success_returns_session_and_clears_failures():
    assert auth.login("admin", "wrong", "4.4.4.4") is None
    token = auth.login("admin", PASSWORD, "4.4.4.4")
    assert auth.session_user(token)["username"] == "admin"
    assert "4.4.4.4" not in auth._failures


def test_parallel_attempts_cannot_pass_the_lockout():
    def attempt(_):
        try:
            auth.login("admin", "wrong", "5.5.5.5")
            return "tried"
        except auth.LockedOut:
            return "locked"

    with ThreadPoolExecutor(40) as ex:
        results = list(ex.map(attempt, range(40)))
    assert results.count("tried") == auth.MAX_FAILURES


def test_expired_ips_are_swept():
    old = time.time() - auth.LOCKOUT_S - 1
    for i in range(auth._SWEEP_AT + 50):
        auth._failures[f"10.0.{i // 256}.{i % 256}"] = [old]
    auth.login("admin", "wrong", "9.9.9.9")
    assert list(auth._failures) == ["9.9.9.9"]


def test_failed_login_events_are_capped():
    for i in range(auth.MAX_FAILURE_LOGS + 10):
        auth.login("admin", "wrong", f"20.0.0.{i}")
    count = lambda pattern: db.rows("SELECT COUNT(*) c FROM events WHERE message LIKE ?", (pattern,))[0]["c"]
    assert count("failed login%") == auth.MAX_FAILURE_LOGS
    assert count("further failed logins%") == 1


def test_event_trim_keeps_auth_and_other_events_separately(monkeypatch):
    monkeypatch.setattr(db, "KEEP_EVENTS", 50)
    for i in range(70):
        db.log("info", "auth", f"auth {i}")
    for i in range(120):
        db.log("info", "app", f"noise {i}")
    counts = {r["auth"]: r["c"] for r in db.rows("SELECT source = 'auth' auth, COUNT(*) c FROM events GROUP BY 1")}
    assert counts == {1: 50, 0: 50}
    assert db.rows("SELECT message FROM events WHERE source='auth' ORDER BY id DESC LIMIT 1")[0]["message"] == "auth 69"


def test_reset_password_for_missing_user_fails():
    with pytest.raises(ValueError):
        auth.reset_password("ghost")
    assert not db.rows("SELECT 1 FROM events WHERE message LIKE '%ghost%'")
    assert auth.reset_password("admin")


def test_reset_admin_cli_exits_1_for_missing_user():
    r = subprocess.run([sys.executable, "-m", "app.auth", "reset-admin", "ghost"], capture_output=True, text=True)
    assert r.returncode == 1
    assert not r.stdout
    assert "ghost" in r.stderr
