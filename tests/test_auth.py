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


def test_unknown_user_costs_a_password_check(monkeypatch):
    checked = []
    monkeypatch.setattr(auth, "_verify_password", lambda password, stored: checked.append(stored) or False)
    assert auth.login("nobody", "x", "2.2.2.2") is None
    assert checked == [auth._DUMMY_HASH]


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


def test_live_ips_are_not_rescanned_on_every_login(monkeypatch):
    monkeypatch.setattr(auth, "_next_sweep", auth._SWEEP_AT)
    now = time.time()
    for i in range(auth._SWEEP_AT + 50):
        auth._failures[f"10.0.{i // 256}.{i % 256}"] = [now]
    auth.login("admin", "wrong", "9.9.9.9")
    assert len(auth._failures) == auth._SWEEP_AT + 51
    assert auth._next_sweep == 2 * (auth._SWEEP_AT + 50)  # set by the sweep, before 9.9.9.9 was added


def test_errors_during_the_check_do_not_count_as_failures(monkeypatch):
    def broken(password, stored):
        raise ValueError("malformed hash")
    monkeypatch.setattr(auth, "_verify_password", broken)
    for _ in range(auth.MAX_FAILURES + 1):
        with pytest.raises(ValueError):
            auth.login("admin", PASSWORD, "6.6.6.6")
    assert not auth._failures.get("6.6.6.6")
    monkeypatch.undo()
    assert auth.login("admin", PASSWORD, "6.6.6.6")


def test_sweep_after_an_error_during_the_check(monkeypatch):
    def broken(password, stored):
        raise ValueError("malformed hash")
    monkeypatch.setattr(auth, "_verify_password", broken)
    with pytest.raises(ValueError):
        auth.login("admin", PASSWORD, "6.6.6.6")
    assert "6.6.6.6" not in auth._failures
    monkeypatch.undo()
    auth._failures["6.6.6.7"] = []  # an emptied entry must not break the sweep either
    monkeypatch.setattr(auth, "_next_sweep", 0)
    assert auth.login("admin", PASSWORD, "6.6.6.6")
    assert "6.6.6.7" not in auth._failures


def test_failed_login_events_are_capped():
    for i in range(auth.MAX_FAILURE_LOGS + 10):
        auth.login("admin", "wrong", f"20.0.0.{i}")
    count = lambda pattern: db.rows("SELECT COUNT(*) c FROM events WHERE message LIKE ?", (pattern,))[0]["c"]
    assert count("failed login%") == auth.MAX_FAILURE_LOGS
    assert count("further failed logins%") == 1


def _end_log_window():
    auth._failure_logs[0] -= auth.LOCKOUT_S
    auth._unlogged[0] -= auth.LOCKOUT_S


def _messages(pattern):
    return [r["message"] for r in db.rows("SELECT message FROM events WHERE message LIKE ?", (pattern,))]


def test_lockout_is_logged_and_rejected_attempts_summarized():
    for _ in range(auth.MAX_FAILURES):
        auth.login("admin", "wrong", "7.7.7.7")
    for _ in range(30):
        with pytest.raises(auth.LockedOut):
            auth.login("admin", "wrong", "7.7.7.7")
    assert len(_messages("%7.7.7.7 locked out%")) == 1
    assert len(_messages("%7.7.7.7%")) == auth.MAX_FAILURES  # rejections aren't written one by one
    _end_log_window()
    with pytest.raises(auth.LockedOut):
        auth.login("admin", "wrong", "7.7.7.7")
    assert _messages("%not logged one by one%") == [
        "30 failed or locked-out logins from 1 IP(s) were not logged one by one in the last window, "
        "most from 7.7.7.7 (30)"]


def test_capped_failures_still_show_the_busiest_ip():
    for i in range(auth.MAX_FAILURE_LOGS):  # chaff that uses up the window's cap
        auth.login("admin", "wrong", f"21.0.0.{i}")
    for _ in range(auth.MAX_FAILURES + 10):  # then a targeted attempt from one IP
        try:
            auth.login("admin", "wrong", "8.8.8.8")
        except auth.LockedOut:
            pass
    assert not _messages("failed login%8.8.8.8%")
    _end_log_window()
    auth.login("admin", PASSWORD, "1.2.3.4")  # a successful login also writes the summary
    assert _messages("%not logged one by one%") == [
        f"{auth.MAX_FAILURES + 10} failed or locked-out logins from 1 IP(s) were not logged one by one "
        f"in the last window, most from 8.8.8.8 ({auth.MAX_FAILURES + 10})"]


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
