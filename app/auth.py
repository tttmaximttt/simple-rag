"""Admin login for the web UI and its internal /api routes.

On first start an `admin` account is created with a temporary password (from
ADMIN_PASSWORD, or random and printed to the container logs). The first login must
change it before anything else is reachable.
"""
import hashlib
import hmac
import logging
import os
import secrets
import threading
import time

from . import db

log = logging.getLogger("uvicorn.error")

SESSION_COOKIE = "rag_session"
SESSION_TTL_S = 7 * 86400
MIN_PASSWORD_LEN = 10
# Brute-force protection: after this many failures from one IP, lock it out for a while.
MAX_FAILURES, LOCKOUT_S = 5, 300
# At most this many failed-login events are written per LOCKOUT_S window, so a flood can't bury the audit log.
MAX_FAILURE_LOGS = 20
_SWEEP_AT = 1000  # sweep expired IPs once _failures grows past this
_failures: dict[str, list[float]] = {}
_next_sweep = _SWEEP_AT
_failure_logs = [0.0, 0]  # window start, failures logged in it
# Attempts not written one by one (over the cap, or rejected while locked out), summarized when the window ends:
# window start, total, count per IP (at most _SWEEP_AT IPs).
_unlogged: list = [0.0, 0, {}]
_lock = threading.Lock()


class LockedOut(Exception):
    """Too many failed logins from this IP; retry after LOCKOUT_S."""


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    _, salt, _ = stored.split("$")
    return hmac.compare_digest(_hash_password(password, bytes.fromhex(salt)), stored)


# Checked against when the username doesn't exist, so unknown users cost the same scrypt.
_DUMMY_HASH = _hash_password(secrets.token_urlsafe(16))


def secret_hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _reserve_attempt(ip: str) -> tuple[float, int]:
    """Count this attempt as a failure up front, or raise LockedOut. Cleared on success.

    Returns its timestamp and how many failures that makes for this IP."""
    global _next_sweep
    now = time.time()
    cutoff = now - LOCKOUT_S
    with _lock:
        if len(_failures) > _next_sweep:
            for k in [k for k, ts in _failures.items() if not ts or ts[-1] <= cutoff]:
                del _failures[k]
            # Rescan only once it has doubled again, so many live IPs don't make every login a full scan.
            _next_sweep = max(_SWEEP_AT, 2 * len(_failures))
        recent = [t for t in _failures.get(ip, []) if t > cutoff]
        if len(recent) >= MAX_FAILURES:
            _failures[ip] = recent
            raise LockedOut(ip)
        _failures[ip] = recent + [now]
        return now, len(recent) + 1


def _roll_window(now: float) -> str | None:
    """Start a new log window if the current one is over. Returns the summary of what it didn't log. Holds _lock."""
    if now - _failure_logs[0] < LOCKOUT_S:
        return None
    summary = None
    if _unlogged[0] == _failure_logs[0] and _unlogged[1]:
        by_ip = _unlogged[2]
        top = max(by_ip, key=by_ip.get)
        summary = (f"{_unlogged[1]} failed or locked-out logins from {len(by_ip)}"
                   f"{'+' if len(by_ip) >= _SWEEP_AT else ''} IP(s) were not logged one by one "
                   f"in the last window, most from {top} ({by_ip[top]})")
    _failure_logs[:] = [now, 0]
    _unlogged[:] = [now, 0, {}]
    return summary


def _log_failure(message: str | None, ip: str):
    """Log a failed login, or count it once the window's cap is reached. message=None is an attempt rejected by
    the lockout: those are never written one by one, so they can't use up the cap, only summarized."""
    with _lock:
        summary = _roll_window(time.time())
        if message is not None:
            _failure_logs[1] += 1
        n = _failure_logs[1]
        write = message is not None and n <= MAX_FAILURE_LOGS
        if not write:
            _unlogged[1] += 1
            by_ip = _unlogged[2]
            if ip in by_ip or len(by_ip) < _SWEEP_AT:
                by_ip[ip] = by_ip.get(ip, 0) + 1
    if summary:
        db.log("warn", "auth", summary)
    if write:
        db.log("warn", "auth", message)
    elif message is not None and n == MAX_FAILURE_LOGS + 1:
        db.log("warn", "auth", f"further failed logins suppressed for up to {LOCKOUT_S // 60} minutes")


def bootstrap():
    """Create the admin account with a temporary password on first start."""
    if db.rows("SELECT 1 FROM users LIMIT 1"):
        return
    username = os.environ.get("ADMIN_USER", "admin")
    password = os.environ.get("ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    db.execute("INSERT INTO users (username, password_hash, must_change, created) VALUES (?,?,1,?)",
               (username, _hash_password(password), time.time()))
    db.log("info", "auth", f"admin account {username!r} created with a temporary password")
    if not os.environ.get("ADMIN_PASSWORD"):
        log.warning("\n\n  RAG Studio first-run login:  user %s  temporary password %s\n"
                    "  You will be asked to change it after logging in.\n", username, password)


def reset_password(username: str) -> str:
    """Set a new temporary password (used by the `reset-admin` CLI)."""
    password = secrets.token_urlsafe(12)
    if not db.execute("UPDATE users SET password_hash=?, must_change=1 WHERE username=?",
                      (_hash_password(password), username)):
        raise ValueError(f"No user named {username!r}")
    db.execute("DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE username=?)", (username,))
    db.log("warn", "auth", f"password for {username!r} reset from the command line")
    return password


def login(username: str, password: str, ip: str) -> str | None:
    """Return a session token, or None on bad credentials. Raises LockedOut."""
    try:
        reserved, n = _reserve_attempt(ip)
    except LockedOut:
        _log_failure(None, ip)
        raise
    try:
        rows = db.rows("SELECT id, password_hash FROM users WHERE username=?", (username,))
        ok = _verify_password(password, rows[0]["password_hash"] if rows else _DUMMY_HASH)
    except Exception:
        # An error isn't a wrong password: don't let it count toward the lockout.
        with _lock:
            if reserved in _failures.get(ip, []):
                _failures[ip].remove(reserved)
                if not _failures[ip]:
                    del _failures[ip]
        raise
    if not rows or not ok:
        locked = f", {ip} locked out for {LOCKOUT_S // 60} minutes" if n >= MAX_FAILURES else ""
        _log_failure(f"failed login for {username!r} from {ip}{locked}", ip)
        return None
    with _lock:
        _failures.pop(ip, None)
        summary = _roll_window(time.time())
    if summary:
        db.log("warn", "auth", summary)
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.execute("DELETE FROM sessions WHERE expires < ?", (now,))
    db.execute("INSERT INTO sessions (token_hash, user_id, created, expires, ip) VALUES (?,?,?,?,?)",
               (secret_hash(token), rows[0]["id"], now, now + SESSION_TTL_S, ip))
    db.execute("UPDATE users SET last_login=? WHERE id=?", (now, rows[0]["id"]))
    db.log("info", "auth", f"{username} logged in from {ip}")
    return token


def session_user(token: str | None) -> dict | None:
    if not token:
        return None
    rows = db.rows("""SELECT u.id, u.username, u.must_change FROM sessions s JOIN users u ON u.id = s.user_id
                      WHERE s.token_hash=? AND s.expires > ?""", (secret_hash(token), time.time()))
    return rows[0] if rows else None


def logout(token: str):
    db.execute("DELETE FROM sessions WHERE token_hash=?", (secret_hash(token),))


def change_password(user_id: int, current: str, new: str, keep_token: str):
    row = db.rows("SELECT username, password_hash FROM users WHERE id=?", (user_id,))[0]
    if not _verify_password(current, row["password_hash"]):
        raise ValueError("Current password is wrong")
    if len(new) < MIN_PASSWORD_LEN:
        raise ValueError(f"New password must be at least {MIN_PASSWORD_LEN} characters")
    if new == current:
        raise ValueError("New password must be different from the current one")
    db.execute("UPDATE users SET password_hash=?, must_change=0 WHERE id=?", (_hash_password(new), user_id))
    # Sign out every other session.
    db.execute("DELETE FROM sessions WHERE user_id=? AND token_hash<>?", (user_id, secret_hash(keep_token)))
    db.log("info", "auth", f"{row['username']} changed their password")


if __name__ == "__main__":
    # docker compose exec app python -m app.auth reset-admin [username]
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "reset-admin":
        db.init()
        user = sys.argv[2] if len(sys.argv) > 2 else "admin"
        try:
            print(f"New temporary password for {user}: {reset_password(user)}")
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
    else:
        print("usage: python -m app.auth reset-admin [username]")
