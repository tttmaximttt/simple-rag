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
import time

from . import db

log = logging.getLogger("uvicorn.error")

SESSION_COOKIE = "rag_session"
SESSION_TTL_S = 7 * 86400
MIN_PASSWORD_LEN = 10
# Brute-force protection: after this many failures from one IP, lock it out for a while.
MAX_FAILURES, LOCKOUT_S = 5, 300
_failures: dict[str, list[float]] = {}


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    _, salt, _ = stored.split("$")
    return hmac.compare_digest(_hash_password(password, bytes.fromhex(salt)), stored)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


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
    db.execute("UPDATE users SET password_hash=?, must_change=1 WHERE username=?", (_hash_password(password), username))
    db.execute("DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE username=?)", (username,))
    db.log("warn", "auth", f"password for {username!r} reset from the command line")
    return password


def locked_out(ip: str) -> bool:
    recent = [t for t in _failures.get(ip, []) if t > time.time() - LOCKOUT_S]
    _failures[ip] = recent
    return len(recent) >= MAX_FAILURES


def login(username: str, password: str, ip: str) -> str | None:
    rows = db.rows("SELECT id, password_hash FROM users WHERE username=?", (username,))
    if not rows or not _verify_password(password, rows[0]["password_hash"]):
        _failures.setdefault(ip, []).append(time.time())
        db.log("warn", "auth", f"failed login for {username!r} from {ip}")
        return None
    _failures.pop(ip, None)
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.execute("DELETE FROM sessions WHERE expires < ?", (now,))
    db.execute("INSERT INTO sessions (token_hash, user_id, created, expires, ip) VALUES (?,?,?,?,?)",
               (_token_hash(token), rows[0]["id"], now, now + SESSION_TTL_S, ip))
    db.execute("UPDATE users SET last_login=? WHERE id=?", (now, rows[0]["id"]))
    db.log("info", "auth", f"{username} logged in from {ip}")
    return token


def session_user(token: str | None) -> dict | None:
    if not token:
        return None
    rows = db.rows("""SELECT u.id, u.username, u.must_change FROM sessions s JOIN users u ON u.id = s.user_id
                      WHERE s.token_hash=? AND s.expires > ?""", (_token_hash(token), time.time()))
    return rows[0] if rows else None


def logout(token: str):
    db.execute("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))


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
    db.execute("DELETE FROM sessions WHERE user_id=? AND token_hash<>?", (user_id, _token_hash(keep_token)))
    db.log("info", "auth", f"{row['username']} changed their password")


if __name__ == "__main__":
    # docker compose exec app python -m app.auth reset-admin [username]
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "reset-admin":
        db.init()
        user = sys.argv[2] if len(sys.argv) > 2 else "admin"
        print(f"New temporary password for {user}: {reset_password(user)}")
    else:
        print("usage: python -m app.auth reset-admin [username]")
