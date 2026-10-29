from app import db


def test_init_creates_event_partition_index_idempotently():
    db.init()
    db.init()
    names = [r["name"] for r in db.rows("PRAGMA index_list(events)")]
    assert names.count("events_part") == 1


def test_event_trim_uses_the_partition_index(monkeypatch):
    statements = []
    real_execute = db.execute
    monkeypatch.setattr(db, "execute", lambda sql, args=(): statements.append(sql) or real_execute(sql, args))
    for source in ("auth", "app"):
        statements.clear()
        db.log("info", source, "x")
        delete = next(sql for sql in statements if sql.startswith("DELETE"))
        plan = db.rows("EXPLAIN QUERY PLAN " + delete)
        assert any("events_part" in r["detail"] for r in plan)
        assert not any(r["detail"].startswith("SCAN") for r in plan)


def test_event_trim_keeps_newest_rows_per_partition(monkeypatch):
    monkeypatch.setattr(db, "KEEP_EVENTS", 5)
    for i in range(12):
        db.log("info", "auth", f"a{i}")
        db.log("info", "app", f"b{i}")
    for source, last in (("auth", "a"), ("app", "b")):
        msgs = [r["message"] for r in db.rows("SELECT message FROM events WHERE source=? ORDER BY id", (source,))]
        assert msgs[-1] == f"{last}11"
        assert len(msgs) == 5
