import json

import pytest

from evals import compare


def result(config, ranks, sha256="f" * 64, tags=None):
    """A result file with only what compare reads: each question's id, tags and first-hit rank."""
    tags = tags or [["exact"] if i % 2 else [] for i in range(len(ranks))]
    return {"config": config, "commit": "abc1234", "dataset": {"sha256": sha256} if sha256 else {},
            "questions": [{"id": f"q{i}", "tags": t, "rank": r} for i, (r, t) in enumerate(zip(ranks, tags))]}


A = result("vector", [1, 2, None, 1, 3, None, 1, 6])
B = result("hybrid", [1, 1, 1, 1, 1, None, 1, 2])


def test_rows_per_slice_and_metric():
    rows = compare.compare(A, B)
    assert [(r["slice"], r["metric"]) for r in rows] == [
        (s, m) for s in ("all", "exact") for m in ("recall@1", "recall@5", "mrr@10")]
    r1 = rows[0]
    assert (r1["questions"], r1["a"], r1["b"]) == (8, 3 / 8, 6 / 8) and r1["diff"] == pytest.approx(3 / 8)
    exact = rows[3]  # q1, q3, q5, q7: ranks 2, 1, None, 6 against 1, 1, None, 2
    assert (exact["questions"], exact["a"], exact["b"]) == (4, 1 / 4, 2 / 4)


def test_pairs_by_question_id_not_position():
    shuffled = {**B, "questions": list(reversed(B["questions"]))}
    assert compare.compare(A, shuffled) == compare.compare(A, B)


def test_refuses_different_datasets():
    with pytest.raises(compare.Mismatch, match="Different datasets"):
        compare.check_same_questions(A, result("hybrid", [1] * 8, sha256="0" * 64))


def test_refuses_different_questions_or_tags():
    with pytest.raises(compare.Mismatch, match="different questions"):
        compare.check_same_questions(A, result("hybrid", [1] * 7))
    with pytest.raises(compare.Mismatch, match="different questions"):
        compare.check_same_questions(A, result("hybrid", [1] * 8, tags=[["exact"]] * 8))


def test_files_without_a_fingerprint_are_matched_by_id_with_a_note():
    assert compare.check_same_questions(A, B) is None
    unhashed = result("hybrid", [1] * 8, sha256=None)
    assert compare.check_same_questions(A, unhashed) == (
        "One of the files has no dataset fingerprint: questions were matched by id and tags only.")
    assert compare.check_same_questions({**A, "dataset": {}}, unhashed).startswith("Neither file has a dataset")


def test_table_and_main(tmp_path, capsys):
    rows = compare.compare(A, B)
    out = compare.table(A, B, rows)
    assert "| all | 8 | recall@1 | 37.5% | 75.0% | +37.5 [" in out and "| all | 8 | mrr@10 |" in out
    # recall@1: q2, q4, q7 go from a miss or rank > 1 to rank 1, none the other way: 3 / 0, sign test 2 / 2^3.
    r1 = rows[0]
    assert (r1["wins"], r1["losses"], r1["sign_p"]) == (3, 0, 0.25)
    assert "| +3 / −0 | 0.250 |" in out.splitlines()[4]
    assert compare._p(0.0004) == "<0.001" and compare._p(0.25) == "0.250"
    for name, r in (("a.json", A), ("b.json", B)):
        (tmp_path / name).write_text(json.dumps(r), encoding="utf-8")
    compare.main([str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert capsys.readouterr().out.startswith("A = vector (`abc1234`), B = hybrid (`abc1234`).")
