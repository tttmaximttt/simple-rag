"""Compare two eval results question by question.

  python -m evals.compare evals/results/2026-11-05-hybrid.json evals/results/2026-11-05-hybrid-rerank.json

For Recall@1, Recall@5 and MRR@10, overall and per tag, prints B minus A with its 95% paired bootstrap interval
and a two-sided p-value, and how many questions B wins and loses with the exact sign test (evals/stats.py). Both files must have asked the same questions: results with different
dataset fingerprints are refused. Files written before the fingerprint was stored are compared when their question
ids and tags match, with a note saying so.
"""
import argparse
import json
import sys
from pathlib import Path

from . import stats


class Mismatch(SystemExit):
    pass


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_same_questions(a: dict, b: dict) -> str | None:
    """Raise Mismatch when a and b can't be paired; return a note when the pairing rests on question ids only."""
    ha, hb = a["dataset"].get("sha256"), b["dataset"].get("sha256")
    if ha and hb and ha != hb:
        raise Mismatch(f"Different datasets: {a['config']} has {ha[:12]}, {b['config']} has {hb[:12]}. "
                       "Re-run both on the same dataset.jsonl and PDFs.")
    ids = lambda r: {q["id"]: tuple(q.get("tags", [])) for q in r["questions"]}
    if ids(a) != ids(b):
        raise Mismatch(f"{a['config']} and {b['config']} asked different questions (or tagged them differently)")
    if not (ha and hb):
        which = "Neither file has" if not (ha or hb) else "One of the files has no"
        return f"{which}{' a' if not (ha or hb) else ''} dataset fingerprint: questions were matched by id and tags only."
    return None


def compare(a: dict, b: dict, **kw) -> list[dict]:
    """One row per slice and metric: the means of A and B, the paired bootstrap of B minus A, and the exact sign
    test over the questions that moved. Call check_same_questions first."""
    qa, qb = ({q["id"]: q for q in r["questions"]} for r in (a, b))
    tags = sorted({t for q in a["questions"] for t in q.get("tags", [])})
    rows = []
    for name in ["all", *tags]:
        ids = [i for i, q in qa.items() if name == "all" or name in q.get("tags", [])]
        va, vb = ([stats.per_question(q[i]["rank"]) for i in ids] for q in (qa, qb))
        for metric, field in stats.METRICS.items():
            xa, xb = [v[field] for v in va], [v[field] for v in vb]
            sign = stats.sign_test(xa, xb)
            rows.append({"slice": name, "questions": len(ids), "metric": metric, "a": sum(xa) / len(xa),
                         "b": sum(xb) / len(xb), **stats.paired_bootstrap(xa, xb, **kw),
                         "wins": sign["wins"], "losses": sign["losses"], "sign_p": sign["p"]})
    return rows


def _value(metric: str, x: float) -> str:
    return f"{x:.3f}" if metric.startswith("mrr") else f"{x * 100:.1f}%"


def _diff(metric: str, x: float) -> str:
    return f"{x:+.3f}" if metric.startswith("mrr") else f"{x * 100:+.1f}"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table(a: dict, b: dict, rows: list[dict]) -> str:
    lines = [f"A = {a['config']} (`{a['commit']}`), B = {b['config']} (`{b['commit']}`). "
             "B − A with its 95% paired bootstrap interval (Recall in points, MRR as is) and two-sided p-value; "
             "moved: questions B wins / loses, with the exact sign test's p. Not corrected for multiple comparisons.",
             "", "| slice | questions | metric | A | B | B − A [95% CI] | p | moved | sign test p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["metric"]
        lines.append(f"| {r['slice']} | {r['questions']} | {m} | {_value(m, r['a'])} | {_value(m, r['b'])} "
                     f"| {_diff(m, r['diff'])} [{_diff(m, r['ci'][0])}, {_diff(m, r['ci'][1])}] | {_p(r['p'])} "
                     f"| +{r['wins']} / −{r['losses']} | {_p(r['sign_p'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare two eval results on the same questions.")
    ap.add_argument("a", type=Path, help="baseline result JSON")
    ap.add_argument("b", type=Path, help="result JSON to compare against it")
    args = ap.parse_args(argv)
    a, b = _load(args.a), _load(args.b)
    note = check_same_questions(a, b)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if note:
        print(f"Note: {note}\n")
    print(table(a, b, compare(a, b)), end="")


if __name__ == "__main__":
    main()
