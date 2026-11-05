"""Summarize the eval results in one Markdown table, for the README.

  python -m evals.report                      # evals/results/*.json
  python -m evals.report --dir some/folder

One row per config, in ORDER and then by name, from the newest file of each config. Under the table, one line
with what the runs were measured on, taken from the files. The numbers are only comparable when every file
comes from the same commit and hardware, so a warning row says when they don't.

Recall, MRR and p95 are computed from each question's rank and latency in the file, not read from its summary:
the summary is rounded to four places, and rounding that again for display can be off by 0.1 point. Recall and
MRR carry their 95% bootstrap interval (evals/stats.py), so files from before intervals were stored get them too.
"""
import argparse
import json
import sys
from pathlib import Path

from . import metrics as m
from . import stats
from .run import RESULTS, _ci, _ms

ORDER = ["vector", "vector-rewrite", "hybrid", "hybrid-rewrite", "hybrid-rerank", "hybrid-rerank-gpu"]
COLUMNS = ["config", "Recall@1", "Recall@5", "MRR@10", "Recall@1 exact", "Recall@1 cross-language", "p95 latency"]


def load(folder: Path) -> tuple[list[dict], list[str]]:
    """The newest result of each config (file names sort by date, then time), and a warning for each config that
    has older results beside it."""
    files = {}
    for path in sorted(folder.glob("*.json"), key=lambda p: p.stem):  # stem: "x" before "x-213000"
        r = json.loads(path.read_text(encoding="utf-8"))
        files.setdefault(r["config"], []).append((path, r))
    warnings = [f"{len(found)} results for {c}, using {found[-1][0].name}" for c, found in files.items() if len(found) > 1]
    rank = lambda c: (ORDER.index(c) if c in ORDER else len(ORDER), c)
    return [files[c][-1][1] for c in sorted(files, key=rank)], warnings


def scores(r: dict, tag: str | None = None) -> dict | None:
    """Recall@1/5 and MRR@10 with their 95% intervals ("ci95"), and p95 latency, over the result's questions or
    those with `tag`; None if there are none."""
    rows = [q for q in r["questions"] if tag is None or tag in q.get("tags", [])]
    if not rows:
        return None
    ranks = [q["rank"] for q in rows]
    return {"recall@1": m.recall_at(ranks, 1), "recall@5": m.recall_at(ranks, 5), "mrr@10": m.mrr_at(ranks, 10),
            "latency_p95_ms": m.percentile([q["latency_ms"] for q in rows], 95), "ci95": stats.intervals(ranks)}


def hardware(h: dict | None) -> str:
    if not h:
        return "unknown hardware"
    cores = f"{h['cores']} cores" if h.get("cores") else None
    return ", ".join(str(x) for x in [h.get("os"), h.get("cpu"), cores, h.get("gpu")] if x) or "unknown hardware"


def _recall1(r: dict, tag: str) -> str:
    s = scores(r, tag)
    return _ci(s, "recall@1") if s else "n/a"


def _values(results: list[dict], get) -> list[str]:
    """The distinct values across the files, in first-seen order."""
    return list(dict.fromkeys(get(r) for r in results))


def table(results: list[dict], warnings=()) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in results:
        s = scores(r)
        lines.append(f"| {r['config']} | {_ci(s, 'recall@1')} | {_ci(s, 'recall@5')} | {_ci(s, 'mrr@10')} "
                     f"| {_recall1(r, 'exact')} | {_recall1(r, 'cross-language')} | {_ms(s['latency_p95_ms'])} |")
    warnings = list(warnings)
    for what, get in [("commit", lambda r: r["commit"] or "unknown"),
                      ("hardware", lambda r: hardware(r["setup"].get("hardware")))]:
        if len(found := _values(results, get)) > 1:
            warnings.append(f"the files disagree on {what}: {'; '.join(found)}")
    lines += [f"| ⚠ {w} |" + " |" * (len(COLUMNS) - 1) for w in warnings]

    rerankers = [f"`{r['setup']['reranker']}` ({r['config']})" for r in results
                 if r["setup"].get("reranker", "off") != "off"]
    sizes = _values(results, lambda r: f"{r['dataset']['questions']} questions over "
                                       f"{len(r['dataset']['documents'])} PDFs")
    facts = [
        "; ".join(_values(results, lambda r: r["date"])),
        "commit " + "; ".join(f"`{c}`" for c in _values(results, lambda r: r["commit"] or "unknown")),
        "; ".join(_values(results, lambda r: hardware(r["setup"].get("hardware")))),
        "embeddings " + "; ".join(f"`{e}`" for e in _values(results, lambda r: r["setup"]["embeddings"])),
        "reranker " + ("; ".join(rerankers) or "none"),
        "; ".join(sizes),
        "95% bootstrap intervals in brackets",
    ]
    return "\n".join(lines) + "\n\n" + " · ".join(facts) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the eval results as one Markdown table.")
    ap.add_argument("--dir", type=Path, default=RESULTS, help="results folder (default: evals/results)")
    args = ap.parse_args(argv)
    results, warnings = load(args.dir)
    if not results:
        raise SystemExit(f"No results in {args.dir}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # "·" and "⚠" in a cp1252 console or a redirect on Windows
    print(table(results, warnings), end="")


if __name__ == "__main__":
    main()
