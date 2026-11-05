import json

import pytest

from evals import report, stats
from evals.run import _pct

HW = {"os": "Linux-6", "cpu": "Test CPU", "cores": 8, "gpu": None, "onnx_providers": ["CPUExecutionProvider"]}
# 8 questions: ranks 1, 2, 4, miss over exact ones, then 1, 1, 7, 3 (two of them cross-language).
QUESTIONS = [(1, ["exact"]), (2, ["exact"]), (4, ["exact"]), (None, ["exact"]),
             (1, ["cross-language"]), (1, []), (7, ["cross-language"]), (3, [])]


def result(config, commit="abc1234", hardware=HW, reranker="off", questions=QUESTIONS):
    """A small result file in the shape evals.run writes, with only the fields the report reads."""
    return {"date": "2026-11-05", "config": config, "commit": commit,
            "setup": {"embeddings": "local:emb", "reranker": reranker, "hardware": hardware},
            "dataset": {"questions": len(questions), "documents": ["a.pdf", "b.pdf"]},
            "questions": [{"id": f"q{i}", "tags": tags, "rank": rank, "latency_ms": 10.0 + i}
                          for i, (rank, tags) in enumerate(questions)]}


@pytest.fixture
def results_dir(tmp_path):
    def write(*files):
        for name, r in files:
            (tmp_path / name).write_text(json.dumps(r), encoding="utf-8")
        return tmp_path
    return write


def test_one_row_per_config_in_pipeline_order(results_dir):
    folder = results_dir(("2026-11-05-zeta.json", result("zeta")), ("2026-11-05-hybrid.json", result("hybrid")),
                         ("2026-11-05-vector.json", result("vector")))
    out = report.table(*report.load(folder))
    rows = [line for line in out.splitlines() if line.startswith("| ") and "---" not in line]
    assert rows[0] == "| " + " | ".join(report.COLUMNS) + " |"
    # Recall@1 3/8, Recall@5 6/8, MRR (1 + 1/2 + 1/4 + 0 + 1 + 1 + 1/7 + 1/3) / 8, exact 1/4, cross-language 1/2,
    # p95 of latencies 10..17 = 16.65; each with the bootstrap interval of its per-question values.
    hits1, hits5 = [1, 0, 0, 0, 1, 1, 0, 0], [1, 1, 1, 0, 1, 1, 0, 1]
    rr = [1, 1 / 2, 1 / 4, 0, 1, 1, 1 / 7, 1 / 3]
    pct = lambda ci: f"[{ci[0] * 100:.1f}–{ci[1] * 100:.1f}]"
    mrr = lambda ci: f"[{ci[0]:.3f}–{ci[1]:.3f}]"
    assert rows[1] == (f"| vector | 37.5% {pct(stats.bootstrap_ci(hits1))} | 75.0% {pct(stats.bootstrap_ci(hits5))} "
                       f"| 0.528 {mrr(stats.bootstrap_ci(rr))} | 25.0% {pct(stats.bootstrap_ci([1, 0, 0, 0]))} "
                       f"| 50.0% {pct(stats.bootstrap_ci([1, 0]))} | 17 ms |")
    assert [r.split(" | ")[0] for r in rows[2:]] == ["| hybrid", "| zeta"]  # known configs first, then by name
    assert "⚠" not in out


def test_scores_come_from_the_ranks_not_the_rounded_summary():
    # 56 of 66 at rank 1 is 84.848...%: the summary's 0.8485 rounded again would print 84.9%.
    r = result("hybrid-rerank", questions=[(1, [])] * 56 + [(2, [])] * 10)
    r["retrieval"] = {"recall@1": 0.8485}
    assert _pct(report.scores(r)["recall@1"]) == "84.8%"
    assert report.scores(r, "exact") is None


def test_the_line_under_the_table_comes_from_the_files(results_dir):
    folder = results_dir(("2026-11-05-vector.json", result("vector")),
                         ("2026-11-05-hybrid-rerank.json", result("hybrid-rerank", reranker="local:rr top 30")))
    line = report.table(*report.load(folder)).splitlines()[-1]
    assert line == ("2026-11-05 · commit `abc1234` · Linux-6, Test CPU, 8 cores · embeddings `local:emb` · "
                    "reranker `local:rr top 30` (hybrid-rerank) · 8 questions over 2 PDFs · "
                    "95% bootstrap intervals in brackets")


def test_a_warning_row_when_commit_or_hardware_differ(results_dir):
    folder = results_dir(("2026-11-05-vector.json", result("vector")),
                         ("2026-11-05-hybrid.json", result("hybrid", commit="def5678", hardware=None)))
    out = report.table(*report.load(folder))
    warnings = [line for line in out.splitlines() if line.startswith("| ⚠")]
    assert warnings == [
        "| ⚠ the files disagree on commit: abc1234; def5678 |" + " |" * 6,
        "| ⚠ the files disagree on hardware: Linux-6, Test CPU, 8 cores; unknown hardware |" + " |" * 6,
    ]
    assert "commit `abc1234`; `def5678`" in out.splitlines()[-1]


def test_hardware_leaves_out_missing_fields():
    assert report.hardware({"os": "Linux-6", "cpu": "X", "gpu": None}) == "Linux-6, X"
    assert report.hardware({}) == report.hardware(None) == "unknown hardware"


def test_the_newest_file_of_a_config_wins_with_one_warning(results_dir):
    folder = results_dir(("2026-11-05-vector.json", result("vector", questions=[(None, [])])),
                         ("2026-11-05-vector-213000.json", result("vector", questions=[(2, [])])),
                         ("2026-11-05-vector-090000.json", result("vector", questions=[(3, [])])))
    results, warnings = report.load(folder)
    assert [q["rank"] for q in results[0]["questions"]] == [2]
    assert warnings == ["3 results for vector, using 2026-11-05-vector-213000.json"]


def test_a_missing_slice_reads_na(results_dir):
    out = report.table(*report.load(results_dir(("2026-11-05-vector.json", result("vector", questions=[(1, [])])))))
    assert "| vector | 100.0% [100.0–100.0] | 100.0% [100.0–100.0] | 1.000 [1.000–1.000] | n/a | n/a | 10 ms |" in out


def test_main_prints_the_table_and_refuses_an_empty_folder(results_dir, tmp_path, capsys):
    report.main(["--dir", str(results_dir(("2026-11-05-vector.json", result("vector"))))])
    assert capsys.readouterr().out.startswith("| config | Recall@1 |")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="No results"):
        report.main(["--dir", str(empty)])


def test_the_committed_results_agree():
    """The README's table is generated from evals/results: those files must be one consistent run."""
    results, warnings = report.load(report.RESULTS)
    out = report.table(results)
    assert results and not warnings and "⚠" not in out, (
        "evals/results has several files per config or mixes commits/hardware: keep one run per config")


def test_the_readme_table_is_the_report_output():
    readme = (report.RESULTS.parents[1] / "README.md").read_text(encoding="utf-8")
    assert report.table(*report.load(report.RESULTS)) in readme, (
        "README's Results table is stale: paste the output of `python -m evals.report`")
