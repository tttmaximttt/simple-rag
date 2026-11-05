import pytest

from evals import dataset, metrics, run


# --- retrieval metrics ----------------------------------------------------------------------------------------

def test_first_hit_is_the_rank_of_the_first_expected_page():
    ranking = [("a.pdf", 3), ("b.pdf", 2), ("a.pdf", 2), ("a.pdf", 5)]
    assert metrics.first_hit(ranking, "a.pdf", [2, 5]) == 3
    assert metrics.first_hit(ranking, "a.pdf", [3]) == 1
    assert metrics.first_hit(ranking, "b.pdf", [3]) is None  # right page number, wrong document
    assert metrics.first_hit([], "a.pdf", [1]) is None
    assert metrics.first_hit([["a.pdf", 1]], "a.pdf", [1]) == 1  # rankings read back from JSON are lists


def test_recall_counts_questions_with_a_hit_in_the_top_k():
    ranks = [1, 3, None, 7]
    assert metrics.recall_at(ranks, 1) == 0.25
    assert metrics.recall_at(ranks, 5) == 0.5
    assert metrics.recall_at(ranks, 10) == 0.75
    assert metrics.recall_at([], 5) == 0.0


def test_mrr_ignores_hits_below_the_cutoff():
    assert metrics.mrr_at([1, 2, None, 4], 10) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert metrics.mrr_at([1, 11], 10) == 0.5
    assert metrics.mrr_at([None], 10) == 0.0


def test_percentile_interpolates_like_numpy():
    values = [40, 10, 30, 20]  # sorted: 10 20 30 40
    assert metrics.percentile(values, 50) == 25
    assert metrics.percentile(values, 95) == pytest.approx(38.5)
    assert metrics.percentile(values, 0) == 10 and metrics.percentile(values, 100) == 40
    assert metrics.percentile([7], 95) == 7
    assert metrics.percentile([], 50) is None


def test_summarize():
    rows = [{"rank": 1, "latency_ms": 10, "llm_calls": 1}, {"rank": 6, "latency_ms": 30, "llm_calls": 1},
            {"rank": None, "latency_ms": 20, "llm_calls": 0}, {"rank": 2, "latency_ms": 40, "llm_calls": 2}]
    s = metrics.summarize(rows)
    assert s["questions"] == 4
    assert (s["recall@1"], s["recall@5"], s["recall@10"]) == (0.25, 0.5, 0.75)
    assert s["mrr@10"] == round((1 + 1 / 6 + 0 + 0.5) / 4, 4)
    assert s["latency_p50_ms"] == 25 and s["llm_calls_per_query"] == 1.0


# --- answer metrics -------------------------------------------------------------------------------------------

def test_citations_check_quotes_against_the_cited_page():
    sources = ["Limit all lines to a maximum of\n79 characters.", "Use 4 spaces per indentation level."]
    answer = ('PEP 8 says to "limit all lines to a maximum of 79 characters" [1] and to "use 4 spaces per '
              'indentation level" [1]. Tabs are discussed too [2], see also [3].')
    c = metrics.citations(answer, sources)
    # The second quote is real, but not on the page it cites; [3] doesn't exist.
    assert c == {"citations": 4, "valid": 3, "quotes": 2, "quotes_found": 1}


def test_citations_accept_typographic_quotes_and_punctuation():
    c = metrics.citations("It says “use 4 spaces per indentation level”, [1]", ["Use 4 spaces per indentation level."])
    assert c == {"citations": 1, "valid": 1, "quotes": 1, "quotes_found": 1}
    assert metrics.citations("", []) == {"citations": 0, "valid": 0, "quotes": 0, "quotes_found": 0}


def test_citations_count_grouped_markers():
    c = metrics.citations('Both say so [1, 3], see [2-4] and [5–5]. "use 4 spaces per level" [2, 3]',
                          ["a", "use 4 spaces per level", "c", "d"])
    assert c == {"citations": 8, "valid": 7, "quotes": 1, "quotes_found": 1}  # 1,3 · 2,3,4 · 5 (no such source) · 2,3


def test_parse_judgement():
    reply = 'Here you go:\n```json\n{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false},' \
            ' {"claim": "c", "supported": "yes"}]}\n```'
    assert metrics.parse_judgement(reply) == (1, 3)  # only a literal true counts
    assert metrics.parse_judgement('{"claims": []}') == (0, 0)
    for bad in ("I think it is fine.", '{"claims": "all"}', "{not json}", None):
        assert metrics.parse_judgement(bad) is None


def test_summarize_answers_micro_averages_and_skips_unjudged():
    none = {"citations": 0, "valid": 0, "quotes": 0, "quotes_found": 0}
    rows = [{"supported": 3, "claims": 4, "citations": {"citations": 2, "valid": 2, "quotes": 1, "quotes_found": 1}},
            {"supported": 1, "claims": 1, "citations": {"citations": 1, "valid": 0, "quotes": 1, "quotes_found": 0}},
            {"supported": None, "claims": None, "citations": none}]
    s = metrics.summarize_answers(rows)
    assert s["faithfulness"] == 0.8 and s["claims"] == 5 and s["judge_failures"] == 1
    assert s["citations_per_answer"] == 1.0 and s["citation_validity"] == round(2 / 3, 4)
    assert s["cited_quotes"] == 2 and s["citation_accuracy"] == 0.5
    assert metrics.summarize_answers([{"supported": None, "claims": None, "citations": none}])["faithfulness"] is None


def test_judge_input_numbers_the_pages():
    text = metrics.judge_input("It is 79 [1].", [("pep8.pdf", 4, "Limit all lines to 79 characters.")])
    assert text.startswith("Source pages:\n[1] (pep8.pdf, page 4)\nLimit all lines") and text.endswith("Answer:\nIt is 79 [1].")


# --- the dataset ----------------------------------------------------------------------------------------------

def test_dataset_labels_match_the_pdfs():
    """Every label's pages are exactly the pages of its PDF that contain one of its evidence snippets."""
    items = dataset.load()
    assert 40 <= len(items) <= 80
    assert sum("exact" in q["tags"] for q in items) >= 10
    assert sum("cross-language" in q["tags"] for q in items) >= 10
    for q in items:
        assert q["question"] and q["evidence"] and (dataset.PDF_DIR / q["doc"]).exists(), q["id"]
        for snippet in q["evidence"]:
            assert dataset.pages_with(q["doc"], [snippet]), f"{q['id']}: evidence not found: {snippet!r}"
        assert dataset.pages_with(q["doc"], q["evidence"]) == q["pages"], q["id"]


def test_cross_language_questions_point_at_a_spanish_document():
    from app import pdftext
    for doc in {q["doc"] for q in dataset.load() if "cross-language" in q["tags"]}:
        assert pdftext.detect_language(" ".join(dataset.page_texts(doc))) == "Spanish"


def test_eval_configs_load():
    assert run.load_config("vector")["rag"]["query_rewrite"] is False
    assert run.load_config("vector-rewrite")["rag"]["query_rewrite"] is True
    with pytest.raises(SystemExit):
        run.load_config("nope")


def test_run_refuses_to_start_with_the_app_already_imported(tmp_path):
    # tests/conftest.py imported app, so its DATA_DIR is fixed: the eval must not write into it.
    settings = tmp_path / "config.json"
    settings.write_text("{}")
    with pytest.raises(SystemExit, match="fresh process"):
        run.main(["--config", "vector", "--settings", str(settings)])
    with pytest.raises(SystemExit):
        run.main(["--config", "vector", "--settings", str(settings), "--limit", "0"])
