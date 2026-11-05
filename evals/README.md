# Retrieval evals

A small labeled dataset and a harness that measures how well RAG Studio finds the right page, so changes to
retrieval (hybrid search, reranking, chunking) can be compared with numbers instead of impressions.

```bash
python -m evals.run --config vector --settings data/config.json            # current vector search
python -m evals.run --config vector-rewrite --settings data/config.json    # with the LLM query rewrite
python -m evals.run --config hybrid --settings data/config.json            # vector + keyword, fused by RRF
python -m evals.run --config hybrid-rewrite --settings data/config.json    # hybrid with the query rewrite
python -m evals.run --config vector --settings data/config.json --answers  # also score generated answers
python -m evals.report                                                     # all results as one table, for the README
python -m evals.compare results/<a>.json results/<b>.json                  # B minus A, paired, with CI and p-value
python -m pytest tests/test_evals.py -q                                    # metric tests and label checks
```

`--settings` is a RAG Studio `config.json`, the file the Setup page saves in `DATA_DIR`. It decides the embedding
model, LLM and vector store. The named config in `configs/` is merged over it. Each run uses a temporary
`DATA_DIR` and a new collection named `eval_<config>_<time>`, which is dropped at the end unless you pass `--keep`, so
the app's own documents and settings are never touched. For a Docker install, copy the settings out of the
volume first (`docker compose cp app:/data/config.json /tmp/rag-settings.json`). A pgvector or Qdrant address
like `postgres:5432` has to be reachable from where you run the eval, so either run it inside the container
or edit the copy to point at `localhost`.

Results are written to `results/<date>-<config>.json`, with every question's rank, latency and top pages, and to a
Markdown table beside it (`results/<date>-<config>.md`). A second run of the same config on the same day
adds the time to the name instead of replacing the first.

## What is measured

- **Recall@1/5/10**: the share of questions with an expected page among the first 1, 5 or 10 pages retrieved.
  Any one of a question's expected pages counts as a hit. Retrieval runs once per question with k = 10, and
  Recall@1 and @5 are read from that same ranking.
- **MRR@10**: the mean of 1/rank of the first expected page, or 0 when none is in the top 10.
- **Latency p50/p95**: wall time of `rag.search` per question (rewrite, embedding and store queries). Indexing
  and one warm-up query, which loads a local embedding model, are not included.
- **LLM calls per query**: calls made during retrieval. That's 0 without the rewrite and 1 with it.

Every number is also broken down by tag: `exact` (codes, numbers and names such as `a52082`, `dvfs=2`,
`IEEE 802.3at-2009`) and `cross-language` (an English question whose answer is in the Spanish document).

### Intervals and comparing configs

Each metric is a mean of per-question values (hit@1, hit@5, reciprocal rank; stored on every question from this
version on, and recomputed from the ranks for older files). `stats.py` resamples the questions with replacement 10,000
times (seed 0) and takes the middle 95% of the resampled means as the interval: `ci95` in each summary, in brackets
in `evals.report`. `evals.compare` pairs two runs question by question: it resamples the per-question differences, for
the interval of the mean difference and a two-sided p-value (the resampled means shifted to centre on zero), and adds
how many questions B wins and loses with the exact sign test. It refuses results whose dataset fingerprints
(`dataset.sha256`, the questions and the PDFs) differ, and matches older files without one by question id.

Pairing matters with 66 questions: each config's own interval is wide, so two configs' intervals overlap even when
one wins on several questions and loses on none. Most questions are easy or hard for every config, so the differences
vary much less than the scores, and the paired test sees a change the overlapping intervals hide. When only a few
questions move, all in one direction, the bootstrap interval can exclude zero while its p is above 0.05; the sign test
is exact there. The p-values are not corrected for the many slices and metrics compared.

With `--answers`, each question is also answered by `rag.answer` (the app's prompt and `top_k`) and scored:

- **Faithfulness**: the share of the answer's claims that the retrieved pages support. The configured LLM
  judges this with the fixed prompt `JUDGE_PROMPT` in `metrics.py`, and it sees exactly the passages the
  answering model was given. The score is supported claims divided by all claims across the run. Answers whose judgement
  can't be parsed are counted as `judge_failures` and left out.
- **Citation validity**: the share of cited source numbers that exist. Grouped markers such as `[1, 3]` or `[2-4]`
  count each number.
- **Citation accuracy**: the share of quoted passages cited as `"…" [n]` that really appear on page n,
  ignoring case and spacing. The app's prompt doesn't ask for quotes, so `cited_quotes` shows how many there were
  to check, and the accuracy is `n/a` when there were none.

The judge is the same model as the one answering, so faithfulness is a relative measure for comparing runs, not
an absolute one.

## The dataset

`dataset.jsonl` holds 66 questions over 4 PDFs: 27 tagged `exact` and 19 `cross-language`. Each line has:

| Field | Meaning |
|---|---|
| `id` | Stable name, such as `rpi-01` |
| `question` | What a user would ask |
| `doc`, `pages` | The PDF in `pdfs/`, and the 1-based pages that answer the question |
| `evidence` | Verbatim snippets from those pages. `pages` is exactly the set of pages containing one of them. |
| `answer` | A short reference answer, for reading the reports |
| `tags` | `exact`, `cross-language`, or neither |

`tests/test_evals.py` extracts the PDFs with the app's own `pdftext` and checks every label against them, so a
label can't silently point at the wrong page. To add a question, add a line with its evidence snippet, then run
`python -c "from evals.dataset import pages_with; print(pages_with('<doc>', ['<snippet>']))"` to get its pages.

## The documents

The sandbox where this was built could only reach GitHub, so the PDFs are built from the published source text
of each document, at a pinned commit, by `build_pdfs.py`. They are plain text PDFs with a running header and footer.
They don't have the layout, tables and images of the originals, so they test retrieval more than PDF parsing.
`python -m evals.build_pdfs` rebuilds them byte for byte from the same sources.

| PDF | Pages | Language | Source | License |
|---|---|---|---|---|
| `nist-sp800-63b.pdf` | 29 | English | NIST SP 800-63B (2017, with errata), Digital Identity Guidelines: Authentication and Lifecycle Management, sections 4, 5, 7 and appendix A, from [usnistgov/800-63-3](https://github.com/usnistgov/800-63-3) at `7dee3ca` | US government work, not subject to copyright in the United States. NIST grants a worldwide right to reprint and make derivative works. Republished courtesy of the National Institute of Standards and Technology. |
| `pep8-style-guide.pdf` | 27 | English | PEP 8, Style Guide for Python Code (van Rossum, Warsaw, Coghlan), from [python/peps](https://github.com/python/peps) at `28581cc` | Public domain, as stated in the PEP |
| `raspberry-pi-hardware.pdf` | 16 | English | Raspberry Pi documentation: power supply, revision codes, OTP bits, frequency management, RTC and power button chapters, from [raspberrypi/documentation](https://github.com/raspberrypi/documentation) at `5307424` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), © Raspberry Pi Ltd |
| `mdn-http-es.pdf` | 42 | Spanish | MDN Web Docs in Spanish: 22 HTTP status code pages and 10 header pages (Cache-Control, Strict-Transport-Security, Content-Security-Policy, Set-Cookie, X-Frame-Options, ETag, Content-Disposition, X-Content-Type-Options, Referrer-Policy, Access-Control-Allow-Origin), from [mdn/translated-content](https://github.com/mdn/translated-content) at `0862a7d` | [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/), © Mozilla Contributors |

The PDFs are adapted works: the source markup was converted to plain text, images and macros were removed, and
tables were flattened to one row per line. The Raspberry Pi and MDN PDFs are shared under the same licenses as
their sources (CC BY-SA 4.0 and CC BY-SA 2.5). The evidence snippets in `dataset.jsonl` are short quotations
from these documents, under the same terms.

## Limitations

The set is small (66 questions, 4 documents), and the questions and labels were written for this project rather
than collected from real users, and haven't been reviewed by a second person. Treat a difference of a few points between configs as noise. The
labels mark the pages that hold the key sentence of each answer. A page that also answers a question but doesn't
contain an evidence snippet counts as a miss. Each document covers a different topic, so picking the right
document is easier here than in a real collection of similar manuals.
