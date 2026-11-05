"""Measure retrieval quality: index the eval PDFs into a fresh collection, ask every question, and report
Recall@1/5/10, MRR@10, latency and LLM calls per query.

  python -m evals.run --config vector --settings data/config.json
  python -m evals.run --config vector-rewrite --settings data/config.json --answers

--settings is a RAG Studio config.json (the one the Setup page saves in DATA_DIR) that says which embedding
model, LLM and vector store to use. The named config in evals/configs/ is merged over it. The run works in
a temporary DATA_DIR with its own metadata database and a new collection (eval_<config>_<time>), which
is dropped afterwards unless --keep is given, so the app's own documents and settings are never touched.

--answers also generates an answer per question with the app's prompt and scores it:
  faithfulness       share of the answer's claims that the retrieved pages support, judged by the
                     configured LLM with the fixed prompt in evals/metrics.py
  citation accuracy  share of quoted passages cited as [n] that really are on page n

Results go to evals/results/<date>-<config>.json and .md. Nothing is written if a step fails.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
CONFIGS = HERE / "configs"
RESULTS = HERE / "results"
K = 10


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(name: str) -> dict:
    path = CONFIGS / f"{name}.json"
    if not path.exists():
        names = ", ".join(sorted(p.stem for p in CONFIGS.glob("*.json")))
        raise SystemExit(f"No eval config {name!r} in {CONFIGS} (have: {names})")
    return json.loads(path.read_text())


class CountingLLM:
    """Wraps the configured LLM to count complete() calls."""
    calls = 0

    def __init__(self, inner):
        self.inner = inner

    def complete(self, system, user):
        CountingLLM.calls += 1
        return self.inner.complete(system, user)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate RAG Studio retrieval on the labeled dataset.")
    ap.add_argument("--config", required=True, help="name of a file in evals/configs/, without .json")
    ap.add_argument("--settings", type=Path, default=Path(os.environ.get("DATA_DIR", "data")) / "config.json",
                    help="RAG Studio config.json with the providers to use (default: $DATA_DIR/config.json)")
    ap.add_argument("--answers", action="store_true", help="also generate answers and score faithfulness")
    ap.add_argument("--limit", type=int, help="only the first N questions (for a quick check)")
    ap.add_argument("--keep", action="store_true", help="keep the eval collection and temporary DATA_DIR")
    ap.add_argument("--out", type=Path, default=RESULTS, help="results folder (default: evals/results)")
    args = ap.parse_args(argv)

    if not args.settings.exists():
        raise SystemExit(f"Settings file {args.settings} not found: pass --settings with a RAG Studio config.json")
    overrides = load_config(args.config)
    settings = json.loads(args.settings.read_text())

    # app.config reads DATA_DIR at import, so point it at a scratch folder first.
    work = Path(tempfile.mkdtemp(prefix="rag-eval-"))
    os.environ["DATA_DIR"] = str(work)
    os.environ.setdefault("IMPORT_DIR", str(work / "import"))
    try:
        result = run(args, overrides, settings, work)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept {work}")
    stem = f"{date.today().isoformat()}-{args.config}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (args.out / f"{stem}.md").write_text(markdown(result))
    print(markdown(result))
    print(f"wrote {args.out / stem}.json and .md")


def run(args, overrides, settings, work: Path) -> dict:
    from app import config, db, rag, stores
    from app.providers import llm
    from . import dataset, metrics

    everything = dataset.load()
    questions = everything[:args.limit] if args.limit else everything
    stamp = time.strftime("%Y%m%d%H%M%S")
    cfg = _merge(config.DEFAULTS, _merge(settings, {k: v for k, v in overrides.items() if k != "description"}))
    cfg["configured"] = True
    cfg["vector_store"]["collection"] = f"eval_{args.config.replace('-', '_')}_{stamp}"
    cfg["vector_store"]["chroma"]["path"] = str(work / "chroma")  # a relative or container path won't do here
    config.save(cfg)
    db.init()

    real_get = llm.get
    llm.get = lambda c: CountingLLM(real_get(c))  # rag calls llm.get(cfg) per operation

    store = stores.get(cfg)
    try:
        ingest_s = _ingest(everything, config, db, rag, dataset)  # all of them, even with --limit
        rewrite = bool(cfg["rag"]["query_rewrite"])
        rag.search("warm-up", K, rewrite_question=False)  # loads a local embedding model outside the timings
        rows = []
        for q in questions:
            CountingLLM.calls = 0
            t0 = time.perf_counter()
            _, queries, hits = rag.search(q["question"], K, rewrite_question=rewrite)
            ms = (time.perf_counter() - t0) * 1000
            ranking = [(h["filename"], h["page"]) for h in hits]
            rows.append({"id": q["id"], "tags": q.get("tags", []), "rank": metrics.first_hit(ranking, q["doc"], q["pages"]),
                         "latency_ms": round(ms, 1), "llm_calls": CountingLLM.calls, "queries": queries,
                         "top": [f"{d} p{p}" for d, p in ranking[:5]]})
        answers = _answers(questions, cfg, rag, llm, dataset, metrics) if args.answers else None
    finally:
        llm.get = real_get
        if not args.keep:
            _drop(store)

    by_tag = {tag: metrics.summarize([r for r in rows if tag in r["tags"]])
              for tag in sorted({t for r in rows for t in r["tags"]})}
    if untagged := [r for r in rows if not r["tags"]]:
        by_tag["untagged"] = metrics.summarize(untagged)
    return {
        "date": date.today().isoformat(),
        "config": args.config,
        "description": overrides.get("description", ""),
        "commit": _commit(),
        "setup": {
            "vector_store": cfg["vector_store"]["type"],
            "embeddings": f"{cfg['embeddings']['provider']}:{cfg['embeddings'][cfg['embeddings']['provider']]['model']}",
            "llm": f"{cfg['llm']['provider']}:{cfg['llm'][cfg['llm']['provider']]['model']}",
            "chunk_tokens": cfg["rag"]["chunk_tokens"], "chunk_overlap_tokens": cfg["rag"]["chunk_overlap_tokens"],
            "query_rewrite": bool(cfg["rag"]["query_rewrite"]), "k": K,
        },
        "dataset": {"questions": len(questions), "documents": sorted({q["doc"] for q in everything}),
                    "ingest_seconds": round(ingest_s, 1)},
        "retrieval": metrics.summarize(rows),
        "by_tag": by_tag,
        "answers": answers and answers["summary"],
        "questions": [{**r, **({"answer": answers["rows"][i]} if answers else {})} for i, r in enumerate(rows)],
    }


def _ingest(questions, config, db, rag, dataset) -> float:
    """Register and index each document synchronously, as the app's worker would."""
    t0 = time.time()
    for doc in sorted({q["doc"] for q in questions}):
        doc_id = uuid.uuid4().hex
        shutil.copyfile(dataset.PDF_DIR / doc, config.UPLOAD_DIR / f"{doc_id}.pdf")
        now = time.time()
        db.execute("INSERT INTO documents (id, filename, size, status, created, updated) VALUES (?,?,?,?,?,?)",
                   (doc_id, doc, (dataset.PDF_DIR / doc).stat().st_size, "queued", now, now))
        rag.ingest(doc_id)
        row = db.rows("SELECT status, error, chunks, language FROM documents WHERE id=?", (doc_id,))[0]
        if row["status"] != "ready":
            raise SystemExit(f"Indexing {doc} failed: {row['error'] or row['status']}")
        print(f"indexed {doc}: {row['chunks']} chunks, language {row['language']}")
    return time.time() - t0


def _answers(questions, cfg, rag, llm, dataset, metrics) -> dict:
    """Answer each question with the app's own pipeline, then judge it against the retrieved pages."""
    judge = llm.get(cfg)
    rows = []
    for q in questions:
        CountingLLM.calls = 0
        t0 = time.perf_counter()
        r = rag.answer(q["question"])
        ms = (time.perf_counter() - t0) * 1000
        calls = CountingLLM.calls
        pages = [(h["filename"], h["page"], dataset.page_texts(h["filename"])[h["page"] - 1]) for h in r["sources"]]
        verdict = metrics.parse_judgement(judge.complete(metrics.JUDGE_PROMPT, metrics.judge_input(r["answer"], pages)))
        cites = metrics.citations(r["answer"], [text for _, _, text in pages])
        rows.append({"text": r["answer"], "latency_ms": round(ms, 1), "llm_calls": calls,
                     "supported": verdict[0] if verdict else None, "claims": verdict[1] if verdict else None,
                     "citations": cites})
    summary = metrics.summarize_answers(rows)
    summary["latency_p50_ms"] = metrics._round(metrics.percentile([r["latency_ms"] for r in rows], 50))
    summary["latency_p95_ms"] = metrics._round(metrics.percentile([r["latency_ms"] for r in rows], 95))
    summary["llm_calls_per_answer"] = round(sum(r["llm_calls"] for r in rows) / len(rows), 3) if rows else 0.0
    return {"summary": summary, "rows": rows}


def _drop(store):
    """Remove the eval collection. The store interface has no drop, so this knows the three backends."""
    try:
        if hasattr(store, "table"):  # pgvector
            with store._conn() as c:
                c.execute(f"DROP TABLE IF EXISTS {store.table}")
        elif hasattr(store, "client") and hasattr(store, "collection"):  # Qdrant
            if store.client.collection_exists(store.collection):
                store.client.delete_collection(store.collection)
        # Chroma lives in the temporary DATA_DIR, which is removed with it.
    except Exception as e:  # noqa: BLE001 - the results are still good
        print(f"warning: could not drop the eval collection: {e}", file=sys.stderr)


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _ms(x):
    return "n/a" if x is None else f"{x:.0f} ms"


def markdown(r: dict) -> str:
    s = r["setup"]
    lines = [
        f"# Retrieval eval: {r['config']} ({r['date']})", "",
        r["description"] or "", "",
        f"Commit `{r['commit']}` · store {s['vector_store']} · embeddings `{s['embeddings']}` · LLM `{s['llm']}` · "
        f"chunks {s['chunk_tokens']}/{s['chunk_overlap_tokens']} tokens · rewrite {'on' if s['query_rewrite'] else 'off'} · "
        f"{r['dataset']['questions']} questions over {len(r['dataset']['documents'])} PDFs", "",
        "| Slice | Questions | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 | p95 | LLM calls/query |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in [("all", r["retrieval"]), *r["by_tag"].items()]:
        lines.append(f"| {name} | {m['questions']} | {_pct(m['recall@1'])} | {_pct(m['recall@5'])} | {_pct(m['recall@10'])} "
                     f"| {m['mrr@10']:.3f} | {_ms(m['latency_p50_ms'])} | {_ms(m['latency_p95_ms'])} | {m['llm_calls_per_query']} |")
    if r.get("answers"):
        a = r["answers"]
        lines += ["", "| Answers | Faithfulness | Claims | Judge failures | Citations/answer | Valid citations "
                      "| Cited quotes | Citation accuracy | p50 | p95 | LLM calls/answer |",
                  "|---|---|---|---|---|---|---|---|---|---|---|",
                  f"| {a['answers']} | {_pct(a['faithfulness'])} | {a['claims']} | {a['judge_failures']} | "
                  f"{a['citations_per_answer']} | {_pct(a['citation_validity'])} | {a['cited_quotes']} | "
                  f"{_pct(a['citation_accuracy'])} | {_ms(a['latency_p50_ms'])} | {_ms(a['latency_p95_ms'])} | "
                  f"{a['llm_calls_per_answer']} |"]
    missed = [q for q in r["questions"] if q["rank"] is None or q["rank"] > 5]
    if missed:
        lines += ["", "Not in the top 5:", ""]
        lines += [f"- `{q['id']}` (rank {q['rank'] or '>10'}): got {', '.join(q['top']) or 'nothing'}" for q in missed]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
