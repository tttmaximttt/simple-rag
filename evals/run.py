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
import copy
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


def load_config(name: str) -> dict:
    path = CONFIGS / f"{name}.json"
    if not path.exists():
        names = ", ".join(sorted(p.stem for p in CONFIGS.glob("*.json")))
        raise SystemExit(f"No eval config {name!r} in {CONFIGS} (have: {names})")
    return json.loads(path.read_text(encoding="utf-8"))


class CountingLLM:
    """Wraps the configured LLM to count complete() calls and record the ones that fail. rag.rewrite swallows a
    failure and searches with the question alone, which would quietly turn a rewrite run into a no-rewrite one."""
    calls = 0
    failures: list[str] = []

    def __init__(self, inner):
        self.inner = inner

    def complete(self, system, user):
        CountingLLM.calls += 1
        try:
            reply = self.inner.complete(system, user)
        except Exception as e:
            CountingLLM.failures.append(f"{type(e).__name__}: {e}")
            raise
        if not isinstance(reply, str) or not reply.strip():
            CountingLLM.failures.append(f"empty reply {reply!r}")
        return reply

    @classmethod
    def reset(cls):
        cls.calls, cls.failures = 0, []

    @classmethod
    def check(cls, qid: str):
        """Stop the run at the first failed call, with its error: the event log goes with the temporary DATA_DIR."""
        if cls.failures:
            raise SystemExit(f"LLM call failed for {qid}, so the results would mix configs: {cls.failures[0]}")

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
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")
    if "app.config" in sys.modules:
        # DATA_DIR is read when app.config is imported: running now would use the app's real data folder.
        raise SystemExit("evals.run must start in a fresh process (app is already imported)")

    if not args.settings.exists():
        raise SystemExit(f"Settings file {args.settings} not found: pass --settings with a RAG Studio config.json")
    overrides = load_config(args.config)
    settings = json.loads(args.settings.read_text(encoding="utf-8"))

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
    path = write(result, args.out, args.config)
    print(markdown(result))
    print(f"wrote {path}.json and .md")


def write(result: dict, out: Path, config: str) -> Path:
    """Write <date>-<config>.json and .md to out and return their path without the suffix. The bytes are the same
    on every OS: always UTF-8 (cp1252 on Windows would change the Markdown's "·") with LF line endings."""
    stem = f"{date.today().isoformat()}-{config}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{stem}.json").exists():  # a second run the same day mustn't replace the first
        stem += time.strftime("-%H%M%S")
    text = {"json": json.dumps(result, indent=2, ensure_ascii=False) + "\n", "md": markdown(result)}
    for suffix, body in text.items():
        (out / f"{stem}.{suffix}").write_text(body, encoding="utf-8", newline="\n")
    return out / stem


def run(args, overrides, settings, work: Path) -> dict:
    from app import config, db, rag, stores
    from app.providers import llm
    from . import dataset, metrics, stats

    everything = dataset.load()
    questions = everything[:args.limit] if args.limit is not None else everything
    stamp = time.strftime("%Y%m%d%H%M%S")
    # The app's own merge, so the eval sees the same effective settings; a copy, so DEFAULTS stays untouched.
    cfg = config._merge(copy.deepcopy(config.DEFAULTS),
                        config._merge(settings, {k: v for k, v in overrides.items() if k != "description"}))
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
            CountingLLM.reset()
            t0 = time.perf_counter()
            _, queries, hits = rag.search(q["question"], K, rewrite_question=rewrite)
            ms = (time.perf_counter() - t0) * 1000
            CountingLLM.check(q["id"])
            ranking = [(h["filename"], h["page"]) for h in hits]
            rank = metrics.first_hit(ranking, q["doc"], q["pages"])
            rows.append({"id": q["id"], "tags": q.get("tags", []), "rank": rank, **stats.per_question(rank),
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
            "query_rewrite": bool(cfg["rag"]["query_rewrite"]), "search_mode": cfg["rag"]["search_mode"], "k": K,
            "reranker": _reranker_name(cfg),
            "hardware": _hardware(),
        },
        "dataset": {"questions": len(questions), "documents": sorted({q["doc"] for q in everything}),
                    "sha256": dataset.fingerprint(),
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
        CountingLLM.reset()
        t0 = time.perf_counter()
        r = rag.answer(q["question"])
        ms = (time.perf_counter() - t0) * 1000
        CountingLLM.check(q["id"])
        calls = CountingLLM.calls
        # The judge sees what the answering model saw (the merged chunks); quotes are checked against the whole
        # cited page.
        seen = [(h["filename"], h["page"], h["text"]) for h in r["sources"]]
        verdict = metrics.parse_judgement(judge.complete(metrics.JUDGE_PROMPT, metrics.judge_input(r["answer"], seen)))
        cites = metrics.citations(r["answer"], [dataset.page_texts(h["filename"])[h["page"] - 1] for h in r["sources"]])
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


def _hardware() -> dict:
    """Where the latency was measured: the CPU, and which ONNX Runtime devices the local models could use."""
    import platform
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 check=True).stdout.strip() or cpu
        elif sys.platform.startswith("linux"):
            cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo", encoding="utf-8")
                        if line.startswith("model name")), cpu)
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
    except ImportError:
        providers = []
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip().replace("\n", ", ") or None
    except (OSError, subprocess.SubprocessError):
        gpu = None
    return {"os": platform.platform(terse=True), "cpu": cpu, "cores": os.cpu_count(), "gpu": gpu,
            "onnx_providers": providers}


def _reranker_name(cfg) -> str:
    r = cfg["reranker"]
    p = r["provider"]
    return "off" if p == "off" else f"{p}:{r[p]['model']} top {r['top_n']}"


def _commit() -> str | None:
    """HEAD, with "-dirty" when tracked files have uncommitted changes, so a result that checking out its commit
    can't reproduce says so."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    try:
        head = git("rev-parse", "--short", "HEAD")
        return head and head + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _ms(x):
    return "n/a" if x is None else f"{x:.0f} ms"


def _ci(m: dict, metric: str) -> str:
    """The metric with its 95% interval when the result has one: 62.1% [50.0–73.5], 0.763 [0.671–0.851]."""
    mrr = metric.startswith("mrr")
    fmt = (lambda x: f"{x:.3f}") if mrr else (lambda x: f"{x * 100:.1f}")
    value = fmt(m[metric]) if mrr else _pct(m[metric])
    ci = (m.get("ci95") or {}).get(metric)
    return f"{value} [{fmt(ci[0])}–{fmt(ci[1])}]" if ci else value


def markdown(r: dict) -> str:
    s = r["setup"]
    lines = [
        f"# Retrieval eval: {r['config']} ({r['date']})", "",
        r["description"] or "", "",
        f"Commit `{r['commit']}` · store {s['vector_store']} · embeddings `{s['embeddings']}` · LLM `{s['llm']}` · "
        f"chunks {s['chunk_tokens']}/{s['chunk_overlap_tokens']} tokens · {s.get('search_mode', 'vector')} search · "
        f"rewrite {'on' if s['query_rewrite'] else 'off'} · reranker {s.get('reranker', 'off')} · "
        f"{r['dataset']['questions']} questions over {len(r['dataset']['documents'])} PDFs", "",
        *([f"Measured on {h['cpu']} ({h['cores']} cores, {h['os']})" + (f", GPU {h['gpu']}" if h.get("gpu") else "")
           + "; ONNX Runtime devices: "
           f"{', '.join(h['onnx_providers']) or 'none'}", ""] if (h := s.get("hardware")) else []),
        "| Slice | Questions | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 | p95 | LLM calls/query |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in [("all", r["retrieval"]), *r["by_tag"].items()]:
        lines.append(f"| {name} | {m['questions']} | {_ci(m, 'recall@1')} | {_ci(m, 'recall@5')} | {_pct(m['recall@10'])} "
                     f"| {_ci(m, 'mrr@10')} | {_ms(m['latency_p50_ms'])} | {_ms(m['latency_p95_ms'])} | {m['llm_calls_per_query']} |")
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
