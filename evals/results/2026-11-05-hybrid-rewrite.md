# Retrieval eval: hybrid-rewrite (2026-11-05)

Hybrid search with the LLM query rewrite: every rewritten query is searched by vector and keyword.

Commit `ff4641a` · store chroma · embeddings `local:BAAI/bge-small-en-v1.5` · LLM `anthropic:claude-sonnet-5-5` · chunks 400/60 tokens · hybrid search · rewrite on · 66 questions over 4 PDFs

| Slice | Questions | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 | p95 | LLM calls/query |
|---|---|---|---|---|---|---|---|---|
| all | 66 | 62.1% | 93.9% | 98.5% | 0.766 | 2011 ms | 2952 ms | 1.0 |
| cross-language | 19 | 52.6% | 89.5% | 100.0% | 0.699 | 2079 ms | 2646 ms | 1.0 |
| exact | 27 | 59.3% | 92.6% | 96.3% | 0.752 | 2007 ms | 3034 ms | 1.0 |
| untagged | 28 | 67.9% | 100.0% | 100.0% | 0.809 | 1985 ms | 2815 ms | 1.0 |

Not in the top 5:

- `rpi-01` (rank 7): got raspberry-pi-hardware.pdf p6, raspberry-pi-hardware.pdf p8, raspberry-pi-hardware.pdf p4, raspberry-pi-hardware.pdf p10, raspberry-pi-hardware.pdf p5
- `rpi-02` (rank >10): got raspberry-pi-hardware.pdf p9, raspberry-pi-hardware.pdf p6, raspberry-pi-hardware.pdf p8, raspberry-pi-hardware.pdf p4, raspberry-pi-hardware.pdf p10
- `mdn-04` (rank 6): got mdn-http-es.pdf p8, pep8-style-guide.pdf p22, mdn-http-es.pdf p5, mdn-http-es.pdf p2, mdn-http-es.pdf p3
- `mdn-07` (rank 9): got mdn-http-es.pdf p12, mdn-http-es.pdf p19, mdn-http-es.pdf p14, mdn-http-es.pdf p18, mdn-http-es.pdf p20
