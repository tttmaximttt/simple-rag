# Retrieval eval: hybrid-rewrite (2026-11-05)

Hybrid search with the LLM query rewrite: every rewritten query is searched by vector and keyword.

Commit `1c7e40a` · store chroma · embeddings `local:BAAI/bge-small-en-v1.5` · LLM `anthropic:claude-sonnet-5-5` · chunks 400/60 tokens · hybrid search · rewrite on · 66 questions over 4 PDFs

| Slice | Questions | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 | p95 | LLM calls/query |
|---|---|---|---|---|---|---|---|---|
| all | 66 | 63.6% | 95.5% | 98.5% | 0.775 | 1947 ms | 2506 ms | 1.0 |
| cross-language | 19 | 57.9% | 94.7% | 100.0% | 0.730 | 1940 ms | 2330 ms | 1.0 |
| exact | 27 | 59.3% | 92.6% | 96.3% | 0.752 | 1989 ms | 2628 ms | 1.0 |
| untagged | 28 | 67.9% | 100.0% | 100.0% | 0.809 | 1846 ms | 2389 ms | 1.0 |

Not in the top 5:

- `rpi-01` (rank 7): got raspberry-pi-hardware.pdf p6, raspberry-pi-hardware.pdf p8, raspberry-pi-hardware.pdf p4, raspberry-pi-hardware.pdf p9, raspberry-pi-hardware.pdf p5
- `rpi-02` (rank >10): got raspberry-pi-hardware.pdf p9, raspberry-pi-hardware.pdf p6, raspberry-pi-hardware.pdf p4, raspberry-pi-hardware.pdf p10, raspberry-pi-hardware.pdf p8
- `mdn-07` (rank 8): got mdn-http-es.pdf p12, mdn-http-es.pdf p19, mdn-http-es.pdf p18, mdn-http-es.pdf p14, mdn-http-es.pdf p20
