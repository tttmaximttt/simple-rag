# Retrieval eval: vector-rewrite (2026-11-05)

Current vector search with the LLM query rewrite (one LLM call per question).

Commit `3b78b73` · store chroma · embeddings `local:BAAI/bge-small-en-v1.5` · LLM `anthropic:claude-sonnet-5-5` · chunks 400/60 tokens · rewrite on · 66 questions over 4 PDFs

| Slice | Questions | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 | p95 | LLM calls/query |
|---|---|---|---|---|---|---|---|---|
| all | 66 | 51.5% | 89.4% | 93.9% | 0.693 | 1897 ms | 2368 ms | 1.0 |
| cross-language | 19 | 47.4% | 89.5% | 89.5% | 0.660 | 1969 ms | 2247 ms | 1.0 |
| exact | 27 | 44.4% | 92.6% | 96.3% | 0.672 | 1897 ms | 2301 ms | 1.0 |
| untagged | 28 | 57.1% | 89.3% | 96.4% | 0.725 | 1841 ms | 2462 ms | 1.0 |

Not in the top 5:

- `rpi-01` (rank 10): got raspberry-pi-hardware.pdf p6, raspberry-pi-hardware.pdf p10, raspberry-pi-hardware.pdf p4, raspberry-pi-hardware.pdf p5, raspberry-pi-hardware.pdf p8
- `rpi-02` (rank >10): got raspberry-pi-hardware.pdf p6, raspberry-pi-hardware.pdf p9, mdn-http-es.pdf p3, mdn-http-es.pdf p40, mdn-http-es.pdf p1
- `nist-13` (rank >10): got nist-sp800-63b.pdf p11, nist-sp800-63b.pdf p14, mdn-http-es.pdf p1, mdn-http-es.pdf p3, nist-sp800-63b.pdf p7
- `nist-14` (rank 7): got nist-sp800-63b.pdf p27, nist-sp800-63b.pdf p29, nist-sp800-63b.pdf p9, nist-sp800-63b.pdf p10, nist-sp800-63b.pdf p18
- `nist-17` (rank 7): got nist-sp800-63b.pdf p26, nist-sp800-63b.pdf p3, nist-sp800-63b.pdf p5, mdn-http-es.pdf p1, nist-sp800-63b.pdf p1
- `mdn-04` (rank >10): got mdn-http-es.pdf p5, mdn-http-es.pdf p20, mdn-http-es.pdf p11, mdn-http-es.pdf p2, mdn-http-es.pdf p8
- `mdn-07` (rank >10): got mdn-http-es.pdf p14, mdn-http-es.pdf p19, mdn-http-es.pdf p20, mdn-http-es.pdf p13, mdn-http-es.pdf p21
