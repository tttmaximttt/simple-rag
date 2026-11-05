# Retrieval eval: hybrid-rewrite (2026-11-05)

Hybrid search with the LLM query rewrite: every rewritten query is searched by vector and keyword.

Commit `fdd7c03` · store chroma · embeddings `local:BAAI/bge-small-en-v1.5` · LLM `anthropic:claude-sonnet-5-5` · chunks 400/60 tokens · hybrid search · rewrite on · reranker off · 66 questions over 4 PDFs

Measured on Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (32 cores, Windows-11), GPU NVIDIA GeForce RTX 4080; ONNX Runtime devices: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider

| Slice | Questions | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 | p95 | LLM calls/query |
|---|---|---|---|---|---|---|---|---|
| all | 66 | 62.1% | 95.5% | 98.5% | 0.763 | 2157 ms | 2684 ms | 1.0 |
| cross-language | 19 | 52.6% | 94.7% | 100.0% | 0.700 | 2225 ms | 2931 ms | 1.0 |
| exact | 27 | 59.3% | 92.6% | 96.3% | 0.752 | 2146 ms | 3021 ms | 1.0 |
| untagged | 28 | 67.9% | 100.0% | 100.0% | 0.800 | 2114 ms | 2449 ms | 1.0 |

Not in the top 5:

- `rpi-01` (rank 7): got raspberry-pi-hardware.pdf p6, raspberry-pi-hardware.pdf p8, raspberry-pi-hardware.pdf p4, raspberry-pi-hardware.pdf p10, raspberry-pi-hardware.pdf p5
- `rpi-02` (rank >10): got raspberry-pi-hardware.pdf p9, raspberry-pi-hardware.pdf p6, raspberry-pi-hardware.pdf p4, raspberry-pi-hardware.pdf p5, raspberry-pi-hardware.pdf p10
- `mdn-07` (rank 10): got mdn-http-es.pdf p12, mdn-http-es.pdf p19, mdn-http-es.pdf p14, mdn-http-es.pdf p18, mdn-http-es.pdf p13
