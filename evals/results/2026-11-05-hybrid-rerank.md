# Retrieval eval: hybrid-rerank (2026-11-05)

Hybrid search, then the local cross-encoder (bge-reranker-v2-m3, int8) re-scores the top 30 candidates. No query rewrite.

Commit `fdd7c03` · store chroma · embeddings `local:BAAI/bge-small-en-v1.5` · LLM `anthropic:claude-sonnet-5-5` · chunks 400/60 tokens · hybrid search · rewrite off · reranker local:BAAI/bge-reranker-v2-m3-int8 top 30 · 66 questions over 4 PDFs

Measured on Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (32 cores, Windows-11), GPU NVIDIA GeForce RTX 4080; ONNX Runtime devices: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider

| Slice | Questions | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 | p95 | LLM calls/query |
|---|---|---|---|---|---|---|---|---|
| all | 66 | 84.9% | 100.0% | 100.0% | 0.920 | 4312 ms | 4652 ms | 0.0 |
| cross-language | 19 | 84.2% | 100.0% | 100.0% | 0.921 | 4361 ms | 4667 ms | 0.0 |
| exact | 27 | 88.9% | 100.0% | 100.0% | 0.944 | 4319 ms | 4618 ms | 0.0 |
| untagged | 28 | 82.1% | 100.0% | 100.0% | 0.902 | 4275 ms | 4604 ms | 0.0 |
