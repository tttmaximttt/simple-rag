# Retrieval eval: hybrid-rerank-gpu (2026-11-05)

Hybrid search, then the local cross-encoder (bge-reranker-v2-m3, fp32) re-scores the top 30 candidates. No query rewrite. For a GPU: the int8 model runs slower on CUDA than on a CPU.

Commit `2e85055` · store chroma · embeddings `local:BAAI/bge-small-en-v1.5` · LLM `anthropic:claude-sonnet-5-5` · chunks 400/60 tokens · hybrid search · rewrite off · reranker local:BAAI/bge-reranker-v2-m3 top 30 · 66 questions over 4 PDFs

Measured on Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (32 cores, Windows-11), GPU NVIDIA GeForce RTX 4080; ONNX Runtime devices: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider

| Slice | Questions | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 | p95 | LLM calls/query |
|---|---|---|---|---|---|---|---|---|
| all | 66 | 84.9% | 100.0% | 100.0% | 0.918 | 394 ms | 420 ms | 0.0 |
| cross-language | 19 | 84.2% | 100.0% | 100.0% | 0.921 | 401 ms | 429 ms | 0.0 |
| exact | 27 | 88.9% | 100.0% | 100.0% | 0.944 | 387 ms | 423 ms | 0.0 |
| untagged | 28 | 82.1% | 100.0% | 100.0% | 0.896 | 376 ms | 411 ms | 0.0 |
