---
paths:
  - "retrieval/**"
  - "tools/resume_tool.py"
  - "tests/integration/**"
  - "tests/**/test_resume_tool.py"
  - "tests/**/test_fusion.py"
  - "tests/**/test_chunking.py"
  - "tests/**/test_reranker.py"
  - "tests/**/test_retrieval_cache.py"
---

# Hybrid RAG pipeline (resume path only)

The `resume` handler calls `tools/resume_tool.py:search_resume()`, which orchestrates three lazy-initialised singletons via `_init_retrieval()`:

- **`QdrantVectorStore`** (`retrieval/vectorstore.py`) — dense cosine search using `text-embedding-3-small`. Collection `resume_chunks` is built from `RESUME_PATH` (default `data/resume.md`) on first query, or whenever `needs_rebuild()` detects the source bytes or `CHUNKER_VERSION` changed. Markdown is chunked structurally on headings (`retrieval/chunking.py`); a PDF falls back to `SemanticChunker`, since extraction strips its headings.
- **`BM25Index`** (`retrieval/bm25.py`) — sparse keyword search, built from the vector store's own chunks so its returned ids are the store's ids (fusion joins on real identity, not list position).
- **`CrossEncoderReranker`** (`retrieval/reranker.py`) — `cross-encoder/ms-marco-MiniLM-L-6-v2` (downloads ~90 MB on first run, then cached).

`RETRIEVAL_STRATEGY` selects how these combine: `auto` (default — the whole corpus, reranked, while it fits `CORPUS_FITS_CONTEXT_CHARS`; plain dense search once it doesn't), `dense`, `sparse`, `rrf`, or `rrf_rerank` (RRF fusion of top-`CANDIDATE_POOL` from each, then cross-encoder rerank). See `eval/results/ablation-retrieval.json` and the README's ablation table before assuming hybrid beats dense on a given corpus size — on the current 7 KB resume it does not.

`_init_retrieval()` commits `_store`/`_bm25`/`_reranker` to module globals only after each stage fully succeeds, so a transient failure (a Qdrant container not yet ready) is retried on the next call instead of poisoning the process. `retrieval/cache.py` (`RetrievalCache`) caches query embeddings and full retrieval results, same Redis-or-in-process pattern as session memory (shared connection logic in `redis_backend.py`), scoped by the corpus fingerprint so a rebuilt index can't serve a stale cached answer.

## Constraints

- The `_store`, `_bm25`, `_reranker`, and `_cache` singletons in `resume_tool.py` are module-level globals, initialised lazily on first query. Each is committed to its global only after that stage's setup fully succeeds — a transient failure (a Qdrant container not yet ready) must be retried on the next call, not permanently cached as broken. `tests/unit/test_resume_tool.py` guards this against regressing; it is the fix for a real incident where a CI service-container race silently broke retrieval for an entire scheduled run.
- The Qdrant index rebuilds when `needs_rebuild()` finds the source bytes or `retrieval/vectorstore.py:CHUNKER_VERSION` changed, not merely when the collection is absent. A stale index answers happily from the previous corpus.
- The Qdrant client is built by `retrieval/backends.py:create_qdrant_client()` from `Settings`, never from hardcoded coordinates. `QdrantVectorStore` takes an injected client and an optional `RetrievalCache`, which is how the integration tests run without a server or a cache.
