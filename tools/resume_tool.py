from __future__ import annotations

import logging
import os
import time

from langchain.tools import tool

from config import CORPUS_FITS_CONTEXT_CHARS, load_settings
from retrieval.bm25 import BM25Index
from retrieval.cache import RetrievalCache
from retrieval.fallback import DiskResumeStore
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.types import ScoredChunk
from retrieval.vectorstore import QdrantVectorStore
from retrieval.vectorstore import fingerprint as corpus_fingerprint
from tools.result import ToolResult

logger = logging.getLogger(__name__)

# How many candidates each retriever contributes before fusion.
CANDIDATE_POOL = 20

# While Qdrant is failing it is retried at most this often, so an outage costs
# one connection attempt per interval rather than one per question.
QDRANT_RETRY_SECONDS = 30.0

_store: QdrantVectorStore | None = None
_bm25: BM25Index | None = None
_reranker: CrossEncoderReranker | None = None
_cache: RetrievalCache | None = None
_corpus_chars: int = 0
_corpus_fp: str = ""

# The disk fallback is deliberately NOT the _store singleton: committing it
# there would turn one transient Qdrant failure into a permanent downgrade,
# which is the poisoning bug _init_retrieval's contract exists to prevent.
_qdrant_failed_at: float | None = None
_qdrant_last_error: Exception | None = None
_disk_store: DiskResumeStore | None = None
_disk_store_fp: str = ""


def _init_retrieval() -> tuple[QdrantVectorStore, BM25Index, CrossEncoderReranker]:
    """Build the retrieval singletons, or reuse them if already built.

    Each stage is committed to the module-level globals only after it fully
    succeeds. A fresh Qdrant container that is not yet accepting connections —
    exactly what a CI service container is on its first request — must degrade
    that one call, not leave a half-built store that every later call in the
    process reuses without ever retrying.

    The bug this replaces: ``_bm25 = BM25Index()`` used to run before the
    network call that fills it. If that call failed, ``_bm25`` was left
    non-None but never built, so every later query silently returned no
    results — no error, just quietly wrong — because ``_bm25 is None`` was
    false from then on.
    """
    global _store, _bm25, _reranker, _cache, _corpus_chars, _corpus_fp

    if _store is None:
        settings = load_settings()
        resume_path = settings.resume_path
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume not found at {resume_path}. Set RESUME_PATH to point at it.")
        # RetrievalCache never raises — connect() degrades to in-process or
        # disabled internally — so building it before the store cannot itself
        # trigger the retry-on-failure path this function exists to guarantee.
        cache = RetrievalCache()
        store = QdrantVectorStore(settings=settings, cache=cache)
        # Rebuilds when the resume or the chunking strategy changed, not merely
        # when the collection is absent. A stale index answers happily.
        if store.needs_rebuild(resume_path):
            logger.info("Building the resume index from %s", resume_path)
            store.build_from_file(resume_path)
        _store = store  # only now: setup fully succeeded
        _cache = cache
        _corpus_fp = corpus_fingerprint(resume_path)

    if _bm25 is None:
        # The chunks themselves, so BM25 returns the vector store's identifiers
        # and fusion joins on something real.
        chunks = _store.get_all_chunks()
        bm25 = BM25Index()
        bm25.build(chunks)
        _bm25 = bm25
        _corpus_chars = sum(len(c.text) for c in chunks)

    return _store, _bm25, _get_reranker()


def _get_reranker() -> CrossEncoderReranker:
    """The shared reranker, committed only once it has actually loaded."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def search_resume(query: str, top_n: int | None = None, strategy: str | None = None) -> list[ScoredChunk]:
    """Retrieve resume chunks for a query.

    The strategy is configuration, not conviction. On the current corpus plain
    dense search wins outright (see eval/results/ablation-retrieval.json); the
    sparse, fused and reranked paths stay available because that ranking is a
    property of this corpus, not a law, and is expected to change as it grows.

    When the vector store is unreachable the resume is served from disk
    instead (see ``_search_from_disk``), so an outage degrades retrieval
    quality rather than silently answering from nothing. Raises only when
    neither path can work, so a caller can still tell "nothing relevant" from
    "the knowledge base is unreachable". See ``get_resume_info_result`` for the
    typed, non-raising wrapper.
    """
    global _qdrant_failed_at, _qdrant_last_error
    settings = load_settings()
    top_n = top_n or settings.retrieval_top_n
    strategy = strategy or settings.retrieval_strategy

    if not _qdrant_cooling_down():
        try:
            result = _search_via_store(query, top_n, strategy)
        except Exception as exc:
            _qdrant_failed_at = time.monotonic()
            _qdrant_last_error = exc
            logger.warning("Vector store unavailable (%s); answering from the resume on disk", exc)
        else:
            if _qdrant_failed_at is not None:
                logger.info("Vector store reachable again")
            _qdrant_failed_at = None
            _qdrant_last_error = None
            return result

    return _search_from_disk(query, top_n, strategy, _qdrant_last_error)


def _qdrant_cooling_down() -> bool:
    return _qdrant_failed_at is not None and time.monotonic() - _qdrant_failed_at < QDRANT_RETRY_SECONDS


def _search_via_store(query: str, top_n: int, strategy: str) -> list[ScoredChunk]:
    """The normal path: Qdrant plus BM25 plus the reranker.

    The full result is cached per (strategy, top_n, query), scoped to the
    corpus fingerprint so a rebuilt index — a new resume, a changed chunker —
    cannot serve a stale answer: the fingerprint in the key simply stops
    matching. "auto" is cached as itself rather than its resolved branch,
    which is safe because it always resolves the same way for a given corpus.
    """
    store, bm25, reranker = _init_retrieval()

    cache_key = f"{strategy}:{top_n}:{query}"
    if _cache:
        cached = _cache.get_retrieval(_corpus_fp, cache_key)
        if cached is not None:
            return cached

    result = _dispatch(query, top_n, strategy, store, bm25, reranker)

    if _cache:
        _cache.set_retrieval(_corpus_fp, cache_key, result)
    return result


def _search_from_disk(query: str, top_n: int, strategy: str, cause: Exception | None) -> list[ScoredChunk]:
    """Answer from the resume file when the vector store cannot be reached.

    Uncached on purpose: a degraded result must not be stored where a healthy
    index's result would later be looked up. If the file cannot serve as a
    fallback either, the original store error is the one worth surfacing.
    """
    settings = load_settings()
    try:
        disk = _get_disk_store(settings.resume_path)
        reranker = _get_reranker()
    except Exception as exc:
        logger.warning("Disk fallback unavailable: %s", exc)
        if cause is None:
            raise
        raise cause from None
    return _dispatch(query, top_n, strategy, disk, disk.bm25, reranker, corpus_chars=disk.corpus_chars)


def _get_disk_store(path: str) -> DiskResumeStore:
    global _disk_store, _disk_store_fp
    fingerprint = corpus_fingerprint(path)
    if _disk_store is None or _disk_store_fp != fingerprint:
        _disk_store = DiskResumeStore(path)
        _disk_store_fp = fingerprint
    return _disk_store


def _dispatch(
    query: str,
    top_n: int,
    strategy: str,
    store: QdrantVectorStore | DiskResumeStore,
    bm25: BM25Index,
    reranker: CrossEncoderReranker,
    corpus_chars: int | None = None,
) -> list[ScoredChunk]:
    """The actual retrieval logic, uncached — callers own the cache.

    ``corpus_chars`` defaults to the size recorded when the vector store was
    initialised; the disk fallback passes its own, since that initialisation
    never ran.
    """
    chars = _corpus_chars if corpus_chars is None else corpus_chars
    if strategy == "auto":
        if chars and chars <= CORPUS_FITS_CONTEXT_CHARS:
            # Everything, still ranked: the model reads it all, and the most
            # relevant material first is worth something.
            all_chunks = store.get_all_chunks()
            return reranker.rerank(query, all_chunks, top_n=len(all_chunks))
        strategy = "dense"

    if strategy == "dense":
        return store.dense_search(query, top_k=top_n)
    if strategy == "sparse":
        return bm25.search(query, top_k=top_n)

    fused = reciprocal_rank_fusion(
        store.dense_search(query, top_k=CANDIDATE_POOL),
        bm25.search(query, top_k=CANDIDATE_POOL),
    )
    if strategy == "rrf":
        return fused[:top_n]
    return reranker.rerank(query, fused, top_n=top_n)


def format_chunks(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as the context block handed to the model."""
    if not chunks:
        return "No relevant information found in the resume."
    return "\n\n".join(c.text for c in chunks)


def get_resume_info_result(query: str) -> ToolResult:
    """Typed retrieval: content on success, an error that never becomes context."""
    try:
        return ToolResult.success(format_chunks(search_resume(query)))
    except Exception as e:
        logger.warning("Resume retrieval failed: %s", e)
        return ToolResult.failure(str(e))


@tool
def get_resume_info(query: str) -> str:
    """Search the resume using hybrid retrieval (dense + BM25 + reranking)."""
    return get_resume_info_result(query).as_tool_string("Error searching resume")
