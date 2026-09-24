from __future__ import annotations

import logging
import os

from langchain.tools import tool

from config import CORPUS_FITS_CONTEXT_CHARS, load_settings
from retrieval.bm25 import BM25Index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.types import ScoredChunk
from retrieval.vectorstore import QdrantVectorStore
from tools.result import ToolResult

logger = logging.getLogger(__name__)

# How many candidates each retriever contributes before fusion.
CANDIDATE_POOL = 20

_store: QdrantVectorStore | None = None
_bm25: BM25Index | None = None
_reranker: CrossEncoderReranker | None = None
_corpus_chars: int = 0


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
    global _store, _bm25, _reranker, _corpus_chars

    if _store is None:
        settings = load_settings()
        resume_path = settings.resume_path
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume not found at {resume_path}. Set RESUME_PATH to point at it.")
        store = QdrantVectorStore(settings=settings)
        # Rebuilds when the resume or the chunking strategy changed, not merely
        # when the collection is absent. A stale index answers happily.
        if store.needs_rebuild(resume_path):
            logger.info("Building the resume index from %s", resume_path)
            store.build_from_file(resume_path)
        _store = store  # only now: setup fully succeeded

    if _bm25 is None:
        # The chunks themselves, so BM25 returns the vector store's identifiers
        # and fusion joins on something real.
        chunks = _store.get_all_chunks()
        bm25 = BM25Index()
        bm25.build(chunks)
        _bm25 = bm25
        _corpus_chars = sum(len(c.text) for c in chunks)

    if _reranker is None:
        _reranker = CrossEncoderReranker()

    return _store, _bm25, _reranker


def search_resume(query: str, top_n: int | None = None, strategy: str | None = None) -> list[ScoredChunk]:
    """Retrieve resume chunks for a query.

    The strategy is configuration, not conviction. On the current corpus plain
    dense search wins outright (see eval/results/ablation-retrieval.json); the
    sparse, fused and reranked paths stay available because that ranking is a
    property of this corpus, not a law, and is expected to change as it grows.

    Raises on failure rather than returning an empty list, so a caller that
    needs to distinguish "nothing relevant" from "the store is unreachable"
    can. See ``get_resume_info_result`` for the typed, non-raising wrapper.
    """
    store, bm25, reranker = _init_retrieval()
    settings = load_settings()
    top_n = top_n or settings.retrieval_top_n
    strategy = strategy or settings.retrieval_strategy

    if strategy == "auto":
        if _corpus_chars and _corpus_chars <= CORPUS_FITS_CONTEXT_CHARS:
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
