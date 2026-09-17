from __future__ import annotations

import os

from langchain.tools import tool

from config import load_settings
from retrieval.bm25 import BM25Index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.types import ScoredChunk
from retrieval.vectorstore import QdrantVectorStore

_store: QdrantVectorStore | None = None
_bm25: BM25Index | None = None
_reranker: CrossEncoderReranker | None = None


def _init_retrieval() -> tuple[QdrantVectorStore, BM25Index, CrossEncoderReranker]:
    global _store, _bm25, _reranker

    if _store is None:
        settings = load_settings()
        _store = QdrantVectorStore(settings=settings)
        if not _store.collection_exists():
            resume_path = settings.resume_path
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"Resume not found at {resume_path}. Set RESUME_PATH to point at it.")
            _store.build_from_file(resume_path)

    if _bm25 is None:
        all_chunks = _store.get_all_chunks()
        _bm25 = BM25Index()
        _bm25.build([c.text for c in all_chunks])

    if _reranker is None:
        _reranker = CrossEncoderReranker()

    return _store, _bm25, _reranker


def search_resume(query: str, top_n: int = 3) -> list[ScoredChunk]:
    """Hybrid retrieval over the resume: dense + BM25, fused by RRF, then reranked.

    Returns the chunks themselves. Callers that need per-chunk information — the
    evaluation harness measuring ranking quality, or a UI showing citations — use
    this; the tool below flattens it to a string for the model.
    """
    store, bm25, reranker = _init_retrieval()
    dense = store.dense_search(query, top_k=20)
    sparse = bm25.search(query, top_k=20)
    fused = reciprocal_rank_fusion(dense, sparse)
    return reranker.rerank(query, fused, top_n=top_n)


def format_chunks(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as the context block handed to the model."""
    if not chunks:
        return "No relevant information found in the resume."
    return "\n\n".join(c.text for c in chunks)


@tool
def get_resume_info(query: str) -> str:
    """Search the resume using hybrid retrieval (dense + BM25 + reranking)."""
    try:
        return format_chunks(search_resume(query))
    except Exception as e:
        return f"Error searching resume: {e}"
