from __future__ import annotations

import os

from langchain.tools import tool

from config import load_settings
from retrieval.bm25 import BM25Index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
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


@tool
def get_resume_info(query: str) -> str:
    """Search the resume using hybrid retrieval (dense + BM25 + reranking)."""
    try:
        store, bm25, reranker = _init_retrieval()
        dense = store.dense_search(query, top_k=20)
        sparse = bm25.search(query, top_k=20)
        fused = reciprocal_rank_fusion(dense, sparse)
        top = reranker.rerank(query, fused, top_n=3)

        if not top:
            return "No relevant information found in the resume."
        return "\n\n".join(c.text for c in top)
    except Exception as e:
        return f"Error searching resume: {e}"
