"""Serve the resume straight from disk when the vector store is unreachable.

Qdrant is a network dependency; the resume is a small file in the repo. When
the store cannot be reached (a proxy that rejects the domain, an outage, a cold
container), answering from the file beats answering from nothing, and for a
corpus this small it is the same context the healthy "auto" strategy returns.

No embeddings and no network: chunks come from the same structural chunker the
index is built with, and ``dense_search`` degrades to BM25 over those chunks.
"""

from __future__ import annotations

from pathlib import Path

from retrieval.bm25 import BM25Index
from retrieval.chunking import chunk_markdown
from retrieval.types import ScoredChunk

_MARKDOWN_SUFFIXES = (".md", ".markdown")


class DiskResumeStore:
    """The read surface of QdrantVectorStore that tools/resume_tool.py uses."""

    def __init__(self, path: str) -> None:
        # A PDF has no headings after extraction and needs the embedding-based
        # chunker, which needs the network this fallback exists to avoid.
        if not path.lower().endswith(_MARKDOWN_SUFFIXES):
            raise ValueError(f"disk fallback needs a Markdown resume, not {Path(path).suffix or path!r}")
        text = Path(path).read_text(encoding="utf-8")
        self._chunks = [
            ScoredChunk(chunk_id=str(i), text=chunk.text, score=0.0, section=chunk.section)
            for i, chunk in enumerate(chunk_markdown(text))
        ]
        self.bm25 = BM25Index()
        self.bm25.build(self._chunks)

    @property
    def corpus_chars(self) -> int:
        return sum(len(chunk.text) for chunk in self._chunks)

    def get_all_chunks(self) -> list[ScoredChunk]:
        return list(self._chunks)

    def dense_search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        return self.bm25.search(query, top_k=top_k)
