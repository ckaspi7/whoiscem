from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from retrieval.types import ScoredChunk


class BM25Index:
    """Sparse keyword search over the same chunks the vector store holds.

    Indexes ScoredChunk rather than plain strings so the identifiers it returns
    are the vector store's. It previously keyed on list position while the
    vector store keyed on the Qdrant point id, and reciprocal rank fusion joined
    the two on string equality — correct only while the point ids happened to be
    assigned in list order. Re-chunking or any non-sequential id would have made
    RRF fuse unrelated chunks, silently, in the project's flagship feature.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[ScoredChunk] = []

    def build(self, chunks: Sequence[ScoredChunk]) -> None:
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([self._tokenize(c.text) for c in self._chunks]) if self._chunks else None

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        if self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            ScoredChunk(
                chunk_id=self._chunks[idx].chunk_id,
                text=self._chunks[idx].text,
                score=float(score),
                section=self._chunks[idx].section,
            )
            for idx, score in ranked
            if score > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())
