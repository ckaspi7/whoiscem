from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi

from retrieval.types import ScoredChunk


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: List[str] = []

    def build(self, chunks: List[str]) -> None:
        self._chunks = chunks
        tokenized = [self._tokenize(c) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> List[ScoredChunk]:
        if self._bm25 is None or not self._chunks:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk_id=str(idx), text=self._chunks[idx], score=float(score))
            for idx, score in ranked
            if score > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())
