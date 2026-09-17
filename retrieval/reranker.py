from __future__ import annotations

from retrieval.types import ScoredChunk

# Module-level import so tests can patch `retrieval.reranker._CrossEncoder`
# without needing sentence_transformers installed.
try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
except ImportError:
    _CrossEncoder = None  # type: ignore[assignment,misc]


class CrossEncoderReranker:
    """Reranks candidates with a local cross-encoder model (no API cost).

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB, downloaded on first use).
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        if _CrossEncoder is None:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")
        self._model = _CrossEncoder(self.MODEL_NAME)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int = 3) -> list[ScoredChunk]:
        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                section=chunk.section,
            )
            for chunk, score in ranked[:top_n]
        ]
